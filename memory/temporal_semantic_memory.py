"""
Temporal + Semantic + Entity Memory System for AI Agents.

This implements a sophisticated memory architecture that combines:
1. Temporal links: Memories connected by time proximity
2. Semantic links: Memories connected by meaning/similarity
3. Entity links: Memories connected by shared entities (PERSON, ORG, etc.)
4. Spreading activation: Search through the graph with activation decay
5. Dynamic weighting: Recency and frequency-based importance
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import asyncpg
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import uuid
import logging

from .utils import (
    extract_facts,
    calculate_recency_weight,
    calculate_frequency_weight,
)
from .entity_resolver import EntityResolver


def utcnow():
    """Get current UTC time with timezone info."""
    return datetime.now(timezone.utc)


# Logger for memory system
logger = logging.getLogger(__name__)

# Global process pool for parallel embedding generation
# Each process loads its own copy of the embedding model
# This provides TRUE parallelism for CPU-bound embedding operations
_PROCESS_POOL = None
_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Process-local model cache (one per worker process)
_worker_model = None


def _get_worker_model():
    """Get or load the embedding model in worker process."""
    global _worker_model
    if _worker_model is None:
        _worker_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _worker_model


def _encode_batch_worker(texts: List[str]) -> List[List[float]]:
    """
    Worker function for process pool - encodes texts to embeddings.

    This function runs in a separate process and loads its own model.
    """
    model = _get_worker_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


def _get_process_pool():
    """Get or create the global process pool."""
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        # Use 4 worker processes for true parallelism
        # Adjust based on your CPU cores (each process loads ~500MB model)
        _PROCESS_POOL = ProcessPoolExecutor(max_workers=4)
    return _PROCESS_POOL


class TemporalSemanticMemory:
    """
    Advanced memory system using temporal and semantic linking with PostgreSQL.
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        """
        Initialize the temporal + semantic memory system.

        Args:
            db_url: PostgreSQL connection URL (postgresql://user:pass@host:port/dbname)
            embedding_model: Name of the SentenceTransformer model to use
        """
        load_dotenv()

        # Initialize PostgreSQL connection URL
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError(
                "Database URL not found. "
                "Set DATABASE_URL environment variable."
            )

        # Connection pool (created lazily on first use)
        self._pool = None
        self._pool_lock = asyncio.Lock()

        # Initialize entity resolver (will be created with pool)
        self.entity_resolver = None

        # Initialize local embedding model (384 dimensions)
        logger.info(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(f"Model loaded (embedding dim: {self.embedding_model.get_sentence_embedding_dimension()})")

        # Background queue for access count updates (to avoid blocking searches)
        self._access_count_queue = asyncio.Queue()
        self._access_count_worker_task = None
        self._shutdown_event = asyncio.Event()

    async def _access_count_worker(self):
        """Background worker that processes access count updates in batches."""
        pool = self._pool  # Pool is guaranteed to exist when worker starts

        while not self._shutdown_event.is_set():
            try:
                # Collect updates for up to 1 second or 1000 items
                updates = {}
                deadline = asyncio.get_event_loop().time() + 1.0

                while len(updates) < 1000 and asyncio.get_event_loop().time() < deadline:
                    try:
                        # Wait for items with short timeout
                        remaining_time = max(0.1, deadline - asyncio.get_event_loop().time())
                        node_ids = await asyncio.wait_for(
                            self._access_count_queue.get(),
                            timeout=remaining_time
                        )
                        # Deduplicate by adding to set
                        for node_id in node_ids:
                            updates[node_id] = True
                    except asyncio.TimeoutError:
                        break

                # Process batch if we have updates
                if updates:
                    node_id_list = list(updates.keys())
                    try:
                        # Convert string UUIDs to UUID type for faster matching
                        uuid_list = [uuid.UUID(nid) for nid in node_id_list]
                        async with pool.acquire() as conn:
                            await conn.execute(
                                "UPDATE memory_units SET access_count = access_count + 1 WHERE id = ANY($1::uuid[])",
                                uuid_list
                            )
                    except Exception as e:
                        logger.error(f"Access count worker: Error updating access counts: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Access count worker: Unexpected error: {e}")
                await asyncio.sleep(1)  # Backoff on error

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create the connection pool (lazy initialization)."""
        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:
                    self._pool = await asyncpg.create_pool(
                        self.db_url,
                        min_size=2,
                        max_size=10,
                        command_timeout=60,
                        statement_cache_size=0  # Disable prepared statement cache
                    )
                    # Initialize entity resolver with pool
                    if self.entity_resolver is None:
                        self.entity_resolver = EntityResolver(self._pool)

        # Start access count worker (outside lock, after pool is created)
        if self._access_count_worker_task is None and self._pool is not None:
            self._access_count_worker_task = asyncio.create_task(self._access_count_worker())

        return self._pool

    async def close(self):
        """Close the connection pool and shutdown background workers."""
        # Signal shutdown to worker
        self._shutdown_event.set()

        # Cancel and wait for worker task
        if self._access_count_worker_task is not None:
            self._access_count_worker_task.cancel()
            try:
                await self._access_count_worker_task
            except asyncio.CancelledError:
                pass

        # Close pool
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using local SentenceTransformer model.

        Args:
            text: Text to embed

        Returns:
            384-dimensional embedding vector (bge-small-en-v1.5)
        """
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")

    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using local model in parallel.

        Uses a ProcessPoolExecutor to achieve TRUE parallelism for CPU-bound
        embedding generation. Each worker process loads its own model copy.

        When multiple put_async calls run in parallel, each can generate
        embeddings concurrently in separate processes (no GIL contention).

        Args:
            texts: List of texts to embed

        Returns:
            List of 384-dimensional embeddings in same order as input texts
        """
        try:
            # Run in process pool for true parallelism
            loop = asyncio.get_event_loop()
            pool = _get_process_pool()
            embeddings = await loop.run_in_executor(
                pool,
                _encode_batch_worker,
                texts
            )
            return embeddings
        except Exception as e:
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")

    async def _find_duplicate_facts_batch(
        self,
        conn,
        agent_id: str,
        texts: List[str],
        embeddings: List[List[float]],
        event_date: datetime,
        time_window_hours: int = 24,
        similarity_threshold: float = 0.95
    ) -> List[bool]:
        """
        Check which facts are duplicates using semantic similarity + temporal window.

        For each new fact, checks if a semantically similar fact already exists
        within the time window. Uses pgvector cosine similarity for efficiency.

        Args:
            conn: Database connection
            agent_id: Agent identifier
            texts: List of fact texts to check
            embeddings: Corresponding embeddings
            event_date: Event date for temporal filtering
            time_window_hours: Hours before/after event_date to search (default: 24)
            similarity_threshold: Minimum cosine similarity to consider duplicate (default: 0.95)

        Returns:
            List of booleans - True if fact is a duplicate (should skip), False if new
        """
        is_duplicate = []

        time_lower = event_date - timedelta(hours=time_window_hours)
        time_upper = event_date + timedelta(hours=time_window_hours)

        for text, embedding in zip(texts, embeddings):
            # Query for similar facts within time window
            # Convert embedding list to string for asyncpg vector type
            embedding_str = str(embedding)
            result = await conn.fetchrow(
                """
                SELECT id, text, 1 - (embedding <=> $1::vector) AS similarity
                FROM memory_units
                WHERE agent_id = $2
                  AND event_date BETWEEN $3 AND $4
                  AND 1 - (embedding <=> $1::vector) > $5
                ORDER BY similarity DESC
                LIMIT 1
                """,
                embedding_str, agent_id, time_lower, time_upper, similarity_threshold
            )

            if result:
                is_duplicate.append(True)
            else:
                is_duplicate.append(False)

        return is_duplicate

    def put(
        self,
        agent_id: str,
        content: str,
        context: str = "",
        event_date: Optional[datetime] = None,
    ) -> List[str]:
        """
        Store content as memory units (synchronous wrapper).

        This is a synchronous wrapper around put_async() for convenience.
        For best performance, use put_async() directly.

        Args:
            agent_id: Unique identifier for the agent
            content: Text content to store
            context: Context about when/why this memory was formed
            event_date: When the event occurred (defaults to now)

        Returns:
            List of created unit IDs
        """
        # Run async version synchronously
        return asyncio.run(self.put_async(agent_id, content, context, event_date))

    async def put_async(
        self,
        agent_id: str,
        content: str,
        context: str = "",
        event_date: Optional[datetime] = None,
        document_id: Optional[str] = None,
        document_metadata: Optional[Dict[str, Any]] = None,
        upsert: bool = False,
    ) -> List[str]:
        """
        Store content as memory units with temporal and semantic links (ASYNC version).

        This is a convenience wrapper around put_batch_async for a single content item.

        Args:
            agent_id: Unique identifier for the agent
            content: Text content to store
            context: Context about when/why this memory was formed
            event_date: When the event occurred (defaults to now)
            document_id: Optional document ID for tracking and upsert
            document_metadata: Optional metadata about the document
            upsert: If True and document_id exists, delete old units and create new ones

        Returns:
            List of created unit IDs
        """
        # Use put_batch_async with a single item (avoids code duplication)
        result = await self.put_batch_async(
            agent_id=agent_id,
            contents=[{
                "content": content,
                "context": context,
                "event_date": event_date
            }],
            document_id=document_id,
            document_metadata=document_metadata,
            upsert=upsert
        )

        # Return the first (and only) list of unit IDs
        return result[0] if result else []

    async def put_batch_async(
        self,
        agent_id: str,
        contents: List[Dict[str, Any]],
        document_id: Optional[str] = None,
        document_metadata: Optional[Dict[str, Any]] = None,
        upsert: bool = False,
    ) -> List[List[str]]:
        """
        Store multiple content items as memory units in ONE batch operation.

        This is MUCH more efficient than calling put_async multiple times:
        - Extracts facts from all contents in parallel
        - Generates ALL embeddings in ONE batch
        - Does ALL database operations in ONE transaction

        Args:
            agent_id: Unique identifier for the agent
            contents: List of dicts with keys:
                - "content" (required): Text content to store
                - "context" (optional): Context about the memory
                - "event_date" (optional): When the event occurred
            document_id: Optional document ID for tracking and upsert
            document_metadata: Optional metadata about the document
            upsert: If True and document_id exists, delete old units and create new ones

        Returns:
            List of lists of unit IDs (one list per content item)

        Example:
            unit_ids = await memory.put_batch_async(
                agent_id="user123",
                contents=[
                    {"content": "Alice works at Google", "context": "conversation"},
                    {"content": "Bob loves Python", "context": "conversation"},
                ],
                document_id="meeting-2024-01-15",
                upsert=True
            )
            # Returns: [["unit-id-1"], ["unit-id-2"]]
        """
        start_time = time.time()
        logger.debug(f"\n{'='*60}")
        logger.debug(f"PUT_BATCH_ASYNC START: {agent_id}")
        logger.debug(f"Batch size: {len(contents)} content items")
        logger.debug(f"{'='*60}")

        if not contents:
            return []

        # Step 1: Extract facts from ALL contents in parallel
        step_start = time.time()

        # Create tasks for parallel fact extraction
        fact_extraction_tasks = []
        for item in contents:
            content = item["content"]
            context = item.get("context", "")
            event_date = item.get("event_date") or utcnow()

            task = extract_facts(content, event_date, context)
            fact_extraction_tasks.append((task, event_date, context))

        # Wait for all fact extractions to complete
        all_fact_results = await asyncio.gather(*[task for task, _, _ in fact_extraction_tasks])

        # Flatten and track which facts belong to which content
        all_fact_texts = []
        all_fact_dates = []
        all_contexts = []
        all_fact_entities = []  # NEW: Store LLM-extracted entities per fact
        content_boundaries = []  # [(start_idx, end_idx), ...]

        current_idx = 0
        for i, ((_, event_date, context), fact_dicts) in enumerate(zip(fact_extraction_tasks, all_fact_results)):
            start_idx = current_idx

            for fact_dict in fact_dicts:
                all_fact_texts.append(fact_dict['fact'])
                try:
                    from dateutil import parser as date_parser
                    fact_date = date_parser.isoparse(fact_dict['date'])
                    all_fact_dates.append(fact_date)
                except Exception:
                    all_fact_dates.append(event_date)
                all_contexts.append(context)
                # Extract entities from fact (default to empty list if not present)
                all_fact_entities.append(fact_dict.get('entities', []))

            end_idx = current_idx + len(fact_dicts)
            content_boundaries.append((start_idx, end_idx))
            current_idx = end_idx

        total_facts = len(all_fact_texts)

        if total_facts == 0:
            return [[] for _ in contents]

        # Step 2: Generate ALL embeddings in ONE batch (HUGE speedup!)
        step_start = time.time()
        all_embeddings = await self._generate_embeddings_batch(all_fact_texts)
        logger.debug(f"[2] Generate embeddings (parallel): {len(all_embeddings)} embeddings in {time.time() - step_start:.3f}s")

        # Step 3: Process everything in ONE database transaction
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # Handle document tracking and upsert
                    if document_id:
                        import hashlib
                        import json

                        # Calculate content hash from all content items
                        combined_content = "\n".join([c.get("content", "") for c in contents])
                        content_hash = hashlib.sha256(combined_content.encode()).hexdigest()

                        # If upsert, delete old document first (cascades to units and links)
                        if upsert:
                            deleted = await conn.fetchval(
                                "DELETE FROM documents WHERE id = $1 AND agent_id = $2 RETURNING id",
                                document_id, agent_id
                            )
                            if deleted:
                                logger.debug(f"[3.1] Upsert: Deleted existing document '{document_id}' and all its units")

                        # Insert or update document
                        # Always use ON CONFLICT for idempotent behavior
                        await conn.execute(
                            """
                            INSERT INTO documents (id, agent_id, original_text, content_hash, metadata)
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (id, agent_id) DO UPDATE
                            SET original_text = EXCLUDED.original_text,
                                content_hash = EXCLUDED.content_hash,
                                metadata = EXCLUDED.metadata,
                                updated_at = NOW()
                            """,
                            document_id,
                            agent_id,
                            combined_content,
                            content_hash,
                            json.dumps(document_metadata or {})
                        )
                        logger.debug(f"[3.2] Document '{document_id}' stored/updated")

                    # Deduplication check for all facts
                    step_start = time.time()
                    all_is_duplicate = []
                    for sentence, embedding, fact_date in zip(all_fact_texts, all_embeddings, all_fact_dates):
                        dup_flags = await self._find_duplicate_facts_batch(
                            conn, agent_id, [sentence], [embedding], fact_date
                        )
                        all_is_duplicate.extend(dup_flags)

                    duplicates_filtered = sum(all_is_duplicate)
                    new_facts = total_facts - duplicates_filtered
                    logger.debug(f"[3] Deduplication check: {duplicates_filtered} duplicates filtered, {new_facts} new facts in {time.time() - step_start:.3f}s")

                    # Filter out duplicates
                    filtered_sentences = [s for s, is_dup in zip(all_fact_texts, all_is_duplicate) if not is_dup]
                    filtered_embeddings = [e for e, is_dup in zip(all_embeddings, all_is_duplicate) if not is_dup]
                    filtered_dates = [d for d, is_dup in zip(all_fact_dates, all_is_duplicate) if not is_dup]
                    filtered_contexts = [c for c, is_dup in zip(all_contexts, all_is_duplicate) if not is_dup]
                    filtered_entities = [ents for ents, is_dup in zip(all_fact_entities, all_is_duplicate) if not is_dup]

                    if not filtered_sentences:
                        logger.debug(f"[PUT_BATCH_ASYNC] All facts were duplicates, returning empty")
                        return [[] for _ in contents]

                    # Batch insert ALL units
                    step_start = time.time()
                    # Convert embeddings to strings for asyncpg vector type
                    filtered_embeddings_str = [str(emb) for emb in filtered_embeddings]
                    results = await conn.fetch(
                        """
                        INSERT INTO memory_units (agent_id, document_id, text, context, embedding, event_date, access_count)
                        SELECT * FROM unnest($1::text[], $2::text[], $3::text[], $4::text[], $5::vector[], $6::timestamptz[], $7::integer[])
                        RETURNING id
                        """,
                        [agent_id] * len(filtered_sentences),
                        [document_id] * len(filtered_sentences) if document_id else [None] * len(filtered_sentences),
                        filtered_sentences,
                        filtered_contexts,
                        filtered_embeddings_str,
                        filtered_dates,
                        [0] * len(filtered_sentences)
                    )

                    created_unit_ids = [str(row['id']) for row in results]
                    logger.debug(f"[5] Batch insert units: {len(created_unit_ids)} units in {time.time() - step_start:.3f}s")

                    # Process entities for ALL units
                    step_start = time.time()
                    all_entity_links = await self._extract_entities_batch_optimized(
                        conn, agent_id, created_unit_ids, filtered_sentences, "", filtered_dates, filtered_entities
                    )
                    logger.debug(f"[6] Process entities (batched): {time.time() - step_start:.3f}s")

                    # Create temporal links
                    step_start = time.time()
                    await self._create_temporal_links_batch_per_fact(conn, agent_id, created_unit_ids)
                    logger.debug(f"[7] Batch create temporal links: {time.time() - step_start:.3f}s")

                    # Create semantic links
                    step_start = time.time()
                    await self._create_semantic_links_batch(conn, agent_id, created_unit_ids, filtered_embeddings)
                    logger.debug(f"[8] Batch create semantic links: {time.time() - step_start:.3f}s")

                    # Insert entity links
                    step_start = time.time()
                    if all_entity_links:
                        await self._insert_entity_links_batch(conn, all_entity_links)
                    logger.debug(f"[9] Batch insert entity links: {time.time() - step_start:.3f}s")

                    # Transaction auto-commits on success
                    commit_start = time.time()
                    logger.debug(f"[10] Commit: {time.time() - commit_start:.3f}s")

                    # Map created unit IDs back to original content items
                    # Account for duplicates when mapping back
                    result_unit_ids = []
                    filtered_idx = 0

                    for start_idx, end_idx in content_boundaries:
                        content_unit_ids = []
                        for i in range(start_idx, end_idx):
                            if not all_is_duplicate[i]:
                                content_unit_ids.append(created_unit_ids[filtered_idx])
                                filtered_idx += 1
                        result_unit_ids.append(content_unit_ids)

                    total_time = time.time() - start_time
                    logger.debug(f"\n{'='*60}")
                    logger.debug(f"PUT_BATCH_ASYNC COMPLETE: {len(created_unit_ids)} units from {len(contents)} contents in {total_time:.3f}s")
                    logger.debug(f"{'='*60}\n")

                    return result_unit_ids

                except Exception as e:
                    # Transaction auto-rolls back on exception
                    import traceback
                    traceback.print_exc()
                    raise Exception(f"Failed to store batch memory: {str(e)}")

    def search(
        self,
        agent_id: str,
        query: str,
        thinking_budget: int = 50,
        top_k: int = 10,
        enable_trace: bool = False,
        weight_activation: float = 0.30,
        weight_semantic: float = 0.30,
        weight_recency: float = 0.25,
        weight_frequency: float = 0.15,
        mmr_lambda: float = 0.5,
    ) -> tuple[List[Dict[str, Any]], Optional[Any]]:
        """
        Search memories using spreading activation (synchronous wrapper).

        This is a synchronous wrapper around search_async() for convenience.
        For best performance, use search_async() directly.

        Args:
            agent_id: Agent ID to search for
            query: Search query
            thinking_budget: How many units to explore (computational budget)
            top_k: Number of results to return
            enable_trace: If True, returns detailed SearchTrace object
            weight_activation: Weight for activation component (default: 0.30)
            weight_semantic: Weight for semantic similarity component (default: 0.30)
            weight_recency: Weight for recency component (default: 0.25)
            weight_frequency: Weight for frequency component (default: 0.15)
            mmr_lambda: Lambda for MMR diversification (0=max diversity, 1=no diversity, default: 0.5)

        Returns:
            Tuple of (results, trace)
        """
        # Run async version synchronously
        return asyncio.run(self.search_async(
            agent_id, query, thinking_budget, top_k, enable_trace,
            weight_activation, weight_semantic, weight_recency, weight_frequency, mmr_lambda
        ))

    async def search_async(
        self,
        agent_id: str,
        query: str,
        thinking_budget: int = 50,
        top_k: int = 10,
        enable_trace: bool = False,
        weight_activation: float = 0.30,
        weight_semantic: float = 0.30,
        weight_recency: float = 0.25,
        weight_frequency: float = 0.15,
        mmr_lambda: float = 0.5,
    ) -> tuple[List[Dict[str, Any]], Optional[Any]]:
        """
        Search memories using spreading activation (ASYNC version).

        This implements the core SEARCH operation:
        1. Find entry points (most relevant units via vector search)
        2. Spread activation through the graph
        3. Weight results by activation + recency + frequency
        4. Return top results

        Args:
            agent_id: Agent ID to search for
            query: Search query
            thinking_budget: How many units to explore (computational budget)
            top_k: Number of results to return
            live_tracer: Optional LiveSearchTracer for visualization

        Returns:
            List of memory units with their weights, sorted by relevance
        """
        # Initialize tracer if requested
        from .search_tracer import SearchTracer
        tracer = SearchTracer(query, thinking_budget, top_k) if enable_trace else None
        if tracer:
            tracer.start()

        pool = await self._get_pool()
        search_start = time.time()

        # Buffer logs for clean output in concurrent scenarios
        search_id = f"{agent_id[:8]}-{int(time.time() * 1000) % 100000}"
        log_buffer = []
        log_buffer.append(f"[SEARCH {search_id}] Query: '{query[:50]}...' (budget={thinking_budget}, top_k={top_k})")

        try:
            # Step 1: Generate query embedding (CPU-bound, no DB needed)
            step_start = time.time()
            query_embedding = self._generate_embedding(query)
            step_duration = time.time() - step_start
            log_buffer.append(f"  [1] Generate query embedding: {step_duration:.3f}s")

            if tracer:
                tracer.record_query_embedding(query_embedding)
                tracer.add_phase_metric("generate_query_embedding", step_duration)

            # Step 2: Find entry points (acquire connection only for this query)
            step_start = time.time()
            query_embedding_str = str(query_embedding)

            # Log connection acquisition
            conn_acquire_start = time.time()
            async with pool.acquire() as conn:
                conn_acquire_time = time.time() - conn_acquire_start
                if conn_acquire_time > 0.1:  # Log if waiting > 100ms
                    log_buffer.append(f"      [2.1] Waited {conn_acquire_time:.3f}s for connection (pool busy)")

                entry_points = await conn.fetch(
                    """
                    SELECT id, text, context, event_date, access_count, embedding,
                           1 - (embedding <=> $1::vector) AS similarity
                    FROM memory_units
                    WHERE agent_id = $2
                      AND embedding IS NOT NULL
                      AND (1 - (embedding <=> $1::vector)) >= 0.5
                    ORDER BY embedding <=> $1::vector
                    LIMIT 3
                    """,
                    query_embedding_str, agent_id
                )

            step_duration = time.time() - step_start
            log_buffer.append(f"  [2] Find entry points: {len(entry_points)} found in {step_duration:.3f}s")

            if tracer:
                tracer.add_phase_metric("find_entry_points", step_duration, {"count": len(entry_points)})
                for rank, ep in enumerate(entry_points, 1):
                    tracer.add_entry_point(
                        node_id=str(ep["id"]),
                        text=ep["text"],
                        similarity=ep["similarity"],
                        rank=rank
                    )

            if not entry_points:
                logger.debug(f"[SEARCH] Complete: 0 results in {time.time() - search_start:.3f}s")
                if tracer:
                    trace = tracer.finalize([])
                    return [], trace
                return [], None

            # Step 3: Spreading activation with budget (in-memory processing)
            step_start = time.time()
            visited = set()
            results = []
            budget_remaining = thinking_budget
            # Initialize entry points with their actual similarity scores instead of 1.0
            # Format: (unit, activation, is_entry, parent_node_id, link_type, link_weight)
            queue = [(dict(unit), unit["similarity"], True, None, None, None) for unit in entry_points]

            # Track substep timings
            calculate_weight_time = 0
            query_neighbors_time = 0
            process_neighbors_time = 0

            # Track which nodes were visited for deferred access count update
            visited_node_ids = []

            # Process nodes in batches for efficient neighbor querying
            BATCH_SIZE = 50
            nodes_to_process = []  # (unit, activation, is_entry_point, parent_node_id, link_type, link_weight)

            while queue and budget_remaining > 0:
                # Collect a batch of nodes to process (in-memory, no DB)
                while queue and len(nodes_to_process) < BATCH_SIZE and budget_remaining > 0:
                    current_unit, activation, is_entry_point, parent_node_id, link_type, link_weight = queue.pop(0)
                    unit_id = str(current_unit["id"])

                    if unit_id not in visited:
                        visited.add(unit_id)
                        budget_remaining -= 1
                        nodes_to_process.append((current_unit, activation, is_entry_point, parent_node_id, link_type, link_weight))
                        visited_node_ids.append(unit_id)  # Track for deferred update
                    elif tracer:
                        # Node already visited - prune
                        tracer.prune_node(unit_id, "already_visited", activation)

                if not nodes_to_process:
                    break

                # Acquire connection ONLY for neighbor queries (defer access count updates)
                node_ids = [str(node[0]["id"]) for node in nodes_to_process]

                # Log connection acquisition for batch queries
                batch_conn_start = time.time()
                async with pool.acquire() as conn:
                    batch_conn_acquire = time.time() - batch_conn_start
                    if batch_conn_acquire > 0.1:  # Log if waiting > 100ms
                        log_buffer.append(f"      [3.3.1] Waited {batch_conn_acquire:.3f}s for connection (pool busy) - batch size: {len(node_ids)}")

                    # Query neighbors for ALL nodes in batch at once (without embeddings for speed)
                    # Convert string UUIDs to UUID type for faster matching
                    substep_start = time.time()
                    uuid_array = [uuid.UUID(nid) for nid in node_ids]
                    all_neighbors = await conn.fetch(
                        """
                        SELECT ml.from_unit_id, ml.to_unit_id, ml.weight, ml.link_type, ml.entity_id,
                               mu.text, mu.context, mu.event_date, mu.access_count,
                               mu.id as neighbor_id
                        FROM memory_links ml
                        JOIN memory_units mu ON ml.to_unit_id = mu.id
                        WHERE ml.from_unit_id = ANY($1::uuid[])
                          AND ml.weight >= 0.1
                        ORDER BY ml.from_unit_id, ml.weight DESC
                        """,
                        uuid_array
                    )
                    neighbor_query_time = time.time() - substep_start
                    if neighbor_query_time > 1.0:  # Log slow neighbor queries
                        log_buffer.append(f"      [3.3.3] Slow NEIGHBOR query: {neighbor_query_time:.3f}s for {len(node_ids)} nodes → {len(all_neighbors)} neighbors")
                    query_neighbors_time += neighbor_query_time

                    # Fetch embeddings for current batch nodes (needed for weight calculation)
                    substep_start = time.time()
                    embeddings = await conn.fetch(
                        "SELECT id, embedding FROM memory_units WHERE id = ANY($1::uuid[])",
                        uuid_array
                    )
                    embedding_map = {str(row["id"]): row["embedding"] for row in embeddings}
                    fetch_embeddings_time = time.time() - substep_start
                    if fetch_embeddings_time > 0.5:
                        log_buffer.append(f"      [3.3.4] Slow EMBEDDING fetch: {fetch_embeddings_time:.3f}s for {len(node_ids)} nodes")
                    query_neighbors_time += fetch_embeddings_time

                # Group neighbors by from_unit_id (in-memory, no DB)
                substep_start = time.time()
                neighbors_by_node = {}
                for neighbor in all_neighbors:
                    from_id = str(neighbor["from_unit_id"])
                    if from_id not in neighbors_by_node:
                        neighbors_by_node[from_id] = []
                    neighbors_by_node[from_id].append(neighbor)

                # Process each node in the batch (CPU-bound, no DB)
                for current_unit, activation, is_entry_point, parent_node_id, parent_link_type, parent_link_weight in nodes_to_process:
                    unit_id = str(current_unit["id"])

                    # Calculate combined weight
                    event_date = current_unit["event_date"]
                    days_since = (utcnow() - event_date).total_seconds() / 86400

                    recency_weight = calculate_recency_weight(days_since)
                    frequency_weight = calculate_frequency_weight(current_unit.get("access_count", 0))

                    # Normalize frequency to [0, 1] range
                    frequency_normalized = (frequency_weight - 1.0) / 1.0

                    # Calculate semantic similarity between query and this memory
                    # Get embedding from the map we fetched
                    memory_embedding = embedding_map.get(unit_id)
                    if memory_embedding is not None:
                        # Convert embedding to list of floats if it's a string or other type
                        if isinstance(memory_embedding, str):
                            import json
                            memory_embedding = json.loads(memory_embedding)
                        elif not isinstance(memory_embedding, (list, np.ndarray)):
                            # If it's some other type, try to convert it
                            memory_embedding = list(memory_embedding)

                        # Cosine similarity = 1 - cosine distance
                        query_vec = np.array(query_embedding, dtype=np.float64)
                        memory_vec = np.array(memory_embedding, dtype=np.float64)
                        # Cosine similarity
                        dot_product = np.dot(query_vec, memory_vec)
                        norm_query = np.linalg.norm(query_vec)
                        norm_memory = np.linalg.norm(memory_vec)
                        semantic_similarity = dot_product / (norm_query * norm_memory) if norm_query > 0 and norm_memory > 0 else 0.0
                    else:
                        semantic_similarity = 0.0

                    # Combined weight using configurable parameters
                    final_weight = (
                        weight_activation * activation +
                        weight_semantic * semantic_similarity +
                        weight_recency * recency_weight +
                        weight_frequency * frequency_normalized
                    )

                    # Notify tracer
                    if tracer:
                        tracer.visit_node(
                            node_id=unit_id,
                            text=current_unit["text"],
                            context=current_unit.get("context", ""),
                            event_date=event_date,
                            access_count=current_unit.get("access_count", 0),
                            is_entry_point=is_entry_point,
                            parent_node_id=parent_node_id,
                            link_type=parent_link_type,
                            link_weight=parent_link_weight,
                            activation=activation,
                            semantic_similarity=semantic_similarity,
                            recency=recency_weight,
                            frequency=frequency_normalized,
                            final_weight=final_weight,
                        )

                    results.append({
                        "id": unit_id,
                        "text": current_unit["text"],
                        "context": current_unit.get("context", ""),
                        "event_date": event_date.isoformat(),
                        "weight": final_weight,
                        "activation": activation,
                        "semantic_similarity": semantic_similarity,
                        "recency": recency_weight,
                        "frequency": frequency_weight,
                        "embedding": memory_embedding,  # Store for MMR
                    })

                    # Spread to neighbors (from batch query results)
                    neighbors = neighbors_by_node.get(unit_id, [])

                    # Group neighbors by to_unit_id to handle multiple connections
                    neighbors_grouped = {}
                    for neighbor in neighbors:
                        neighbor_id = str(neighbor["to_unit_id"])
                        if neighbor_id not in neighbors_grouped:
                            neighbors_grouped[neighbor_id] = []
                        neighbors_grouped[neighbor_id].append(neighbor)

                    # Process each unique neighbor (aggregating multiple links)
                    for neighbor_id, neighbor_links in neighbors_grouped.items():
                        if neighbor_id in visited:
                            continue

                        # Sort links by weight descending to identify primary link
                        neighbor_links_sorted = sorted(neighbor_links, key=lambda x: x["weight"], reverse=True)
                        primary_link = neighbor_links_sorted[0]

                        # Aggregate link weights: max + 30% bonus for additional links
                        max_weight = primary_link["weight"]
                        bonus_weight = sum(link["weight"] for link in neighbor_links_sorted[1:]) * 0.3
                        combined_weight = max_weight + bonus_weight

                        # Calculate new activation using combined weight
                        new_activation = activation * combined_weight * 0.8  # 0.8 = decay factor

                        # Use primary link metadata for queue and trace
                        primary_link_type = primary_link["link_type"]
                        primary_entity_id = str(primary_link["entity_id"]) if primary_link["entity_id"] else None

                        if new_activation > 0.1:
                            queue.append(({
                                "id": primary_link["to_unit_id"],
                                "text": primary_link["text"],
                                "context": primary_link.get("context", ""),
                                "event_date": primary_link["event_date"],
                                "access_count": primary_link["access_count"],
                            }, new_activation, False, unit_id, primary_link_type, combined_weight))  # parent_id, link_type, combined_weight

                            # Record all links in trace (primary + additional)
                            if tracer:
                                # Add primary link with combined activation
                                tracer.add_neighbor_link(
                                    from_node_id=unit_id,
                                    to_node_id=neighbor_id,
                                    link_type=primary_link_type,
                                    link_weight=combined_weight,
                                    entity_id=primary_entity_id,
                                    new_activation=new_activation,
                                    followed=True
                                )

                                # Add additional links as supplementary (if multiple connections exist)
                                for additional_link in neighbor_links_sorted[1:]:
                                    additional_link_type = additional_link["link_type"]
                                    additional_entity_id = str(additional_link["entity_id"]) if additional_link["entity_id"] else None
                                    tracer.add_neighbor_link(
                                        from_node_id=unit_id,
                                        to_node_id=neighbor_id,
                                        link_type=additional_link_type,
                                        link_weight=additional_link["weight"],
                                        entity_id=additional_entity_id,
                                        new_activation=None,  # Don't show activation for supplementary links
                                        followed=True,
                                        is_supplementary=True  # Mark as supplementary link
                                    )
                        elif tracer:
                            # Record pruned link
                            tracer.add_neighbor_link(
                                from_node_id=unit_id,
                                to_node_id=neighbor_id,
                                link_type=primary_link_type,
                                link_weight=combined_weight,
                                entity_id=primary_entity_id,
                                new_activation=new_activation,
                                followed=False,
                                prune_reason="activation_too_low"
                            )

                calculate_weight_time += time.time() - substep_start
                process_neighbors_time += time.time() - substep_start

                # Clear batch for next iteration
                nodes_to_process = []

            spreading_activation_time = time.time() - step_start
            num_batches = (len(visited) + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
            log_buffer.append(f"  [3] Spreading activation: {len(visited)} nodes visited in {spreading_activation_time:.3f}s")
            log_buffer.append(f"      [3.1] Calculate weights: {calculate_weight_time:.3f}s")
            log_buffer.append(f"      [3.2] Query neighbors: {query_neighbors_time:.3f}s ({num_batches} batched queries)")
            log_buffer.append(f"      [3.3] Process neighbors: {process_neighbors_time:.3f}s")

            if tracer:
                tracer.add_phase_metric("spreading_activation", spreading_activation_time, {
                    "nodes_visited": len(visited),
                    "num_batches": num_batches
                })

            # Step 4: Queue access count updates (background worker will process them)
            if visited_node_ids:
                await self._access_count_queue.put(visited_node_ids)
                log_buffer.append(f"  [4] Queued access count updates for {len(visited_node_ids)} nodes")

            # Step 5: Sort by final weight and apply MMR for diversity
            step_start = time.time()
            results.sort(key=lambda x: x["weight"], reverse=True)

            # Apply MMR (Maximal Marginal Relevance) for diversity if lambda < 1.0
            if mmr_lambda < 1.0 and len(results) > top_k:
                top_results = self._apply_mmr(results, top_k, mmr_lambda, log_buffer)
                log_buffer.append(f"  [5] MMR diversification (λ={mmr_lambda}): {time.time() - step_start:.3f}s")
            else:
                top_results = results[:top_k]
                # Add original rank and remove embeddings from results
                for idx, result in enumerate(top_results):
                    result["original_rank"] = idx + 1
                    result["mmr_score"] = None
                    result["mmr_relevance"] = None
                    result["mmr_max_similarity"] = None
                    result["mmr_diversified"] = False
                    result.pop("embedding", None)
                log_buffer.append(f"  [5] Sort and return top {top_k} (no MMR): {time.time() - step_start:.3f}s")

            total_time = time.time() - search_start
            log_buffer.append(f"[SEARCH {search_id}] Complete: {len(top_results)} results in {total_time:.3f}s")

            # Log all buffered logs at once
            logger.info("\n" + "\n".join(log_buffer))

            # Finalize trace if enabled
            if tracer:
                trace = tracer.finalize(top_results)
                return top_results, trace
            return top_results, None

        except Exception as e:
            log_buffer.append(f"[SEARCH {search_id}] ERROR after {time.time() - search_start:.3f}s: {str(e)}")
            logger.error("\n" + "\n".join(log_buffer))
            raise Exception(f"Failed to search memories: {str(e)}")

    def _apply_mmr(
        self,
        results: List[Dict[str, Any]],
        top_k: int,
        mmr_lambda: float,
        log_buffer: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Apply Maximal Marginal Relevance (MMR) to diversify search results.

        MMR balances relevance with diversity by selecting results that are:
        1. Relevant to the query (high score)
        2. Different from already selected results (low similarity)

        Formula: MMR = λ * relevance - (1-λ) * max_similarity_to_selected

        Args:
            results: Sorted list of all results with embeddings
            top_k: Number of results to select
            mmr_lambda: Balance parameter (0=max diversity, 1=max relevance)
            log_buffer: Logging buffer

        Returns:
            Diversified list of top_k results
        """
        if not results or top_k <= 0:
            return []

        # Normalize weights to [0, 1] for fair comparison with similarity
        max_weight = max(r["weight"] for r in results)
        min_weight = min(r["weight"] for r in results)
        weight_range = max_weight - min_weight if max_weight > min_weight else 1.0

        # Pre-compute normalized relevance scores for all results
        for idx, result in enumerate(results):
            result["original_rank"] = idx + 1
            result["normalized_relevance"] = (result["weight"] - min_weight) / weight_range

        # Extract embeddings as a numpy array for vectorized operations
        # Shape: (num_results, embedding_dim)
        embeddings_list = []
        valid_indices = []
        for idx, result in enumerate(results):
            if result.get("embedding") is not None:
                embeddings_list.append(result["embedding"])
                valid_indices.append(idx)

        if not embeddings_list:
            # No embeddings available, just return top-k by relevance
            return results[:top_k]

        # Stack embeddings into a matrix (num_results, embedding_dim)
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)

        # Normalize embeddings for faster cosine similarity (just dot product after normalization)
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        embeddings_matrix = embeddings_matrix / norms

        selected_indices = []
        remaining_indices = list(range(len(results)))
        diversified_count = 0

        for selection_round in range(min(top_k, len(results))):
            if not remaining_indices:
                break

            best_mmr_score = float('-inf')
            best_remaining_idx = 0

            # Vectorized computation for all remaining candidates
            for remaining_idx, candidate_idx in enumerate(remaining_indices):
                candidate = results[candidate_idx]
                normalized_relevance = candidate["normalized_relevance"]

                # Calculate max similarity to selected results
                max_similarity = 0.0
                if selected_indices and candidate_idx in valid_indices:
                    # Find position in embeddings_matrix
                    embedding_idx = valid_indices.index(candidate_idx)
                    candidate_embedding = embeddings_matrix[embedding_idx]

                    # Vectorized similarity calculation with all selected embeddings
                    if selected_indices:
                        selected_embedding_indices = [valid_indices.index(idx) for idx in selected_indices if idx in valid_indices]
                        if selected_embedding_indices:
                            selected_embeddings = embeddings_matrix[selected_embedding_indices]
                            # Compute cosine similarities in one operation (already normalized, so just dot product)
                            similarities = np.dot(selected_embeddings, candidate_embedding)
                            max_similarity = float(np.max(similarities))

                # MMR score: balance relevance and diversity
                mmr_score = mmr_lambda * normalized_relevance - (1 - mmr_lambda) * max_similarity

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_remaining_idx = remaining_idx
                    best_max_similarity = max_similarity

            # Select the best candidate
            best_candidate_idx = remaining_indices.pop(best_remaining_idx)
            best_candidate = results[best_candidate_idx]

            # Store MMR metadata
            best_candidate["mmr_score"] = best_mmr_score
            best_candidate["mmr_relevance"] = best_candidate["normalized_relevance"]
            best_candidate["mmr_max_similarity"] = best_max_similarity
            best_candidate["mmr_diversified"] = best_remaining_idx > 0

            selected_indices.append(best_candidate_idx)

            if best_remaining_idx > 0:
                diversified_count += 1

        log_buffer.append(f"      MMR: Selected {len(selected_indices)} results, {diversified_count} diversified picks")

        # Return selected results in order
        selected_results = [results[idx] for idx in selected_indices]

        # Remove embeddings from final results (not needed in response)
        for result in selected_results:
            result.pop("embedding", None)
            result.pop("normalized_relevance", None)  # Clean up temp field

        return selected_results

    async def get_document(self, document_id: str, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document metadata and statistics.

        Args:
            document_id: Document ID to retrieve
            agent_id: Agent ID that owns the document

        Returns:
            Dictionary with document info or None if not found
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            doc = await conn.fetchrow(
                """
                SELECT d.id, d.agent_id, d.original_text, d.content_hash, d.metadata,
                       d.created_at, d.updated_at, COUNT(mu.id) as unit_count
                FROM documents d
                LEFT JOIN memory_units mu ON mu.document_id = d.id
                WHERE d.id = $1 AND d.agent_id = $2
                GROUP BY d.id, d.agent_id, d.original_text, d.content_hash, d.metadata, d.created_at, d.updated_at
                """,
                document_id, agent_id
            )

            if not doc:
                return None

            import json
            return {
                "id": doc["id"],
                "agent_id": doc["agent_id"],
                "original_text": doc["original_text"],
                "content_hash": doc["content_hash"],
                "metadata": json.loads(doc["metadata"]) if doc["metadata"] else {},
                "unit_count": doc["unit_count"],
                "created_at": doc["created_at"],
                "updated_at": doc["updated_at"]
            }

    async def delete_document(self, document_id: str, agent_id: str) -> Dict[str, int]:
        """
        Delete a document and all its associated memory units and links.

        Args:
            document_id: Document ID to delete
            agent_id: Agent ID that owns the document

        Returns:
            Dictionary with counts of deleted items
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Count units before deletion
                units_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM memory_units WHERE document_id = $1",
                    document_id
                )

                # Delete document (cascades to memory_units and all their links)
                deleted = await conn.fetchval(
                    "DELETE FROM documents WHERE id = $1 AND agent_id = $2 RETURNING id",
                    document_id, agent_id
                )

                return {
                    "document_deleted": 1 if deleted else 0,
                    "memory_units_deleted": units_count if deleted else 0
                }

    async def delete_agent(self, agent_id: str) -> Dict[str, int]:
        """
        Delete all data for a specific agent (multi-tenant cleanup).

        This is much more efficient than dropping all tables and allows
        multiple agents to coexist in the same database.

        Deletes (with CASCADE):
        - All memory units for this agent
        - All entities for this agent
        - All associated links, unit-entity associations, and co-occurrences

        Args:
            agent_id: Agent ID to delete

        Returns:
            Dictionary with counts of deleted items
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # Count before deletion for reporting
                    units_count = await conn.fetchval("SELECT COUNT(*) FROM memory_units WHERE agent_id = $1", agent_id)
                    entities_count = await conn.fetchval("SELECT COUNT(*) FROM entities WHERE agent_id = $1", agent_id)

                    # Delete memory units (cascades to unit_entities, memory_links)
                    await conn.execute("DELETE FROM memory_units WHERE agent_id = $1", agent_id)

                    # Delete entities (cascades to unit_entities, entity_cooccurrences, memory_links with entity_id)
                    await conn.execute("DELETE FROM entities WHERE agent_id = $1", agent_id)

                    return {
                        "memory_units_deleted": units_count,
                        "entities_deleted": entities_count
                    }

                except Exception as e:
                    raise Exception(f"Failed to delete agent data: {str(e)}")

    async def _extract_entities_batch_optimized(
        self,
        conn,
        agent_id: str,
        unit_ids: List[str],
        sentences: List[str],
        context: str,
        fact_dates: List,
        llm_entities: List[List[Dict]],  # NEW: Entities from LLM
    ) -> List[tuple]:
        """
        Process LLM-extracted entities for ALL facts in batch.

        Uses entities provided by the LLM (no spaCy needed), then resolves
        and links them in bulk.

        Returns list of tuples for batch insertion: (from_unit_id, to_unit_id, link_type, weight, entity_id)
        """
        try:
            # Step 1: Convert LLM entities to the format expected by entity resolver
            substep_start = time.time()
            all_entities = []
            for entity_list in llm_entities:
                # Convert List[Entity] or List[dict] to List[Dict] format
                formatted_entities = []
                for ent in entity_list:
                    # Handle both Entity objects and dicts
                    if hasattr(ent, 'text'):
                        formatted_entities.append({'text': ent.text, 'type': ent.type})
                    elif isinstance(ent, dict):
                        formatted_entities.append({'text': ent.get('text', ''), 'type': ent.get('type', 'CONCEPT')})
                all_entities.append(formatted_entities)

            total_entities = sum(len(ents) for ents in all_entities)
            logger.debug(f"  [6.1] Process LLM entities: {total_entities} entities from {len(sentences)} facts in {time.time() - substep_start:.3f}s")

            # Step 2: Resolve entities in BATCH (much faster!)
            substep_start = time.time()
            step_6_2_start = time.time()

            # [6.2.1] Prepare all entities for batch resolution
            substep_6_2_1_start = time.time()
            all_entities_flat = []
            entity_to_unit = []  # Maps flat index to (unit_id, local_index)

            for unit_id, entities, fact_date in zip(unit_ids, all_entities, fact_dates):
                if not entities:
                    continue

                for local_idx, entity in enumerate(entities):
                    all_entities_flat.append({
                        'text': entity['text'],
                        'type': entity['type'],
                        'nearby_entities': entities,
                    })
                    entity_to_unit.append((unit_id, local_idx, fact_date))
            logger.debug(f"    [6.2.1] Prepare entities: {len(all_entities_flat)} entities in {time.time() - substep_6_2_1_start:.3f}s")

            # Resolve ALL entities in one batch call
            if all_entities_flat:
                # [6.2.2] Batch resolve entities
                substep_6_2_2_start = time.time()
                # Group by date for batch resolution (most will have same date)
                entities_by_date = {}
                for idx, (unit_id, local_idx, fact_date) in enumerate(entity_to_unit):
                    date_key = fact_date
                    if date_key not in entities_by_date:
                        entities_by_date[date_key] = []
                    entities_by_date[date_key].append((idx, all_entities_flat[idx]))

                # Resolve each date group in batch
                resolved_entity_ids = [None] * len(all_entities_flat)
                for fact_date, entities_group in entities_by_date.items():
                    indices = [idx for idx, _ in entities_group]
                    entities_data = [entity_data for _, entity_data in entities_group]

                    batch_resolved = await self.entity_resolver.resolve_entities_batch(
                        agent_id=agent_id,
                        entities_data=entities_data,
                        context=context,
                        unit_event_date=fact_date,
                        conn=conn
                    )

                    for idx, entity_id in zip(indices, batch_resolved):
                        resolved_entity_ids[idx] = entity_id
                logger.debug(f"    [6.2.2] Resolve entities: {len(all_entities_flat)} entities in {time.time() - substep_6_2_2_start:.3f}s")

                # [6.2.3] Create unit-entity links in BATCH
                substep_6_2_3_start = time.time()
                # Map resolved entities back to units and collect all (unit, entity) pairs
                unit_to_entity_ids = {}
                unit_entity_pairs = []
                for idx, (unit_id, local_idx, fact_date) in enumerate(entity_to_unit):
                    if unit_id not in unit_to_entity_ids:
                        unit_to_entity_ids[unit_id] = []

                    entity_id = resolved_entity_ids[idx]
                    unit_to_entity_ids[unit_id].append(entity_id)
                    unit_entity_pairs.append((unit_id, entity_id))

                # Batch insert all unit-entity links (MUCH faster!)
                await self.entity_resolver.link_units_to_entities_batch(unit_entity_pairs, conn=conn)
                logger.debug(f"    [6.2.3] Create unit-entity links (batched): {len(unit_entity_pairs)} links in {time.time() - substep_6_2_3_start:.3f}s")

                logger.debug(f"  [6.2] Entity resolution (batched): {len(all_entities_flat)} entities resolved in {time.time() - step_6_2_start:.3f}s")
            else:
                unit_to_entity_ids = {}
                logger.debug(f"  [6.2] Entity resolution (batched): 0 entities in {time.time() - step_6_2_start:.3f}s")

            # Step 3: Create entity links between units that share entities
            substep_start = time.time()
            # Collect all unique entity IDs
            all_entity_ids = set()
            for entity_ids in unit_to_entity_ids.values():
                all_entity_ids.update(entity_ids)

            # For each entity, find all units that reference it (one query per entity)
            entity_to_units = {}
            for entity_id in all_entity_ids:
                rows = await conn.fetch(
                    """
                    SELECT unit_id
                    FROM unit_entities
                    WHERE entity_id = $1
                    """,
                    entity_id
                )
                entity_to_units[entity_id] = [row['unit_id'] for row in rows]

            # Create bidirectional links between units that share entities
            links = []
            for entity_id, units_with_entity in entity_to_units.items():
                # For each pair of units with this entity, create bidirectional links
                for i, unit_id_1 in enumerate(units_with_entity):
                    for unit_id_2 in units_with_entity[i+1:]:
                        # Bidirectional links
                        links.append((unit_id_1, unit_id_2, 'entity', 1.0, entity_id))
                        links.append((unit_id_2, unit_id_1, 'entity', 1.0, entity_id))

            logger.debug(f"  [6.3] Entity link creation: {len(links)} links for {len(all_entity_ids)} unique entities in {time.time() - substep_start:.3f}s")

            return links

        except Exception as e:
            logger.error(f" Failed to extract entities in batch: {str(e)}")
            import traceback
            traceback.print_exc()
            # Re-raise to trigger rollback at put_async level
            raise

    async def _create_temporal_links_batch_per_fact(
        self,
        conn,
        agent_id: str,
        unit_ids: List[str],
        time_window_hours: int = 24,
    ):
        """
        Create temporal links for multiple units, each with their own event_date.

        Queries the event_date for each unit from the database and creates temporal
        links based on individual dates (supports per-fact dating).
        """
        if not unit_ids:
            return

        try:
            # Get the event_date for each new unit
            rows = await conn.fetch(
                """
                SELECT id, event_date
                FROM memory_units
                WHERE id::text = ANY($1)
                """,
                unit_ids
            )
            new_units = {str(row['id']): row['event_date'] for row in rows}

            # Create links based on each unit's individual event_date
            links = []
            for unit_id, unit_event_date in new_units.items():
                # Find units within the time window of THIS specific unit
                recent_units = await conn.fetch(
                    """
                    SELECT id, event_date
                    FROM memory_units
                    WHERE agent_id = $1
                      AND id != $2
                      AND event_date BETWEEN $3 AND $4
                    ORDER BY event_date DESC
                    LIMIT 10
                    """,
                    agent_id,
                    unit_id,
                    unit_event_date - timedelta(hours=time_window_hours),
                    unit_event_date + timedelta(hours=time_window_hours)
                )

                for recent_row in recent_units:
                    recent_id = recent_row['id']
                    recent_event_date = recent_row['event_date']
                    # Calculate temporal proximity weight
                    time_diff_hours = abs((unit_event_date - recent_event_date).total_seconds() / 3600)
                    weight = max(0.3, 1.0 - (time_diff_hours / time_window_hours))
                    links.append((unit_id, str(recent_id), 'temporal', weight, None))

            if links:
                await conn.executemany(
                    """
                    INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                    """,
                    links
                )

        except Exception as e:
            logger.error(f" Failed to create temporal links: {str(e)}")
            import traceback
            traceback.print_exc()
            # Re-raise to trigger rollback at put_async level
            raise

    async def _create_semantic_links_batch(
        self,
        conn,
        agent_id: str,
        unit_ids: List[str],
        embeddings: List[List[float]],
        top_k: int = 5,
        threshold: float = 0.7,
    ):
        """
        Create semantic links for multiple units efficiently.

        For each unit, finds similar units and creates links.
        """
        if not unit_ids or not embeddings:
            return

        try:
            all_links = []

            for unit_id, embedding in zip(unit_ids, embeddings):
                # Find similar units using vector similarity
                # Convert embedding to string for asyncpg
                embedding_str = str(embedding)
                similar_units = await conn.fetch(
                    """
                    SELECT id, 1 - (embedding <=> $1::vector) AS similarity
                    FROM memory_units
                    WHERE agent_id = $2
                      AND id != $3
                      AND embedding IS NOT NULL
                      AND (1 - (embedding <=> $1::vector)) >= $4
                    ORDER BY embedding <=> $1::vector
                    LIMIT $5
                    """,
                    embedding_str, agent_id, unit_id, threshold, top_k
                )

                for row in similar_units:
                    similar_id = row['id']
                    similarity = row['similarity']
                    all_links.append((unit_id, str(similar_id), 'semantic', float(similarity), None))

            if all_links:
                await conn.executemany(
                    """
                    INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                    """,
                    all_links
                )

        except Exception as e:
            logger.error(f" Failed to create semantic links: {str(e)}")
            import traceback
            traceback.print_exc()
            # Re-raise to trigger rollback at put_async level
            raise

    async def _insert_entity_links_batch(self, conn, links: List[tuple]):
        """Insert all entity links in a single batch."""
        if not links:
            return

        try:
            await conn.executemany(
                """
                INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                """,
                links
            )
        except Exception as e:
            logger.warning(f" Failed to insert entity links: {str(e)}")
