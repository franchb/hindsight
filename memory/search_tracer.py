"""
Search tracer for collecting detailed search execution traces.

The SearchTracer collects comprehensive information about each step
of the spreading activation search process for debugging and visualization.
"""
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Literal

from .search_trace import (
    SearchTrace,
    QueryInfo,
    EntryPoint,
    NodeVisit,
    WeightComponents,
    LinkInfo,
    PruningDecision,
    SearchSummary,
    SearchPhaseMetrics,
)


class SearchTracer:
    """
    Tracer for collecting detailed search execution information.

    Usage:
        tracer = SearchTracer(query="Who is Alice?", thinking_budget=50, top_k=10)
        tracer.start()

        # During search...
        tracer.record_query_embedding(embedding)
        tracer.add_entry_point(node_id, text, similarity, rank)
        tracer.visit_node(...)
        tracer.prune_node(...)

        # After search...
        trace = tracer.finalize(final_results)
        json_output = trace.to_json()
    """

    def __init__(self, query: str, thinking_budget: int, top_k: int):
        """
        Initialize tracer.

        Args:
            query: Search query text
            thinking_budget: Maximum nodes to explore
            top_k: Number of results requested
        """
        self.query_text = query
        self.thinking_budget = thinking_budget
        self.top_k = top_k

        # Trace data
        self.query_embedding: Optional[List[float]] = None
        self.start_time: Optional[float] = None
        self.entry_points: List[EntryPoint] = []
        self.visits: List[NodeVisit] = []
        self.pruned: List[PruningDecision] = []
        self.phase_metrics: List[SearchPhaseMetrics] = []

        # Tracking state
        self.current_step = 0
        self.nodes_visited_set = set()  # For quick lookups

        # Link statistics
        self.temporal_links_followed = 0
        self.semantic_links_followed = 0
        self.entity_links_followed = 0

    def start(self):
        """Start timing the search."""
        self.start_time = time.time()

    def record_query_embedding(self, embedding: List[float]):
        """Record the query embedding."""
        self.query_embedding = embedding

    def add_entry_point(self, node_id: str, text: str, similarity: float, rank: int):
        """
        Record an entry point.

        Args:
            node_id: Memory unit ID
            text: Memory unit text
            similarity: Cosine similarity to query
            rank: Rank among entry points (1-based)
        """
        self.entry_points.append(
            EntryPoint(
                node_id=node_id,
                text=text,
                similarity_score=similarity,
                rank=rank,
            )
        )

    def visit_node(
        self,
        node_id: str,
        text: str,
        context: str,
        event_date: datetime,
        access_count: int,
        is_entry_point: bool,
        parent_node_id: Optional[str],
        link_type: Optional[Literal["temporal", "semantic", "entity"]],
        link_weight: Optional[float],
        activation: float,
        semantic_similarity: float,
        recency: float,
        frequency: float,
        final_weight: float,
    ):
        """
        Record visiting a node.

        Args:
            node_id: Memory unit ID
            text: Memory unit text
            context: Memory unit context
            event_date: When the memory occurred
            access_count: Access count before this search
            is_entry_point: Whether this is an entry point
            parent_node_id: Node that led here (None for entry points)
            link_type: Type of link from parent
            link_weight: Weight of link from parent
            activation: Activation score
            semantic_similarity: Semantic similarity to query
            recency: Recency weight
            frequency: Frequency weight
            final_weight: Combined final weight
        """
        self.current_step += 1
        self.nodes_visited_set.add(node_id)

        # Calculate weight contributions for transparency
        weights = WeightComponents(
            activation=activation,
            semantic_similarity=semantic_similarity,
            recency=recency,
            frequency=frequency,
            final_weight=final_weight,
            activation_contribution=0.3 * activation,
            semantic_contribution=0.3 * semantic_similarity,
            recency_contribution=0.25 * recency,
            frequency_contribution=0.15 * frequency,
        )

        visit = NodeVisit(
            step=self.current_step,
            node_id=node_id,
            text=text,
            context=context,
            event_date=event_date,
            access_count=access_count,
            is_entry_point=is_entry_point,
            parent_node_id=parent_node_id,
            link_type=link_type,
            link_weight=link_weight,
            weights=weights,
            neighbors_explored=[],
            final_rank=None,  # Will be set later
        )

        self.visits.append(visit)

        # Track link statistics
        if link_type == "temporal":
            self.temporal_links_followed += 1
        elif link_type == "semantic":
            self.semantic_links_followed += 1
        elif link_type == "entity":
            self.entity_links_followed += 1

    def add_neighbor_link(
        self,
        from_node_id: str,
        to_node_id: str,
        link_type: Literal["temporal", "semantic", "entity"],
        link_weight: float,
        entity_id: Optional[str],
        new_activation: Optional[float],
        followed: bool,
        prune_reason: Optional[str] = None,
        is_supplementary: bool = False,
    ):
        """
        Record a link to a neighbor (whether followed or not).

        Args:
            from_node_id: Source node
            to_node_id: Target node
            link_type: Type of link
            link_weight: Weight of link
            entity_id: Entity ID if link is entity-based
            new_activation: Activation passed to neighbor (None for supplementary links)
            followed: Whether link was followed
            prune_reason: Why link was not followed (if not followed)
            is_supplementary: Whether this is a supplementary link (multiple connections)
        """
        # Find the visit for the source node
        visit = None
        for v in self.visits:
            if v.node_id == from_node_id:
                visit = v
                break

        if visit is None:
            # Node not found, skip
            return

        link_info = LinkInfo(
            to_node_id=to_node_id,
            link_type=link_type,
            link_weight=link_weight,
            entity_id=entity_id,
            new_activation=new_activation,
            followed=followed,
            prune_reason=prune_reason,
            is_supplementary=is_supplementary,
        )

        visit.neighbors_explored.append(link_info)

    def prune_node(
        self,
        node_id: str,
        reason: Literal["already_visited", "activation_too_low", "budget_exhausted"],
        activation: float,
    ):
        """
        Record a node being pruned (not visited).

        Args:
            node_id: Node that was pruned
            reason: Why it was pruned
            activation: Activation value when pruned
        """
        self.pruned.append(
            PruningDecision(
                node_id=node_id,
                reason=reason,
                activation=activation,
                would_have_been_step=self.current_step + 1,
            )
        )

    def add_phase_metric(self, phase_name: str, duration_seconds: float, details: Optional[Dict[str, Any]] = None):
        """
        Record metrics for a search phase.

        Args:
            phase_name: Name of the phase
            duration_seconds: Time taken
            details: Additional phase-specific details
        """
        self.phase_metrics.append(
            SearchPhaseMetrics(
                phase_name=phase_name,
                duration_seconds=duration_seconds,
                details=details or {},
            )
        )

    def finalize(self, final_results: List[Dict[str, Any]]) -> SearchTrace:
        """
        Finalize the trace and return the complete SearchTrace object.

        Args:
            final_results: Final ranked results returned to user

        Returns:
            Complete SearchTrace object
        """
        if self.start_time is None:
            raise ValueError("Tracer not started - call start() first")

        total_duration = time.time() - self.start_time

        # Set final ranks on visits based on results
        for rank, result in enumerate(final_results, 1):
            result_node_id = result["id"]
            for visit in self.visits:
                if visit.node_id == result_node_id:
                    visit.final_rank = rank
                    break

        # Create query info
        query_info = QueryInfo(
            query_text=self.query_text,
            query_embedding=self.query_embedding or [],
            timestamp=datetime.now(timezone.utc),
            thinking_budget=self.thinking_budget,
            top_k=self.top_k,
        )

        # Create summary
        summary = SearchSummary(
            total_nodes_visited=len(self.visits),
            total_nodes_pruned=len(self.pruned),
            entry_points_found=len(self.entry_points),
            budget_used=len(self.visits),
            budget_remaining=self.thinking_budget - len(self.visits),
            total_duration_seconds=total_duration,
            results_returned=len(final_results),
            temporal_links_followed=self.temporal_links_followed,
            semantic_links_followed=self.semantic_links_followed,
            entity_links_followed=self.entity_links_followed,
            phase_metrics=self.phase_metrics,
        )

        # Create complete trace
        trace = SearchTrace(
            query=query_info,
            entry_points=self.entry_points,
            visits=self.visits,
            pruned=self.pruned,
            summary=summary,
            final_results=final_results,
        )

        return trace
