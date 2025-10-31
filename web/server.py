"""
FastAPI server for memory graph visualization and API.

Provides REST API endpoints for memory operations and serves
the interactive visualization interface.
"""
import asyncpg
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from memory import TemporalSemanticMemory

import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Memory Graph API", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str
    agent_id: str = "default"
    thinking_budget: int = 100
    top_k: int = 10
    mmr_lambda: float = 0.5
    trace: bool = False


class MemoryItem(BaseModel):
    """Single memory item for batch put."""
    content: str
    event_date: Optional[datetime] = None
    context: Optional[str] = None


class BatchPutRequest(BaseModel):
    """Request model for batch put endpoint."""
    agent_id: str
    items: List[MemoryItem]
    document_id: Optional[str] = None
    document_metadata: Optional[Dict[str, Any]] = None
    upsert: bool = False


async def get_graph_data():
    """Fetch graph data from database."""
    conn = await asyncpg.connect(
        os.getenv('DATABASE_URL'),
        statement_cache_size=0  # Disable statement caching for pgbouncer compatibility
    )

    # Get all memory units
    units = await conn.fetch("""
        SELECT id, text, event_date, context
        FROM memory_units
        ORDER BY event_date
    """)

    # Get all links with weights
    links = await conn.fetch("""
        SELECT
            ml.from_unit_id,
            ml.to_unit_id,
            ml.link_type,
            ml.weight,
            e.canonical_name as entity_name
        FROM memory_links ml
        LEFT JOIN entities e ON ml.entity_id = e.id
        ORDER BY ml.link_type, ml.weight DESC
    """)

    # Get entity information
    unit_entities = await conn.fetch("""
        SELECT ue.unit_id, e.canonical_name, e.entity_type
        FROM unit_entities ue
        JOIN entities e ON ue.entity_id = e.id
        ORDER BY ue.unit_id
    """)

    await conn.close()

    # Build entity mapping
    entity_map = {}
    for row in unit_entities:
        unit_id = row['unit_id']
        entity_name = row['canonical_name']
        entity_type = row['entity_type']
        if unit_id not in entity_map:
            entity_map[unit_id] = []
        entity_map[unit_id].append(f"{entity_name} ({entity_type})")

    # Build nodes
    nodes = []
    for row in units:
        unit_id = row['id']
        text = row['text']
        event_date = row['event_date']
        context = row['context']

        entities = entity_map.get(unit_id, [])
        entity_count = len(entities)

        # Color by entity count
        if entity_count == 0:
            color = "#e0e0e0"
        elif entity_count == 1:
            color = "#90caf9"
        else:
            color = "#42a5f5"

        nodes.append({
            "data": {
                "id": str(unit_id),
                "label": text[:50] + "..." if len(text) > 50 else text,
                "text": text,
                "context": context,
                "date": str(event_date.date()),
                "entities": ", ".join(entities) if entities else "None",
                "color": color
            }
        })

    # Build edges
    edges = []
    for row in links:
        from_id = row['from_unit_id']
        to_id = row['to_unit_id']
        link_type = row['link_type']
        weight = row['weight']
        entity_name = row['entity_name']

        # Set color based on link type
        if link_type == 'temporal':
            color = "#00bcd4"
            line_style = "dashed"
        elif link_type == 'semantic':
            color = "#ff69b4"
            line_style = "solid"
        elif link_type == 'entity':
            color = "#ffd700"
            line_style = "solid"
        else:
            color = "#999999"
            line_style = "solid"

        edges.append({
            "data": {
                "id": f"{from_id}-{to_id}-{link_type}",
                "source": str(from_id),
                "target": str(to_id),
                "weight": weight,
                "linkType": link_type,
                "entityName": entity_name or "",
                "color": color,
                "lineStyle": line_style
            }
        })

    # Build table rows
    table_rows = []
    for row in units:
        unit_id = row['id']
        text = row['text']
        event_date = row['event_date']
        context = row['context']

        entities = entity_map.get(unit_id, [])
        entity_str = ", ".join(entities) if entities else "None"
        table_rows.append({
            "id": str(unit_id)[:8] + "...",
            "text": text,
            "context": context,
            "date": str(event_date.date()),
            "entities": entity_str
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "table_rows": table_rows,
        "total_units": len(units)
    }

memory = TemporalSemanticMemory()

@app.get("/")
async def index():
    """Serve the visualization page."""
    return FileResponse("web/templates/index.html")


@app.get("/api/graph")
async def api_graph():
    """Get graph data from database."""
    try:
        data = await get_graph_data()
        return data
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"Error in /api/graph: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
async def api_search(request: SearchRequest):
    """Run a search and return results with trace."""
    try:
        # Initialize memory system
        # Run search with tracing
        results, trace = await memory.search_async(
            agent_id=request.agent_id,
            query=request.query,
            thinking_budget=request.thinking_budget,
            top_k=request.top_k,
            enable_trace=request.trace,
            mmr_lambda=request.mmr_lambda
        )

        # Convert trace to dict
        trace_dict = trace.to_dict() if trace else None

        return {
            'results': results,
            'trace': trace_dict
        }
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"Error in /api/search: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents")
async def api_agents():
    """Get list of available agents from database."""
    try:
        conn = await asyncpg.connect(
            os.getenv('DATABASE_URL'),
            statement_cache_size=0
        )

        # Get distinct agent IDs from memory_units
        agents = await conn.fetch("""
            SELECT DISTINCT agent_id
            FROM memory_units
            WHERE agent_id IS NOT NULL
            ORDER BY agent_id
        """)

        await conn.close()

        agent_list = [row['agent_id'] for row in agents]
        return {"agents": agent_list}
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"Error in /api/agents: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/memories/batch")
async def api_batch_put(request: BatchPutRequest):
    """
    Store multiple memories in batch.

    This endpoint calls put_batch_async to efficiently store multiple memory items.
    Supports document tracking and upsert operations.

    Example request:
    {
        "agent_id": "user123",
        "items": [
            {"content": "Alice works at Google", "context": "work"},
            {"content": "Bob went hiking yesterday", "event_date": "2024-01-15T10:00:00Z"}
        ],
        "document_id": "conversation_123",
        "upsert": false
    }
    """
    try:
        # Validate agent_id - prevent writing to reserved agents
        RESERVED_AGENT_IDS = {"locomo"}
        if request.agent_id in RESERVED_AGENT_IDS:
            raise HTTPException(
                status_code=403,
                detail=f"Cannot write to reserved agent_id '{request.agent_id}'. Reserved agents: {', '.join(RESERVED_AGENT_IDS)}"
            )

        # Initialize memory system


        # Prepare contents for put_batch_async
        contents = []
        for item in request.items:
            content_dict = {"content": item.content}
            if item.event_date:
                content_dict["event_date"] = item.event_date
            if item.context:
                content_dict["context"] = item.context
            contents.append(content_dict)

        # Call put_batch_async
        result = await memory.put_batch_async(
            agent_id=request.agent_id,
            contents=contents,
            document_id=request.document_id,
            document_metadata=request.document_metadata,
            upsert=request.upsert
        )

        await memory.close()

        return {
            "success": True,
            "message": f"Successfully stored {len(contents)} memory items",
            "agent_id": request.agent_id,
            "document_id": request.document_id,
            "items_count": len(contents)
        }
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"Error in /api/memories/batch: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/locomo")
async def api_locomo():
    """Get Locomo benchmark results."""
    import json
    try:
        results_path = Path(__file__).parent.parent / "benchmarks" / "locomo" / "benchmark_results.json"
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Benchmark results not found")

        with open(results_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Benchmark results not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 80)
    print("Memory Graph API Server")
    print("=" * 80)
    print("\nStarting server at http://localhost:8080")
    print("\nEndpoints:")
    print("  GET  /                    - Visualization UI")
    print("  GET  /api/graph           - Get graph data")
    print("  POST /api/search          - Run search with trace")
    print("  POST /api/memories/batch  - Store multiple memories in batch")
    print("  GET  /api/agents          - List available agents")
    print("\n" + "=" * 80 + "\n")

    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
