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
from typing import Optional

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
        memory = TemporalSemanticMemory()

        # Run search with tracing
        results, trace = await memory.search_async(
            agent_id=request.agent_id,
            query=request.query,
            thinking_budget=request.thinking_budget,
            top_k=request.top_k,
            enable_trace=True,
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
    print("  GET  /              - Visualization UI")
    print("  GET  /api/graph     - Get graph data")
    print("  POST /api/search    - Run search with trace")
    print("\n" + "=" * 80 + "\n")

    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
