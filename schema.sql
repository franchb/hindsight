-- Enable the pgvector extension and uuid extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- TEMPORAL + SEMANTIC + ENTITY MEMORY ARCHITECTURE
-- ============================================================================

-- Documents: Source of memory units (for tracking, updates, and deletion)
CREATE TABLE IF NOT EXISTS documents (
    id TEXT NOT NULL,  -- User-provided document ID
    agent_id TEXT NOT NULL,
    PRIMARY KEY (id, agent_id),
    original_text TEXT,  -- Full original content (for context expansion)
    content_hash TEXT,  -- SHA256 hash for deduplication
    metadata JSONB DEFAULT '{}'::jsonb,  -- User-provided metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Memory Units: Individual sentence-level memories
CREATE TABLE IF NOT EXISTS memory_units (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL,
    document_id TEXT,  -- Link to source document
    text TEXT NOT NULL,
    embedding vector(384),  -- bge-small-en-v1.5 dimension
    context TEXT,  -- What was happening when this memory was formed
    event_date TIMESTAMPTZ NOT NULL,  -- When the event occurred
    access_count INTEGER DEFAULT 0,  -- For recency/frequency weighting
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Entities: Resolved entities (people, organizations, locations, etc.)
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    canonical_name TEXT NOT NULL,  -- "Alice", "Google", "San Francisco"
    entity_type TEXT NOT NULL,     -- PERSON, ORG, GPE, etc.
    agent_id TEXT NOT NULL,        -- Entities are scoped to agents
    metadata JSONB DEFAULT '{}'::jsonb,  -- Additional entity info
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    mention_count INTEGER DEFAULT 1
);

-- Unit-Entity associations: Which entities appear in which units
CREATE TABLE IF NOT EXISTS unit_entities (
    unit_id UUID REFERENCES memory_units(id) ON DELETE CASCADE,
    entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (unit_id, entity_id)
);

-- Entity Co-occurrences: Materialized cache of which entities appear together
-- This dramatically speeds up entity resolution by avoiding expensive joins
CREATE TABLE IF NOT EXISTS entity_cooccurrences (
    entity_id_1 UUID REFERENCES entities(id) ON DELETE CASCADE,
    entity_id_2 UUID REFERENCES entities(id) ON DELETE CASCADE,
    cooccurrence_count INTEGER DEFAULT 1,
    last_cooccurred TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (entity_id_1, entity_id_2),
    CHECK (entity_id_1 < entity_id_2)  -- Enforce ordering to avoid duplicates
);

-- Memory Links: Temporal, semantic, AND entity connections
CREATE TABLE IF NOT EXISTS memory_links (
    from_unit_id UUID REFERENCES memory_units(id) ON DELETE CASCADE,
    to_unit_id UUID REFERENCES memory_units(id) ON DELETE CASCADE,
    link_type TEXT NOT NULL,  -- 'temporal', 'semantic', or 'entity'
    weight FLOAT NOT NULL DEFAULT 1.0,  -- Link strength
    entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,  -- Set for entity links
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Unique constraint to prevent duplicate links
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_links_unique
ON memory_links (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid));

-- ============================================================================
-- FOREIGN KEY CONSTRAINTS
-- ============================================================================

-- Add foreign key from memory_units to documents (composite key)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'memory_units_document_fkey'
    ) THEN
        ALTER TABLE memory_units
            ADD CONSTRAINT memory_units_document_fkey
            FOREIGN KEY (document_id, agent_id)
            REFERENCES documents(id, agent_id)
            ON DELETE CASCADE;
    END IF;
END $$;

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Document indexes
CREATE INDEX IF NOT EXISTS idx_documents_agent_id ON documents(agent_id);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);

-- Memory unit indexes
CREATE INDEX IF NOT EXISTS idx_memory_units_agent_id ON memory_units(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_units_document_id ON memory_units(document_id);
CREATE INDEX IF NOT EXISTS idx_memory_units_event_date ON memory_units(event_date DESC);
CREATE INDEX IF NOT EXISTS idx_memory_units_agent_date ON memory_units(agent_id, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_memory_units_access_count ON memory_units(access_count DESC);

-- Vector similarity index (HNSW for fast approximate nearest neighbor)
CREATE INDEX IF NOT EXISTS idx_memory_units_embedding ON memory_units
USING hnsw (embedding vector_cosine_ops);

-- Entity indexes
CREATE INDEX IF NOT EXISTS idx_entities_agent_id ON entities(agent_id);
CREATE INDEX IF NOT EXISTS idx_entities_canonical_name ON entities(canonical_name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_agent_name_type ON entities(agent_id, canonical_name, entity_type);

-- Unit-entity indexes
CREATE INDEX IF NOT EXISTS idx_unit_entities_unit ON unit_entities(unit_id);
CREATE INDEX IF NOT EXISTS idx_unit_entities_entity ON unit_entities(entity_id);

-- Entity co-occurrence indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_entity_cooccurrences_entity1 ON entity_cooccurrences(entity_id_1);
CREATE INDEX IF NOT EXISTS idx_entity_cooccurrences_entity2 ON entity_cooccurrences(entity_id_2);
CREATE INDEX IF NOT EXISTS idx_entity_cooccurrences_count ON entity_cooccurrences(cooccurrence_count DESC);

-- Link indexes for graph traversal
CREATE INDEX IF NOT EXISTS idx_memory_links_from ON memory_links(from_unit_id);
CREATE INDEX IF NOT EXISTS idx_memory_links_to ON memory_links(to_unit_id);
CREATE INDEX IF NOT EXISTS idx_memory_links_type ON memory_links(link_type);
CREATE INDEX IF NOT EXISTS idx_memory_links_entity ON memory_links(entity_id) WHERE entity_id IS NOT NULL;

-- Composite index for spreading activation neighbor queries (from_unit_id + weight filter)
CREATE INDEX IF NOT EXISTS idx_memory_links_from_weight ON memory_links(from_unit_id, weight DESC)
WHERE weight >= 0.1;
