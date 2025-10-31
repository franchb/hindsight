-- Migration: Add documents table and document_id to memory_units
-- This enables document tracking, upsert, and cascade deletion

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    PRIMARY KEY (id, agent_id),
    original_text TEXT,
    content_hash TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add document_id column to memory_units (nullable for backward compatibility)
ALTER TABLE memory_units ADD COLUMN IF NOT EXISTS document_id TEXT REFERENCES documents(id) ON DELETE CASCADE;

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_documents_agent_id ON documents(agent_id);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_memory_units_document_id ON memory_units(document_id);
