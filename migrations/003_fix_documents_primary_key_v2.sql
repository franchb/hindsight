-- Migration: Fix documents table primary key to be composite (id, agent_id)
-- This is a safer approach that doesn't drop the table

-- Step 1: Drop the foreign key constraint from memory_units
ALTER TABLE memory_units DROP CONSTRAINT IF EXISTS memory_units_document_id_fkey;
ALTER TABLE memory_units DROP CONSTRAINT IF EXISTS memory_units_document_fkey;

-- Step 2: Drop the old primary key on documents
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_pkey;

-- Step 3: Add the new composite primary key
ALTER TABLE documents ADD PRIMARY KEY (id, agent_id);

-- Step 4: Add back the foreign key constraint with the composite key
ALTER TABLE memory_units
    ADD CONSTRAINT memory_units_document_fkey
    FOREIGN KEY (document_id, agent_id)
    REFERENCES documents(id, agent_id)
    ON DELETE CASCADE;
