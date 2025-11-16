-- Migration 002: Add source tracking to goals table
-- Enables tracking of user vs system-generated goals

PRAGMA foreign_keys=ON;

-- Add new columns for goal source tracking
ALTER TABLE goals ADD COLUMN source TEXT NOT NULL DEFAULT 'user';
ALTER TABLE goals ADD COLUMN created_by TEXT;
ALTER TABLE goals ADD COLUMN proposal_id TEXT;
ALTER TABLE goals ADD COLUMN auto_approved INTEGER NOT NULL DEFAULT 0;

-- Create index for filtering by source
CREATE INDEX IF NOT EXISTS ix_goals_source ON goals(source);

-- Create index for linking proposals to goals
CREATE INDEX IF NOT EXISTS ix_goals_proposal ON goals(proposal_id);
