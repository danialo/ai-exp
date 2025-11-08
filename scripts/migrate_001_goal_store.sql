PRAGMA foreign_keys=ON;

-- Goals table (forward-only migration)
CREATE TABLE IF NOT EXISTS goals (
  id TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  category TEXT NOT NULL,
  value REAL NOT NULL CHECK (value BETWEEN 0.0 AND 1.0),
  effort REAL NOT NULL CHECK (effort BETWEEN 0.0 AND 1.0),
  risk REAL NOT NULL CHECK (risk BETWEEN 0.0 AND 1.0),
  horizon_min_min INTEGER NOT NULL,
  horizon_max_min INTEGER,
  aligns_with TEXT NOT NULL DEFAULT '[]',        -- JSON array
  contradicts TEXT NOT NULL DEFAULT '[]',        -- JSON array
  success_metrics TEXT NOT NULL DEFAULT '{}',    -- JSON object
  state TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  metadata TEXT NOT NULL DEFAULT '{}',           -- JSON object
  version INTEGER NOT NULL DEFAULT 0,
  deleted_at TEXT
);

CREATE INDEX IF NOT EXISTS ix_goals_state ON goals(state);
CREATE INDEX IF NOT EXISTS ix_goals_category ON goals(category);
CREATE INDEX IF NOT EXISTS ix_goals_updated ON goals(updated_at DESC);
CREATE INDEX IF NOT EXISTS ix_goals_deadline ON goals(horizon_max_min);

-- Idempotency keys for command endpoints
CREATE TABLE IF NOT EXISTS goal_idempotency (
  key TEXT PRIMARY KEY,
  op TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  created_at TEXT NOT NULL
);
