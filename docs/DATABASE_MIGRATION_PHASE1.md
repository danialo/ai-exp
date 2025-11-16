# Phase 1 Database Migration: Task Execution Tracking

## Overview

Phase 1 task execution tracking added a `causes` field to the Experience model to track causal relationships between experiences and task executions. This required a database schema migration.

## Migration Applied

**Date**: November 6, 2025
**Branch**: feature/https-setup

### Schema Change

Added `causes` column to the `experience` table:

```sql
ALTER TABLE experience ADD COLUMN causes TEXT DEFAULT "[]"
```

### Affected Databases

- `data/raw_store.db` - Primary experience store
- `data/core.db` - Core database (if exists)

### Result

The `experience` table now has 15 columns:
1. id
2. type
3. created_at
4. content
5. provenance
6. evidence_ptrs
7. confidence
8. embeddings
9. affect
10. parents
11. causes (NEW)
12. sign
13. ownership
14. session_id
15. consolidated

## Issue Resolved

**Symptom**: API endpoints returning "Internal Server Error", GUI showing "loading..." for memory counts

**Error**:
```
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) no such column: experience.causes
```

**Root Cause**: Phase 1 implementation added `causes` field to `src/memory/models.py` Experience model, but existing databases did not have this column.

**Fix Applied**: Added column with default value `"[]"` to maintain backward compatibility.

## Verification

After migration:
- `/api/stats` endpoint working ✓
- `/api/memories` endpoint working ✓
- `/api/conversations` endpoint working ✓
- GUI memory counts displaying correctly ✓
- Total experiences: 3189
- Total vectors: 4419

## Future Migrations

For future schema changes:
1. Update model in `src/memory/models.py`
2. Create migration script in `scripts/migrations/`
3. Document in this file or create new migration doc
4. Test on development database before production
5. Consider using Alembic for automated migrations

## Related

- Phase 1 Status: `.claude/tasks/STATUS-PHASE-1-COMPLETE.md`
- MCP Spec: `.claude/tasks/prompt-008-mcp-task-execution.md`
- Experience Model: `src/memory/models.py`
