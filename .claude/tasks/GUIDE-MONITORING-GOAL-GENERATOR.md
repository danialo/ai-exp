# Goal Generator Monitoring Guide

## Overview

The Goal Generator runs as a background task that:
- Scans for patterns **every hour**
- Evaluates proposals against safety checks
- Creates system-generated goals automatically
- Logs all activity to application logs and identity ledger

## Quick Start Monitoring

### 1. Dashboard View

```bash
./scripts/monitor_goal_generator.sh
```

This shows:
- Startup status
- Recent pattern detection runs
- Latest telemetry
- System-generated goals
- Recent errors

### 2. Live Monitoring

```bash
# Watch all goal generator activity
tail -f logs/app/astra.log | grep -i "goal generator"

# Watch goal creation events in identity ledger
tail -f data/identity_ledger.ndjson | jq 'select(.event=="goal_created")'

# Watch ALL pattern detection (verbose)
tail -f logs/app/astra.log | grep -E "(Goal generator|Pattern|Detector)"
```

### 3. Manual Trigger (for testing)

```bash
# Run pattern detection immediately without waiting for hourly schedule
python3 scripts/trigger_goal_detection.py
```

## What You'll See

### On Application Startup

```
INFO - Goal generator background task started
INFO - Goal generator loop started (interval=1h)
```

This means the background task is running. It will scan every hour.

### During Pattern Detection Run

#### When Patterns Found:

```
INFO - Goal generator: 2 created, 1 rejected
INFO - Goal generator telemetry: {
  'total_proposals_evaluated': 3,
  'total_goals_created': 2,
  'total_goals_rejected': 1,
  'auto_approved_count': 1,
  'rejection_reasons': {
    'confidence_too_low_0.65': 1
  }
}
```

#### When No Patterns Found:

```
INFO - Goal generator: 0 created, 0 rejected
```

(This is normal - detectors only create proposals when they find issues)

### In Identity Ledger

```json
{
  "ts": 1699564823.123,
  "schema": 2,
  "event": "goal_created",
  "meta": {
    "goal_id": "goal_abc123",
    "category": "capability",
    "source": "system",
    "created_by": "task_failure_detector"
  }
}
```

## Understanding Telemetry

```python
{
  'total_proposals_evaluated': 5,  # How many proposals evaluated
  'total_goals_created': 3,        # How many became actual goals
  'total_goals_rejected': 2,       # How many were rejected
  'auto_approved_count': 2,        # How many auto-approved (confidence >= 0.9)
  'rejection_reasons': {           # Why proposals were rejected
    'confidence_too_low_0.65': 1,  # Confidence < 0.7
    'duplicate_goal_exists': 1     # Goal already exists
  }
}
```

## Querying System Goals

### All System-Generated Goals

```bash
sqlite3 data/raw_store.db "
SELECT
  id,
  text,
  source,
  created_by,
  state,
  auto_approved,
  datetime(created_at, 'unixepoch', 'localtime') as created
FROM goals
WHERE source='system'
ORDER BY created_at DESC;
"
```

### Only Active System Goals

```bash
sqlite3 data/raw_store.db "
SELECT id, text, created_by, auto_approved
FROM goals
WHERE source='system' AND state='active'
ORDER BY created_at DESC;
"
```

### Goals by Detector

```bash
sqlite3 data/raw_store.db "
SELECT created_by, COUNT(*) as count
FROM goals
WHERE source='system'
GROUP BY created_by;
"
```

## Pattern Detection Flow

Here's what happens every hour:

```
1. SCAN PHASE
   ├─ Each detector's detect() method is called
   ├─ Detectors analyze their domain
   └─ Return GoalProposal[] (may be empty)

2. EVALUATION PHASE (for each proposal)
   ├─ Check expiration (expires after 7 days)
   ├─ Check confidence (must be >= 0.7)
   ├─ Check rate limits:
   │   ├─ Max 10 system goals/day total
   │   └─ Max 3 goals/day per detector
   ├─ Check duplicates (similar goal already exists?)
   └─ Check belief alignment (if belief_store available)

3. CREATION PHASE
   ├─ If confidence >= 0.9: Create as ACTIVE (auto-approved)
   ├─ If confidence < 0.9: Create as PROPOSED (needs review)
   └─ Log to identity ledger

4. LOGGING PHASE
   ├─ Log counts (created/rejected)
   └─ Log telemetry with rejection reasons
```

## Current Detectors

### TaskFailureDetector

**Watches for:** Recurring task failures
**Triggers when:** Same task type fails 3+ times in 24 hours
**Proposes:** Goal to fix the failing task
**Example:** "Fix recurring failures in 'daily_backup' task (5 failures in 24h)"

**Configuration:**
- Failure threshold: 3 failures
- Lookback window: 24 hours
- Min confidence: 0.75
- Scan interval: 60 minutes

## Rate Limits & Safety

The goal generator has several safety mechanisms:

### Global Rate Limits
- **Max 10 system goals per day** - Prevents goal spam
- **Max 3 goals per detector per day** - Prevents single detector dominating

### Quality Checks
- **Min confidence: 0.7** - Only high-quality proposals accepted
- **Auto-approve threshold: 0.9** - Very confident proposals skip review
- **Duplicate detection** - Won't create similar goals
- **Belief alignment** - Goals must align with active beliefs (if enabled)

### Proposal Expiration
- Proposals expire after **7 days**
- Prevents stale patterns from creating goals

## Troubleshooting

### "Goal generator background task started" never appears

Check:
1. Is `PERSONA_MODE_ENABLED=true` in settings?
2. Is `AWARENESS_ENABLED=true` in settings?
3. Check for errors in `logs/app/astra.log`

### "Goal generator: 0 created, 0 rejected" every hour

This is **normal** if no patterns detected. It means:
- No recurring task failures
- No test coverage drops (when detector added)
- No documentation staleness (when detector added)

The system only creates goals when it finds actual issues.

### "Failed to initialize goal generator" error

Check:
1. GoalStore initialized correctly?
2. Task scheduler available? (required for TaskFailureDetector)
3. Full error in logs: `grep "Failed to initialize goal generator" logs/app/astra.log`

### Goals created but not showing up

Check goal state:
```bash
sqlite3 data/raw_store.db "
SELECT id, text, state, auto_approved
FROM goals
WHERE source='system'
ORDER BY created_at DESC
LIMIT 5;
"
```

Goals with `state='proposed'` need user approval.
Goals with `state='active'` were auto-approved.

## Testing Pattern Detection

### Manual Trigger

```bash
python3 scripts/trigger_goal_detection.py
```

This runs pattern detection immediately and shows detailed output.

### Simulate Task Failures (for TaskFailureDetector)

The TaskFailureDetector is currently a placeholder. To test it properly:

1. Wire it to actual task execution history
2. Create some failing tasks
3. Run manual trigger
4. Check if it proposes fix goals

## Log Locations

- **Application logs:** `logs/app/astra.log`
- **Identity ledger:** `data/identity_ledger.ndjson`
- **Goal database:** `data/raw_store.db` (table: `goals`)

## Next Steps

To extend pattern detection:

1. **Add more detectors:**
   - TestCoverageDetector
   - DocumentationStalenessDetector
   - ComplexityDetector
   - BeliefCoherenceDetector

2. **Add API endpoints:**
   - `GET /v1/goals/proposals` - List proposed goals
   - `POST /v1/goals/proposals/{id}/approve` - Approve proposal
   - `POST /v1/goals/proposals/{id}/reject` - Reject proposal

3. **Wire TaskFailureDetector to real data:**
   - Currently placeholder implementation
   - Needs task execution history
   - Should analyze actual failure patterns
