# End-to-End Goal Generation Example

## Scenario: Detecting Recurring Task Failures

This example shows **exactly** what happens when the TaskFailureDetector finds a problem and creates a goal.

---

## Initial State (Before Detection)

### Task Execution History
```
Time: 2025-11-09 10:00:00 - Task: daily_backup - Result: SUCCESS
Time: 2025-11-09 11:00:00 - Task: daily_backup - Result: FAILED (timeout)
Time: 2025-11-09 12:00:00 - Task: daily_backup - Result: FAILED (timeout)
Time: 2025-11-09 13:00:00 - Task: daily_backup - Result: FAILED (timeout)
Time: 2025-11-09 14:00:00 - Task: daily_backup - Result: FAILED (timeout)
```

### Current Goals in Database
```sql
-- No existing goals about backup failures
SELECT * FROM goals WHERE text LIKE '%backup%';
-- (empty result)
```

---

## Hour 1: Pattern Detection Run (14:00:00)

### Step 1: Background Task Wakes Up

**app.py - Goal Generator Loop**
```python
# Runs every hour
await asyncio.sleep(3600)  # ⏰ Time's up!
created, rejected = await goal_generator.generate_and_create_goals()
```

**Log Output:**
```
2025-11-09 14:00:00 INFO - Starting goal generation scan...
```

---

### Step 2: Detector Scanning Phase

**TaskFailureDetector.detect() is called**

```python
# TaskFailureDetector scans last 24 hours
failures = self._get_recent_failures()  # Queries task execution history

# Finds:
failures = [
    {"task_type": "daily_backup", "timestamp": "2025-11-09 11:00:00", "error": "timeout"},
    {"task_type": "daily_backup", "timestamp": "2025-11-09 12:00:00", "error": "timeout"},
    {"task_type": "daily_backup", "timestamp": "2025-11-09 13:00:00", "error": "timeout"},
    {"task_type": "daily_backup", "timestamp": "2025-11-09 14:00:00", "error": "timeout"},
]

# Count by task type
failure_counts = {"daily_backup": 4}

# Check threshold (3+ failures = problem detected!)
if 4 >= 3:  # ✓ Pattern detected!
```

---

### Step 3: Proposal Creation

**TaskFailureDetector creates GoalProposal**

```python
proposal = GoalProposal(
    # Required fields
    text="Fix recurring failures in 'daily_backup' task",
    category=GoalCategory.CAPABILITY,
    pattern_detected="task_failure_recurring",
    evidence={
        "task_type": "daily_backup",
        "failure_count": 4,
        "lookback_hours": 24,
        "error_type": "timeout",
        "first_failure": "2025-11-09 11:00:00",
        "last_failure": "2025-11-09 14:00:00"
    },
    confidence=0.85,  # High confidence - clear pattern
    estimated_value=0.8,  # High value - backups are critical
    estimated_effort=0.5,  # Medium effort
    estimated_risk=0.3,   # Low risk - debugging task
    detector_name="task_failure_detector",

    # Auto-generated fields
    proposal_id="prop_a7f3c1d9",  # UUID
    detected_at=datetime(2025, 11, 9, 14, 0, 0),
    expires_at=datetime(2025, 11, 16, 14, 0, 0),  # 7 days later
    aligns_with=[],  # Could link to beliefs about reliability
    contradicts=[]
)
```

**Log Output:**
```
2025-11-09 14:00:00 INFO - TaskFailureDetector: Found 1 pattern
2025-11-09 14:00:00 DEBUG - Proposal prop_a7f3c1d9: "Fix recurring failures in 'daily_backup' task" (confidence=0.85)
```

**Returns:** `[proposal]` (1 proposal)

---

### Step 4: Evaluation Phase

**GoalGenerator.evaluate_proposal() runs safety checks**

```python
# Check 1: Expired?
if datetime.now() > proposal.expires_at:  # 2025-11-09 < 2025-11-16
    return False, "expired"
# ✓ PASS

# Check 2: Confidence high enough?
if proposal.confidence < 0.7:  # 0.85 >= 0.7
    return False, f"confidence_too_low_{proposal.confidence}"
# ✓ PASS

# Check 3: Rate limits
today_system_goals = goal_store.count_goals_created_today(source="system")  # = 2
if today_system_goals >= 10:  # 2 < 10
    return False, "rate_limit_exceeded"
# ✓ PASS

detector_goals_today = goal_store.count_goals_created_today(
    source="system",
    created_by="task_failure_detector"
)  # = 0
if detector_goals_today >= 3:  # 0 < 3
    return False, "rate_limit_exceeded"
# ✓ PASS

# Check 4: Duplicate?
existing = goal_store.list_goals(state_filter="all")
for goal in existing:
    if similarity(goal.text, proposal.text) > 0.8:  # No match found
        return False, "duplicate_goal_exists"
# ✓ PASS

# Check 5: Belief alignment (if belief_store exists)
if belief_store:
    beliefs = belief_store.list_beliefs(state="active")
    # Check if proposal aligns with active beliefs about system reliability
    alignment_score = calculate_alignment(proposal, beliefs)  # = 0.75
    if alignment_score < 0.5:  # 0.75 >= 0.5
        return False, "belief_misalignment"
# ✓ PASS

return True, None  # ✅ All checks passed!
```

**Log Output:**
```
2025-11-09 14:00:00 DEBUG - Evaluating proposal prop_a7f3c1d9...
2025-11-09 14:00:00 DEBUG - ✓ Confidence check passed (0.85)
2025-11-09 14:00:00 DEBUG - ✓ Rate limit check passed (2/10 daily, 0/3 per detector)
2025-11-09 14:00:00 DEBUG - ✓ Duplicate check passed
2025-11-09 14:00:00 DEBUG - ✓ Belief alignment passed (0.75)
2025-11-09 14:00:00 INFO - Proposal prop_a7f3c1d9 APPROVED for creation
```

---

### Step 5: Goal Creation Decision

**Should this be auto-approved?**

```python
if proposal.confidence >= 0.9:  # 0.85 < 0.9
    state = GoalState.ACTIVE
    auto_approved = True
else:
    state = GoalState.PROPOSED  # ← This path (needs user review)
    auto_approved = False
```

**Decision:** Create as **PROPOSED** (user review required)

---

### Step 6: Goal Creation

**GoalGenerator.create_system_goal() creates GoalDefinition**

```python
goal = GoalDefinition(
    id="goal_8f2e4a9c",  # Generated UUID
    text="Fix recurring failures in 'daily_backup' task",
    category=GoalCategory.CAPABILITY,
    value=0.8,
    effort=0.5,
    risk=0.3,
    horizon_min_min=60,    # At least 1 hour
    horizon_max_min=1440,  # At most 1 day

    # Source tracking (NEW!)
    source=GoalSource.SYSTEM,  # ← System-generated
    created_by="task_failure_detector",  # ← Which detector
    proposal_id="prop_a7f3c1d9",  # ← Link to original proposal
    auto_approved=False,  # ← Needs user review

    state=GoalState.PROPOSED,  # ← Not active yet

    aligns_with=[],
    contradicts=[],
    success_metrics={
        "task_success_rate": 0.95,
        "consecutive_successes": 3
    },

    metadata={
        "confidence": 0.85,
        "pattern": "task_failure_recurring",
        "evidence": {
            "task_type": "daily_backup",
            "failure_count": 4,
            "error_type": "timeout"
        }
    },

    created_at=datetime(2025, 11, 9, 14, 0, 0).timestamp(),
    updated_at=datetime(2025, 11, 9, 14, 0, 0).timestamp(),
)
```

**goal_store.create_goal(goal) executes:**

```sql
INSERT INTO goals (
    id, text, category, value, effort, risk,
    horizon_min_min, horizon_max_min,
    aligns_with, contradicts, success_metrics,
    state, created_at, updated_at, metadata,
    version, deleted_at,
    source, created_by, proposal_id, auto_approved  -- NEW COLUMNS
)
VALUES (
    'goal_8f2e4a9c',
    'Fix recurring failures in ''daily_backup'' task',
    'capability',
    0.8, 0.5, 0.3,
    60, 1440,
    '[]', '[]', '{"task_success_rate": 0.95, "consecutive_successes": 3}',
    'proposed',  -- ← State
    1699538400.0, 1699538400.0,
    '{"confidence": 0.85, "pattern": "task_failure_recurring", ...}',
    0, NULL,
    'system', 'task_failure_detector', 'prop_a7f3c1d9', 0  -- ← Source tracking
);
```

---

### Step 7: Identity Ledger Logging

**Audit trail written to data/identity_ledger.ndjson:**

```json
{
  "ts": 1699538400.123,
  "schema": 2,
  "event": "goal_created",
  "meta": {
    "goal_id": "goal_8f2e4a9c",
    "category": "capability",
    "source": "system",
    "created_by": "task_failure_detector"
  }
}
```

---

### Step 8: Telemetry Logging

**Log Output:**
```
2025-11-09 14:00:00 INFO - Goal generator: 1 created, 0 rejected
2025-11-09 14:00:00 INFO - Goal generator telemetry: {
  'total_proposals_evaluated': 1,
  'total_goals_created': 1,
  'total_goals_rejected': 0,
  'auto_approved_count': 0,
  'rejection_reasons': {}
}
```

---

## Final State (After Detection)

### Database State

```sql
-- Query system-generated goals
SELECT id, text, source, created_by, state, auto_approved
FROM goals
WHERE source='system'
ORDER BY created_at DESC
LIMIT 1;

-- Result:
-- id: goal_8f2e4a9c
-- text: Fix recurring failures in 'daily_backup' task
-- source: system
-- created_by: task_failure_detector
-- state: proposed
-- auto_approved: 0
```

### Identity Ledger

```bash
tail -1 data/identity_ledger.ndjson | jq
```

```json
{
  "ts": 1699538400.123,
  "schema": 2,
  "event": "goal_created",
  "meta": {
    "goal_id": "goal_8f2e4a9c",
    "category": "capability",
    "source": "system",
    "created_by": "task_failure_detector"
  }
}
```

---

## User Interaction

### Option 1: User Reviews via API (future endpoint)

```bash
# List proposed system goals
curl https://172.239.66.45/api/v1/goals?state=proposed&source=system

# Response:
[
  {
    "id": "goal_8f2e4a9c",
    "text": "Fix recurring failures in 'daily_backup' task",
    "source": "system",
    "created_by": "task_failure_detector",
    "state": "proposed",
    "confidence": 0.85,
    "evidence": {
      "task_type": "daily_backup",
      "failure_count": 4,
      "error_type": "timeout"
    }
  }
]

# Approve the goal
curl -X POST https://172.239.66.45/api/v1/goals/goal_8f2e4a9c/approve

# Goal state changes: proposed → active
```

### Option 2: User Queries via CLI

```bash
# See what the system is proposing
sqlite3 data/raw_store.db "
SELECT
  id,
  text,
  created_by,
  json_extract(metadata, '$.evidence.failure_count') as failures,
  json_extract(metadata, '$.confidence') as confidence
FROM goals
WHERE source='system' AND state='proposed';
"

# Result:
# goal_8f2e4a9c | Fix recurring failures in 'daily_backup' task | task_failure_detector | 4 | 0.85

# Manually activate it
sqlite3 data/raw_store.db "
UPDATE goals
SET state='active', updated_at=unixepoch('now')
WHERE id='goal_8f2e4a9c';
"
```

### Option 3: HTN Planner Auto-Selects

The HTN Planner will see this goal during its next planning cycle:

```python
# htn_planner.py
goals = goal_store.list_goals(state_filter="active")
# Now includes: goal_8f2e4a9c

# Planner evaluates all active goals and selects based on:
# - Utility score (value, effort, risk)
# - Alignment with beliefs
# - Current context

selected_goal = planner.select_goal(goals)
# Might select the backup fix goal if it's high priority

# Decomposes into tasks:
tasks = [
  "Analyze backup task logs for timeout patterns",
  "Identify resource bottleneck causing timeouts",
  "Implement timeout increase or resource optimization",
  "Test backup task with fix applied"
]
```

---

## Alternative Scenario: High Confidence Auto-Approval

If the confidence had been **0.92** instead of 0.85:

```python
if proposal.confidence >= 0.9:  # 0.92 >= 0.9 ✓
    state = GoalState.ACTIVE  # ← Auto-approved!
    auto_approved = True
```

**Database result:**
```sql
-- state: active (immediately available for HTN planner)
-- auto_approved: 1
```

**Log output:**
```
2025-11-09 14:00:00 INFO - Goal goal_8f2e4a9c AUTO-APPROVED (confidence=0.92)
2025-11-09 14:00:00 INFO - Goal generator: 1 created, 0 rejected
2025-11-09 14:00:00 INFO - Goal generator telemetry: {
  'auto_approved_count': 1  ← Note the difference
}
```

---

## Alternative Scenario: Rejection

If there were already 3 goals from task_failure_detector today:

```python
detector_goals_today = 3
if detector_goals_today >= 3:  # 3 >= 3 ✓ FAIL
    return False, "rate_limit_exceeded"
```

**Result:** Proposal rejected, no goal created

**Log output:**
```
2025-11-09 14:00:00 INFO - Proposal prop_a7f3c1d9 REJECTED: rate_limit_exceeded
2025-11-09 14:00:00 INFO - Goal generator: 0 created, 1 rejected
2025-11-09 14:00:00 INFO - Goal generator telemetry: {
  'total_proposals_evaluated': 1,
  'total_goals_created': 0,
  'total_goals_rejected': 1,
  'rejection_reasons': {
    'rate_limit_exceeded': 1
  }
}
```

---

## Timeline Summary

```
10:00 - daily_backup task succeeds
11:00 - daily_backup task fails (timeout)
12:00 - daily_backup task fails (timeout)
13:00 - daily_backup task fails (timeout)
14:00 - daily_backup task fails (timeout)
14:00 - ⏰ Goal generator hourly scan triggers
14:00 - TaskFailureDetector.detect() analyzes last 24h
14:00 - Pattern detected: 4 failures >= threshold (3)
14:00 - GoalProposal created (confidence=0.85)
14:00 - Proposal evaluated: ALL CHECKS PASS ✓
14:00 - Goal created: state=PROPOSED (needs review)
14:00 - Identity ledger updated
14:00 - Telemetry logged: 1 created, 0 rejected
14:00 - [User can now review and approve via API/CLI]
```

---

## Key Takeaways

1. **Pattern Detection is Autonomous** - System watches for problems continuously
2. **Proposals are Smart** - Include evidence, confidence, estimated effort
3. **Safety First** - Multiple checks prevent goal spam and misalignment
4. **Transparency** - Everything logged to identity ledger
5. **User Control** - Low confidence goals require approval
6. **High Confidence = Auto** - Very confident proposals (≥0.9) skip review
7. **Collaborative** - Both user and system feed goals into unified prioritization

The system creates goals **for** you, not **instead of** you!
