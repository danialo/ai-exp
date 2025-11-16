# Schedule Safety Tiers

Astra's autonomous scheduling system uses a 3-tier safety model to balance autonomy with human oversight.

## Safety Tiers

### Tier 0: Read-Only (Auto-Run)
- **Operations**: Introspection, data queries, read-only analysis
- **Examples**:
  - `tasks_list` - Query task execution history
  - `tasks_by_trace` - Inspect specific execution trace
  - `tasks_last_failed` - Analyze failure patterns
- **Enforcement**: No restrictions, runs automatically
- **Budget**: Not budget-limited (can run frequently)

### Tier 1: Local Write (Auto-Run with Budget)
- **Operations**: Local repository changes, goal creation, belief updates
- **Examples**:
  - `execute_goal` - Generate and execute code
  - `astra.desires.record` - Record new desires
  - `astra.goals.create` - Create actionable goals
- **Enforcement**:
  - Runs automatically within per-day budget
  - Default budget: 4 executions per day
  - Budget resets daily at midnight UTC
- **Safeguards**:
  - Budget prevents runaway execution
  - Changes are local to repository
  - No external side effects

### Tier 2: External (Requires Approval)
- **Operations**: External API calls, deployments, notifications
- **Examples**:
  - `deploy_to_production` - Deploy code changes
  - `send_email` - Send external notifications
  - `create_github_pr` - Create GitHub pull request
- **Enforcement**:
  - Requires approval token in `var/approvals/pending/<schedule_id>.token`
  - Token format: `{"approved_by": "human", "approved_at": "2025-11-11T12:00:00Z"}`
  - Token must exist and be valid before execution
  - Token is consumed after single use
- **Safeguards**:
  - Explicit human approval required
  - One-time use tokens
  - Audit trail in approval log

## Budget Tracking

Budgets apply to Tier 1 schedules only:

```python
class RunBudget:
    per_day: int = 4          # Max runs per day
    consumed: int = 0         # Runs consumed today
    last_reset: str           # ISO8601 timestamp of last reset
```

Budget enforcement:
1. Check `schedule.run_budget.consumed < schedule.run_budget.per_day`
2. If budget exhausted, skip execution
3. Reset `consumed = 0` when date changes from `last_reset`
4. Increment `consumed` after successful execution

## Executor Implementation

The scheduler executor (not yet implemented) will enforce these rules:

```python
async def execute_scheduled_task(schedule: Schedule):
    """Execute a scheduled task with safety tier enforcement."""

    # Check if due
    if not is_due(schedule):
        return

    # Tier 0: Always run
    if schedule.safety_tier == SafetyTier.READ_ONLY:
        await execute_tool(schedule.target_tool, schedule.payload)
        return

    # Tier 1: Check budget
    if schedule.safety_tier == SafetyTier.LOCAL_WRITE:
        if not check_budget(schedule):
            logger.info(f"Schedule {schedule.id} budget exhausted, skipping")
            return
        await execute_tool(schedule.target_tool, schedule.payload)
        mark_executed(schedule.id)
        return

    # Tier 2: Check approval token
    if schedule.safety_tier == SafetyTier.EXTERNAL:
        token_path = f"var/approvals/pending/{schedule.id}.token"
        if not Path(token_path).exists():
            logger.info(f"Schedule {schedule.id} requires approval, skipping")
            return

        # Validate token
        token = json.loads(Path(token_path).read_text())
        if not validate_approval_token(token):
            logger.warning(f"Invalid approval token for {schedule.id}")
            return

        # Execute and consume token
        await execute_tool(schedule.target_tool, schedule.payload)
        Path(token_path).unlink()  # One-time use
        mark_executed(schedule.id)
        return
```

## Approval Workflow for Tier 2

Human operator approves tier-2 operations:

```bash
# List pending tier-2 schedules
python -m scripts.list_pending_approvals

# Approve a specific schedule
python -m scripts.approve_schedule sch_abc12345

# This creates: var/approvals/pending/sch_abc12345.token
```

The approval token contains:
```json
{
  "schedule_id": "sch_abc12345",
  "schedule_name": "deploy_to_production",
  "approved_by": "human@example.com",
  "approved_at": "2025-11-11T12:34:56Z",
  "expires_at": "2025-11-11T13:34:56Z"
}
```

## Audit Trail

All schedule executions are logged to:
- **NDJSON chain**: `var/schedules/YYYY-MM.ndjson.gz`
- **Raw store**: Task execution records with `scheduled_vs_manual: "scheduled"`
- **Approval log**: `var/approvals/approved.ndjson.gz` (for tier-2)

This provides complete traceability:
- What ran when (NDJSON timestamps)
- Who approved tier-2 operations (approval tokens)
- What changes were made (task execution records)
- Budget consumption over time (schedule index)

## Examples

### Tier 0: Daily introspection
```python
schedule_service.create(
    name="morning_health_check",
    cron="0 9 * * *",  # 9 AM daily
    target_tool="tasks_last_failed",
    payload={"limit": 10},
    safety_tier=SafetyTier.READ_ONLY,
    per_day_budget=0,  # Not budget-limited
)
```

### Tier 1: Autonomous goal execution
```python
schedule_service.create(
    name="autonomous_coding",
    cron="0 */4 * * *",  # Every 4 hours
    target_tool="execute_goal",
    payload={"goal_source": "top_desire"},
    safety_tier=SafetyTier.LOCAL_WRITE,
    per_day_budget=4,  # Max 4 goals per day
)
```

### Tier 2: Deploy to production (requires approval)
```python
schedule_service.create(
    name="production_deploy",
    cron="0 0 * * 1",  # Monday midnight
    target_tool="deploy_to_vercel",
    payload={"environment": "production"},
    safety_tier=SafetyTier.EXTERNAL,
    per_day_budget=1,  # Just in case, but approval is primary gate
)
```

## Design Rationale

This 3-tier model provides:

1. **Autonomy for safe operations** - Tier 0/1 can run without human intervention
2. **Safety guardrails** - Budgets prevent runaway Tier 1 execution
3. **Explicit approval for risky operations** - Tier 2 requires human decision
4. **Auditability** - Complete trail of what ran, when, and who approved it
5. **Gradual trust** - Start with low tiers, increase as confidence grows

The system is designed to let Astra be autonomous for development tasks (Tier 0/1) while keeping humans in the loop for production changes (Tier 2).
