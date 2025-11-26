# Runbook: When Astra Feels Weird

Quick diagnostic guide for when Astra's behavior seems off.

## Quick Health Check

```bash
# Get current IL state
curl -s http://localhost:8443/api/integration/state | jq .

# Check focus stack
curl -s http://localhost:8443/api/integration/focus | jq .

# Check budgets
curl -s http://localhost:8443/api/integration/budgets | jq .

# Recent history
curl -s http://localhost:8443/api/integration/history?limit=10 | jq .

# Prometheus metrics
curl -s http://localhost:8443/api/integration/metrics
```

## Symptom: Astra is slow to respond

### Check tick duration
```bash
curl -s http://localhost:8443/api/integration/metrics | grep tick_duration
```

If p95 > 1s:
1. Check which action types are being dispatched (history endpoint)
2. Look for INTROSPECTION or BELIEF_GARDENING running during interactive mode
3. Check if awareness_loop or belief_gardener are slow

### Check cognitive load
```bash
curl -s http://localhost:8443/api/integration/state | jq '.modulation.cognitive_load'
```

If > 0.8:
- Focus stack may be full (7 items max)
- Too many active goals competing
- Consider clearing stale focus items

## Symptom: Astra seems confused or inconsistent

### Check identity drift
```bash
curl -s http://localhost:8443/api/integration/state | jq '.identity.anchor_drift'
```

If > 0.15:
- Core beliefs may be shifting
- Check `astra_beliefs_formed_today` - too many new beliefs?
- Review recent belief gardening actions in history

### Check for dissonance
```bash
curl -s http://localhost:8443/api/integration/state | jq '.signals.dissonance_alert_count'
```

If > 0:
- Unresolved belief conflicts exist
- Check focus stack for dissonance items
- May need manual intervention or belief review

## Symptom: Astra not doing autonomous tasks

### Check execution mode
```bash
curl -s http://localhost:8443/api/integration/state | jq '.mode'
```

Expected modes:
- `interactive`: User is active, prioritizes responses
- `autonomous`: No user, pursues goals
- `maintenance`: Background consolidation (2am-5am)

To force mode change:
```bash
curl -X POST http://localhost:8443/api/integration/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "autonomous", "reason": "manual_override"}'
```

### Check budget exhaustion
```bash
curl -s http://localhost:8443/api/integration/budgets | jq '.tokens, .beliefs'
```

If tokens_available < 100 or beliefs.form_remaining == 0:
- Budget exhausted, wait for reset
- Or manually adjust limits in config

## Symptom: Astra not forming/updating beliefs

### Check belief budgets
```bash
curl -s http://localhost:8443/api/integration/budgets | jq '.beliefs'
```

Key values:
- `form_remaining`: Can form new beliefs?
- `promote_remaining`: Can promote beliefs to core?
- `deprecate_remaining`: Can deprecate beliefs?

### Check BELIEF_MUTATIONS_ENABLED
If beliefs not changing at all, check if mutations are frozen:
```bash
grep BELIEF_MUTATIONS_ENABLED .env
```

## Log Locations

- Integration Layer: `grep "\[IL\]" logs/astra.log`
- Belief changes: `grep "belief" logs/astra.log`
- Mode transitions: `grep "Mode transition" logs/astra.log`

## Emergency Actions

### Force maintenance mode (stop all actions)
```bash
curl -X POST http://localhost:8443/api/integration/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "maintenance", "reason": "emergency_stop"}'
```

### Check if IL is running at all
```bash
curl -s http://localhost:8443/api/integration/state
# 503 = IL not initialized
# 200 = IL running
```

## Metrics to Watch

| Metric | Normal | Warning | Critical |
|--------|--------|---------|----------|
| `astra_tick_duration_seconds` p95 | < 0.5s | 0.5-1.0s | > 1.0s |
| `astra_cognitive_load` | < 0.6 | 0.6-0.8 | > 0.8 |
| `astra_focus_stack_size` | 1-4 | 5-6 | 7 (max) |
| `astra_anchor_drift` | < 0.1 | 0.1-0.2 | > 0.2 |
| `astra_budget_tokens_available` | > 500 | 100-500 | < 100 |

## Contact

If issues persist after following this runbook, check:
1. Integration Layer logs for exceptions
2. Subsystem health (Redis, awareness_loop, belief_store)
3. Recent code changes on the branch
