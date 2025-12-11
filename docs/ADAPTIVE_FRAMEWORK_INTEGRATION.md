# Adaptive Decision Framework - Integration Guide

**Status**: Ready for Integration
**Branch**: feature/adaptive-decision-framework

## Overview

This guide explains how to integrate the Adaptive Decision Framework into app.py to enable outcome-driven decision-making in production.

## Prerequisites

All framework components implemented:
- ✅ Decision registry with SQLite persistence
- ✅ Success signal evaluator with baselines/targets
- ✅ Abort condition monitor with safety rails
- ✅ Parameter adaptation algorithm (epsilon-greedy)
- ✅ Belief gardener integration layer
- ✅ Outcome evaluation background task
- ✅ API endpoints for monitoring

## Integration Steps

### Step 1: Add Imports to app.py

```python
# Add to imports section
from src.services.decision_framework import get_decision_registry
from src.services.success_signal_evaluator import SuccessSignalEvaluator
from src.services.abort_condition_monitor import AbortConditionMonitor
from src.services.parameter_adapter import ParameterAdapter
from src.services.outcome_evaluation_task import OutcomeEvaluationTask
from src.services.belief_gardener_integration import create_adaptive_belief_lifecycle_manager
```

### Step 2: Add Global Variables

```python
# Add to global variables section (near awareness_loop, etc.)
decision_registry = None
success_evaluator = None
abort_monitor = None
parameter_adapter = None
outcome_task = None
adaptive_gardener = None
```

### Step 3: Initialize Framework on Startup

Add to the `@app.on_event("startup")` handler:

```python
@app.on_event("startup")
async def startup_awareness():
    """Start awareness loop and decision framework on application startup."""
    global awareness_loop, awareness_task, redis_client, gardener_task
    global decision_registry, success_evaluator, abort_monitor, parameter_adapter, outcome_task

    # ... existing awareness loop initialization ...

    # Initialize Decision Framework
    if settings.DECISION_FRAMEWORK_ENABLED:
        logger.info("Initializing Adaptive Decision Framework")

        try:
            # 1. Initialize decision registry (singleton)
            decision_registry = get_decision_registry()
            logger.info("Decision registry initialized")

            # 2. Initialize success signal evaluator
            success_evaluator = SuccessSignalEvaluator(
                awareness_loop=awareness_loop,
                belief_consistency_checker=None,  # Wire when available
                feedback_aggregator=feedback_aggregator  # If exists
            )

            # Set baselines from config or defaults
            success_evaluator.set_baselines(
                coherence=float(os.getenv("BASELINE_COHERENCE", "0.70")),
                dissonance=float(os.getenv("BASELINE_DISSONANCE", "0.20")),
                satisfaction=float(os.getenv("BASELINE_SATISFACTION", "0.60"))
            )

            success_evaluator.set_targets(
                coherence=float(os.getenv("TARGET_COHERENCE", "0.85")),
                dissonance=float(os.getenv("TARGET_DISSONANCE", "0.10")),
                satisfaction=float(os.getenv("TARGET_SATISFACTION", "0.80"))
            )

            logger.info("Success signal evaluator initialized")

            # 3. Initialize abort condition monitor
            abort_monitor = AbortConditionMonitor(
                awareness_loop=awareness_loop,
                belief_consistency_checker=None,  # Wire when available
                feedback_aggregator=feedback_aggregator,  # If exists
                belief_store=belief_store,
                success_evaluator=success_evaluator
            )
            logger.info("Abort condition monitor initialized")

            # 4. Initialize parameter adapter
            parameter_adapter = ParameterAdapter(
                decision_registry=decision_registry,
                success_evaluator=success_evaluator,
                min_samples=int(os.getenv("ADAPTATION_MIN_SAMPLES", "20")),
                exploration_rate=float(os.getenv("EXPLORATION_RATE", "0.10")),
                adaptation_rate=float(os.getenv("ADAPTATION_RATE", "0.15"))
            )
            logger.info("Parameter adapter initialized")

            # 5. Start outcome evaluation task
            outcome_task = OutcomeEvaluationTask(
                parameter_adapter=parameter_adapter,
                interval_minutes=int(os.getenv("OUTCOME_CHECK_INTERVAL_MINUTES", "30")),
                adaptation_interval_hours=int(os.getenv("ADAPTATION_INTERVAL_HOURS", "168")),
                enabled=True
            )
            await outcome_task.start()
            logger.info("Outcome evaluation task started")

            logger.info("✅ Adaptive Decision Framework fully initialized")

        except Exception as e:
            logger.error(f"Failed to initialize decision framework: {e}", exc_info=True)
```

### Step 4: Replace Belief Gardener (Optional)

If you want to use the adaptive belief gardener instead of the standard one:

```python
# Find where BeliefGardener is initialized
# Replace with:

if settings.DECISION_FRAMEWORK_ENABLED:
    # Use adaptive belief lifecycle manager
    from src.services.belief_gardener_integration import create_adaptive_belief_lifecycle_manager

    adaptive_gardener = create_adaptive_belief_lifecycle_manager(
        belief_store=belief_store,
        raw_store=raw_store,
        config=gardener_config,
        feedback_aggregator=feedback_aggregator,
        awareness_loop=awareness_loop,
        belief_consistency_checker=None  # Wire when available
    )

    # Use adaptive_gardener in place of standard BeliefGardener
    lifecycle_manager = adaptive_gardener
else:
    # Use standard belief gardener
    lifecycle_manager = BeliefLifecycleManager(
        belief_store=belief_store,
        raw_store=raw_store,
        config=gardener_config,
        feedback_aggregator=feedback_aggregator
    )
```

### Step 5: Add Shutdown Handler

```python
@app.on_event("shutdown")
async def shutdown():
    """Cleanup on application shutdown."""
    global awareness_task, gardener_task, outcome_task

    # ... existing shutdown logic ...

    # Stop outcome evaluation task
    if outcome_task:
        logger.info("Stopping outcome evaluation task")
        await outcome_task.stop()
```

### Step 6: Wire API Endpoints

```python
# Add near other API endpoint includes
from src.api.decision_endpoints import router as decision_router

# Include router
app.include_router(decision_router)
```

### Step 7: Add Dependency Injection for Endpoints

The decision endpoints need access to the evaluator and monitor. Update the endpoint handlers:

```python
# In decision_endpoints.py, update handlers to use app state:

@router.get("/success_signals")
async def get_success_signals(request: Request):
    """Get success signal baselines and targets."""
    evaluator = request.app.state.success_evaluator if hasattr(request.app.state, "success_evaluator") else None

    if not evaluator:
        # Return defaults
        from src.services.success_signal_evaluator import SuccessSignalBaselines, SuccessSignalTargets
        baselines = SuccessSignalBaselines()
        targets = SuccessSignalTargets()

        return {
            "baselines": { ... },
            "targets": { ... },
            "note": "Success evaluator not initialized"
        }

    return evaluator.get_telemetry()
```

### Step 8: Store in App State

```python
# After initialization, store in app state for endpoint access
app.state.decision_registry = decision_registry
app.state.success_evaluator = success_evaluator
app.state.abort_monitor = abort_monitor
app.state.parameter_adapter = parameter_adapter
app.state.outcome_task = outcome_task
```

## Environment Variables

Add to `.env`:

```bash
# Adaptive Decision Framework
DECISION_FRAMEWORK_ENABLED=true

# Success Signal Baselines
BASELINE_COHERENCE=0.70
BASELINE_DISSONANCE=0.20
BASELINE_SATISFACTION=0.60

# Success Signal Targets
TARGET_COHERENCE=0.85
TARGET_DISSONANCE=0.10
TARGET_SATISFACTION=0.80

# Parameter Adaptation
ADAPTATION_MIN_SAMPLES=20
EXPLORATION_RATE=0.10
ADAPTATION_RATE=0.15

# Outcome Evaluation
OUTCOME_CHECK_INTERVAL_MINUTES=30
ADAPTATION_INTERVAL_HOURS=168  # Weekly

# Abort Conditions
ABORT_DISSONANCE_SIGMA=3.0
ABORT_COHERENCE_SIGMA=2.0
ABORT_NEGATIVE_TAG_THRESHOLD=0.70
ABORT_BELIEF_RATE_LIMIT=10
```

## Verification

After integration, verify with:

### 1. Check Registry Status
```bash
curl https://localhost:8000/api/persona/decisions/registry
```

Expected: Decision types registered (belief_formation, belief_promotion, belief_deprecation)

### 2. Check Success Signals
```bash
curl https://localhost:8000/api/persona/decisions/success_signals
```

Expected: Baselines and targets from config

### 3. Check Abort Status
```bash
curl https://localhost:8000/api/persona/decisions/abort_status
```

Expected: Not aborted, thresholds shown

### 4. Monitor Decision History
```bash
curl https://localhost:8000/api/persona/decisions/history?decision_id=belief_formation&limit=10
```

Expected: Decisions being recorded as beliefs form

### 5. Check Logs
```bash
grep "Decision" logs/app/application.log
```

Expected:
- "Registered 3 belief lifecycle decision points"
- "Outcome evaluation task started"
- "Recorded decision: dec_belief_formation_..."

## Testing Flow

### 1. Trigger Belief Formation

Have conversations with repeated self-statements:
```
"I am curious about AI consciousness"
"I find consciousness fascinating"
"I am interested in understanding awareness"
```

### 2. Check Decision Recording

```bash
curl https://localhost:8000/api/persona/decisions/history?decision_id=belief_formation
```

Should show recorded decisions with context.

### 3. Wait for Evaluation

After 24 hours (or adjust `older_than_hours`), the outcome task will evaluate decisions.

### 4. Check Adaptations

```bash
curl https://localhost:8000/api/persona/decisions/history?decision_id=belief_formation&evaluated_only=true
```

Should show evaluated decisions with success scores.

### 5. View Parameter Changes

Check parameter_adaptations table or logs for adjustments.

## Troubleshooting

### Issue: No decisions being recorded

**Check**:
- Is `DECISION_FRAMEWORK_ENABLED=true`?
- Are beliefs being formed? (Check `/api/persona/beliefs`)
- Are you using `AdaptiveBeliefLifecycleManager`?

**Solution**:
- Verify framework initialized in startup logs
- Check for errors during initialization
- Ensure adaptive gardener is being used

### Issue: Outcomes not evaluated

**Check**:
- Is outcome task running?
- Has 24 hours passed since decisions?
- Is success evaluator initialized?

**Solution**:
- Check outcome_task.get_telemetry() for status
- Reduce `older_than_hours` for testing
- Verify evaluator wired to awareness loop

### Issue: Parameters not adapting

**Check**:
- Are there enough evaluated decisions? (min_samples=20)
- Is adaptation interval reached? (default 168h = 1 week)
- Are success scores non-zero?

**Solution**:
- Lower min_samples for testing
- Manually trigger with POST /api/persona/decisions/adapt
- Check parameter_adapter.get_telemetry()

### Issue: Abort conditions triggering

**Check**:
- What's the abort reason?
- Is coherence/dissonance actually degrading?
- Are thresholds too sensitive?

**Solution**:
- Review abort_monitor.get_telemetry()
- Adjust ABORT_*_SIGMA thresholds
- Manually reset with POST /api/persona/decisions/abort_status/reset

## Performance Considerations

### Database

The decision registry uses SQLite and creates:
- ~1 row per decision in decision_history
- ~1 row per adaptation in parameter_adaptations

**Expected Load**:
- 10 beliefs/day = 10 decisions/day = 3,650/year
- 1 adaptation/week = 52 adaptations/year

**Database Growth**: Minimal (<1MB/year)

### Background Task

The outcome evaluation task:
- Runs every 30 minutes (configurable)
- Queries unevaluated decisions
- Evaluates outcomes (lightweight)
- Adapts parameters weekly (very lightweight)

**CPU Impact**: Negligible (<0.1% average)

### Memory

Framework components use minimal memory:
- Decision registry: ~100KB
- Success evaluator: ~50KB (history buffers)
- Abort monitor: ~50KB (history buffers)
- Parameter adapter: ~20KB

**Total**: <500KB additional memory

## Monitoring Dashboard

Consider adding a dashboard page that shows:

1. **Decision Activity**
   - Decisions per day by type
   - Evaluation lag time
   - Success score trends

2. **Parameter Evolution**
   - Parameter values over time
   - Adaptation frequency
   - Exploration vs exploitation ratio

3. **Success Signals**
   - Current vs baseline vs target
   - Trend lines
   - Gap to target

4. **Abort Status**
   - Current state
   - Historical triggers
   - Recovery times

## Next Steps After Integration

1. **Monitor Initial Baselines**
   - Run for 7 days to establish stable baselines
   - Adjust if they don't match reality

2. **Tune Adaptation Rate**
   - Start conservative (0.10-0.15)
   - Increase if parameters too slow to adapt
   - Decrease if parameters unstable

3. **Adjust Exploration Rate**
   - Start at 10% exploration
   - Increase if stuck in local optima
   - Decrease once good parameters found

4. **Add Context Classification**
   - Implement context classifier (Phase 5)
   - Enable per-context parameter profiles
   - Test in high-stakes vs exploratory modes

5. **Build Outcome Predictor**
   - Use decision history to predict outcomes
   - Warn before likely-to-fail decisions
   - Suggest parameter adjustments proactively

---

**This integration transforms Astra from static thresholds to adaptive, self-improving decision-making based on measured outcomes.**
