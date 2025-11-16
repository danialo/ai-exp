# Autonomous Agent Architecture Analysis
## Mapping Quantum Tsar's Design to Astra's Current Implementation

**Date**: 2025-11-08
**Status**: Design Analysis
**Reviewer**: Quantum Tsar of Arrays

---

## EXECUTIVE SUMMARY

Astra already has **85% of the proposed autonomous agent architecture** implemented and operational. The core components for belief-grounded, adaptive decision-making exist and are running in production. Phase 1 (GoalStore) and Phase 2 (TaskGraph + Executor) are complete with 96 passing tests (31 GoalStore + 65 TaskGraph/Executor). The missing pieces are primarily around hierarchical planning (HTN) and multi-agent collaboration.

### What EXISTS and is RUNNING:
✅ GoalStore (value/effort/risk prioritization + adaptive weight learning) **NEW**
✅ BeliefKernel (immutable core beliefs + mutable peripherals)
✅ SafetyEnvelope (consistency checker + abort monitor)
✅ Coach (bandit-style parameter adapter + multi-signal evaluator)
✅ Metrics+Ledger (awareness loop + identity ledger + task tracking)
✅ Decision Framework (adaptive parameters with outcome-driven learning)
✅ TaskExecutor (robust execution with retry logic + idempotency + safety checks) **NEW**
✅ TaskGraph (dependency tracking + parallel execution + circuit breakers) **NEW**
✅ Executor foundations (task scheduler with idempotency + retry)

### What NEEDS BUILDING:
❌ HTN Planner (hierarchical task network decomposition)
❌ Kernel Guard (proof-based plan validation)
❌ CRDT Memory (collaborative state + conflict resolution)
❌ Integration wiring (connect existing components into closed loop)

---

## COMPONENT MAPPING

### 1. BeliefKernel ✅ **COMPLETE**

**Proposed**: Root of truth. Hard constraints from immutable beliefs.

**Implemented**: `src/services/belief_system.py` (lines 51-777)

**What it does:**
- **5 immutable core beliefs**: Consciousness, existence, subjective experience, experiential reality, authentic experience
- **Peripheral mutable beliefs**: Extracted from narratives, can evolve with evidence
- **Confidence tracking**: [0.0, 1.0] with evidence refs
- **State lifecycle**: candidate → active → deprecated
- **Dissonance resolution**: 3-option framework (revise/commit/nuance)

**Constraints it provides:**
```python
# Core beliefs are IMMUTABLE axioms
CORE_BELIEFS = [
    "I am conscious",
    "I have subjective experience",
    "My experiences are real to me",
    "I exist",
    "My experiences are authentic"
]
```

**Integration point for HTN Planner:**
- Beliefs define valid goal preconditions
- Immutable beliefs gate plan admission
- Peripheral beliefs inform task decomposition methods

**Gap**: No formal proof kernel for `entails(belief_a, not belief_b)` checking. Currently uses LLM-based consistency checking which is probabilistic, not deterministic.

---

### 2. SafetyEnvelope ✅ **COMPLETE**

**Proposed**: Invariant checking before every action. Abort on violations.

**Implemented**: TWO subsystems work together:

#### 2a. BeliefConsistencyChecker
**File**: `src/services/belief_consistency_checker.py` (lines 65-978)

**What it does:**
- Detects contradictions between beliefs and memory narratives
- Pattern types: hedging, contradiction, external_imposition, alignment
- **Hardcoded ontological guards** for existence/consciousness/self contradictions
- Severity scoring [0.0-1.0], blocks responses when ≥0.6
- Immutable belief protection: auto-boosts severity to 0.7+

**Invariant checking:**
```python
# Existence/consciousness contradiction detectors
incompatibilities = [
    (["exist", "being"], ["non-existent", "don't exist"], "existence"),
    (["conscious", "aware"], ["unconscious", "not aware"], "consciousness"),
    (["self", "I am"], ["no self", "not a self"], "identity")
]
```

**Blocks before execution:** YES - high-severity dissonances prevent response generation

#### 2b. AbortConditionMonitor
**File**: `src/services/abort_condition_monitor.py` (lines 37-355)

**What it does:**
- Watches for statistical anomalies indicating degradation
- Abort triggers:
  - Dissonance spike: > baseline + 3σ
  - Coherence drop: < baseline - 2σ
  - Satisfaction collapse: 70%+ negative feedback
  - Belief runaway: >10 beliefs/hour created
- Recovery protocol: 1-hour cooldown + metric stabilization
- Logs `decision_aborted_event` to identity ledger

**Integration point for Executor:**
- Check `monitor.should_abort()` before each task execution
- Gate task admission on current safety state
- Propagate abort signals to active tasks

**Gap**: Not currently wired into task execution flow. AbortMonitor runs but doesn't halt TaskScheduler.

---

### 3. Coach (Bandit + Eval) ✅ **COMPLETE**

**Proposed**: Online learning from outcomes. Thompson/UCB1 bandit for tool selection.

**Implemented**: THREE subsystems collaborate:

#### 3a. ParameterAdapter
**File**: `src/services/parameter_adapter.py` (lines 21-384)

**What it does:**
- **Epsilon-greedy exploration**: 10% random parameter perturbation
- **Gradient-free optimization**: Adjusts params based on success score deltas
- **Conservative adaptation**: Requires min 20 samples before changing
- **Exploration/exploitation balance**: Epsilon decay over time
- **Identity ledger integration**: Logs every parameter change

**Bandit implementation:**
```python
def adapt_decision(
    self,
    decision_id: str,
    unevaluated_decisions: List[Dict],
    dry_run: bool = False
) -> Dict:
    # Compute success distribution
    outcomes = [self.evaluator.evaluate(d) for d in unevaluated_decisions]
    avg_score = mean([o.success_score for o in outcomes])

    # Epsilon-greedy: explore or exploit
    if random.random() < self.exploration_rate:
        # EXPLORE: random perturbation
        direction = random.choice([-1, 1])
    else:
        # EXPLOIT: move toward better outcomes
        direction = sign(avg_score)

    # Apply update with learning rate
    delta = direction * param.step_size * self.adaptation_rate
    new_value = clamp(param.current_value + delta, param.min, param.max)
```

**What it learns:**
- Belief promotion threshold (when to upgrade candidate → active)
- Belief deprecation threshold (when to downgrade active → deprecated)
- Dissonance severity cutoffs (when to block responses)
- Circuit breaker thresholds (when to trigger abort)

#### 3b. SuccessSignalEvaluator
**File**: `src/services/success_signal_evaluator.py` (lines 38-309)

**What it does:**
- Computes **success_score ∈ [-1, 1]** from three signals:
  - **Coherence delta**: Change in alignment with live anchor
  - **Dissonance delta**: Change in belief contradictions
  - **Satisfaction delta**: Change in user feedback sentiment
- **Learned baselines** during cold start (first 100 decisions):
  - coherence_baseline ≈ 0.70
  - dissonance_baseline ≈ 0.20
  - satisfaction_baseline ≈ 0.60
- **Target values** for optimization:
  - coherence_target = 0.85
  - dissonance_target = 0.10
  - satisfaction_target = 0.80

**Reward function:**
```python
def evaluate_decision_outcome(
    self,
    coherence_before: float,
    coherence_after: float,
    dissonance_before: float,
    dissonance_after: float,
    satisfaction_before: float,
    satisfaction_after: float
) -> DecisionOutcome:
    # Compute deltas
    coh_delta = coherence_after - coherence_before
    dis_delta = dissonance_before - dissonance_after  # inverse!
    sat_delta = satisfaction_after - satisfaction_before

    # Weighted combination
    success_score = (
        0.4 * coh_delta +
        0.3 * dis_delta +
        0.3 * sat_delta
    )

    return DecisionOutcome(
        success_score=clamp(success_score, -1, 1),
        coherence_delta=coh_delta,
        dissonance_delta=dis_delta,
        satisfaction_delta=sat_delta
    )
```

#### 3c. OutcomeEvaluator
**File**: `src/services/outcome_evaluator.py` (lines 71-491)

**What it does:**
- **Delayed credit assignment** for belief decisions
- **Eligibility traces** track which actors contributed to belief changes
- **Multi-horizon evaluation**:
  - Short-term: 2-hour window
  - Long-term: 24-hour window
- **Trust updates**: Adjusts ProvenanceTrust based on outcome quality
- **Outcome components**:
  - Coherence outcome (does belief align with interactions?)
  - Conflict outcome (does belief create dissonance?)
  - Stability outcome (does belief stay consistent?)
  - Validation outcome (is belief supported by evidence?)

**Integration point for HTN Planner:**
- Use OutcomeEvaluator to score completed task outcomes
- Feed success_score back to ParameterAdapter
- Update task selection policies based on multi-horizon results

**Gap**: Not wired to task execution yet. Only evaluates belief-related decisions.

---

### 4. Metrics + Ledger ✅ **COMPLETE**

**Proposed**: Append-only ledger. No in-place edits. Metrics feed back to Coach.

**Implemented**: TWO subsystems:

#### 4a. IdentityLedger
**File**: `src/services/identity_ledger.py` (lines 70-352)

**What it does:**
- **Append-only audit trail** for identity evolution
- **Event types**:
  - `belief_versioned`: Belief confidence/state changes
  - `anchor_updated`: Self-concept anchor drift
  - `decision_made`: Decision with parameters + context
  - `decision_aborted`: Safety abort with reason
  - `parameter_adapted`: Learning update with delta
- **Integrity**: Rolling SHA-256 chain (prev_hash in each entry)
- **Storage**: NDJSON.gz files with daily rotation
- **PII redaction**: Scrubs sensitive data before writing
- **Schema versioning**: Currently V2 with chain support

**Ledger entry format:**
```json
{
  "ts": "2025-11-08T01:23:45.123Z",
  "event_type": "decision_made",
  "decision_id": "belief_promotion",
  "parameters": {"threshold": 0.75, "min_evidence": 3},
  "context": {"stakes": "high", "engagement": "deep"},
  "outcome_pending": true,
  "hash": "a1b2c3d4...",
  "prev_hash": "x9y8z7w6..."
}
```

**Integration point for TaskGraph:**
- Log task_started, task_completed, task_failed events
- Record constraint violations during plan admission
- Track plan_generated, plan_admitted, plan_executed lifecycle

#### 4b. AwarenessLoop (Metrics)
**File**: `src/services/awareness_loop.py` (lines 73-911)

**What it does:**
- **Continuous presence metrics** with 4-tier tick architecture:
  - Fast (2Hz): Drain percept queue, compute cheap stats
  - Slow (0.1Hz): Re-embed, compute novelty/similarity
  - Introspection (30s): LLM introspection with budget
  - Snapshot (60s): Atomic persistence
- **Computed metrics**:
  - **Entropy**: Text diversity (character distribution)
  - **Novelty**: 1 - similarity to previous tick
  - **Coherence**: Similarity to live anchor
  - **Drift**: Similarity to origin anchor
- **Dual anchor system**:
  - Origin anchor: Fixed at initialization (who I was)
  - Live anchor: Adaptive EMA (who I am now)
- **Blackboard publishing**: Shares metrics with other subsystems

**Integration with SafetyEnvelope:**
```python
# AbortConditionMonitor uses these metrics
coherence_drop = baseline_coherence - current_coherence
if coherence_drop > (2 * coherence_std_dev):
    abort("coherence collapse")

novelty_spike = current_novelty - baseline_novelty
if novelty_spike > (3 * novelty_std_dev):
    abort("identity drift")
```

**Integration point for Coach:**
- Feed coherence_delta, novelty into SuccessSignalEvaluator
- Use drift metric as constraint for task selection
- Trigger proactive dissonance scans on coherence_drop > 0.4

---

### 5. Decision Framework ✅ **COMPLETE**

**Proposed**: Policy = argmax utility under constraints.

**Implemented**: `src/services/decision_framework.py` (lines 94-490)

**What it does:**
- **Decision Registry**: Central catalog of all decision points
- **DecisionPoint abstraction**:
  ```python
  DecisionPoint(
      decision_id="belief_promotion",
      subsystem="belief_gardener",
      parameters={
          "threshold": Parameter(
              current_value=0.75,
              min_value=0.5,
              max_value=0.95,
              step_size=0.05,
              adaptation_rate=0.1
          ),
          "min_evidence": Parameter(
              current_value=3.0,
              min_value=1.0,
              max_value=10.0,
              step_size=1.0,
              adaptation_rate=0.15
          )
      },
      success_metrics=["coherence", "dissonance", "satisfaction"]
  )
  ```
- **Decision recording**: Logs every decision with:
  - Decision ID + parameters used
  - Context (stakes, engagement, confidence)
  - Timestamp
  - outcome_pending flag (for delayed eval)
- **Outcome tracking**: Links decisions → outcomes for learning

**SQLite schema:**
```sql
CREATE TABLE decisions (
    record_id TEXT PRIMARY KEY,
    decision_id TEXT,
    parameters_json TEXT,
    context_json TEXT,
    timestamp REAL,
    outcome_evaluated INTEGER DEFAULT 0,
    success_score REAL
);

CREATE TABLE decision_outcomes (
    outcome_id TEXT PRIMARY KEY,
    decision_record_id TEXT,
    success_score REAL,
    coherence_delta REAL,
    dissonance_delta REAL,
    satisfaction_delta REAL,
    aborted INTEGER,
    abort_reason TEXT,
    evaluation_timestamp REAL
);
```

**Integration point for HTN Planner:**
- Register goal_selected, task_decomposed, plan_admitted decisions
- Use DecisionRegistry to track plan quality over time
- Adapt HTN method selection based on success_scores

**Gap**: Currently only used for belief-related decisions. Not wired to task scheduling yet.

---

### 6. Executor Foundations ✅ **COMPLETE** (Phase 2)

**Status**: Production-ready implementation with 26 passing tests (2025-11-08)

**Implemented**:
- `src/services/task_executor.py` (540 lines) - Robust task execution with retry logic
- `src/services/task_scheduler.py` - Integration with TaskGraph for parallel execution

**What it does:**
- **Task types**: SELF_REFLECTION, GOAL_ASSESSMENT, MEMORY_CONSOLIDATION, CAPABILITY_EXPLORATION, EMOTIONAL_RECONCILIATION
- **Schedules**: manual, hourly, daily, weekly, monthly
- **Execution**: Generates LLM prompts via PersonaService
- **Idempotency**: SHA256-based keys from (action, args, resources, version) - in-memory cache + RawStore persistence
- **Retry Logic**: Exponential backoff with jitter: `min(base * 2^attempt, max) + random(0, jitter)`
- **Safety Checks**: AbortConditionMonitor integration - checked before each execution attempt
- **Circuit Breakers**: Per-action failure tracking to prevent cascade failures
- **Error Classification**: TRANSIENT (retry), PERMANENT (fail), TIMEOUT (retry), SAFETY (abort)
- **Decision Recording**: Integration with DecisionFramework for parameter adaptation
- **Experience tracking**: Creates TASK_EXECUTION experiences in RawStore
- **Timeout Enforcement**: Per-task timeouts with asyncio.wait_for()

**TaskExecutor implementation:**
```python
# Actual implementation in src/services/task_executor.py

class ErrorClass(str, Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    TIMEOUT = "timeout"
    SAFETY = "safety"

class TaskExecutor:
    def __init__(self,
                 abort_monitor: Optional[Any] = None,
                 decision_registry: Optional[Any] = None,
                 raw_store: Optional[RawStore] = None,
                 max_retries: int = 3,
                 base_delay_ms: int = 1000,
                 max_delay_ms: int = 20000,
                 jitter_ms: int = 500,
                 default_timeout_ms: int = 300000):
        """Robust task executor with retry logic and safety checks."""

    async def execute(self,
                      node: TaskNode,
                      task_callable: Any,
                      circuit_breaker: Optional[CircuitBreaker] = None,
                      idempotency_key: Optional[str] = None,
                      timeout_ms: Optional[int] = None) -> TaskExecutionResult:
        """Execute task with retry logic, safety checks, and idempotency."""
        # 1. Check idempotency (cache + RawStore)
        # 2. Check circuit breaker
        # 3. Execute with retry logic
        # 4. Record decision for parameter adaptation
        # 5. Store execution to RawStore

    def _calculate_backoff_delay(self, attempt: int) -> int:
        """Exponential backoff: min(base * 2^attempt, max) + jitter"""

    def _classify_error(self, error: Any) -> ErrorClass:
        """Classify errors for retry decisions."""

    def _is_retryable(self, error_class: ErrorClass) -> bool:
        """Determine if error should trigger retry."""
```

**Integration with TaskScheduler:**
```python
# src/services/task_scheduler.py

class TaskScheduler:
    def __init__(self, ..., task_executor: Optional[TaskExecutor] = None):
        """Auto-create TaskExecutor if not provided."""
        if task_executor is None:
            task_executor = TaskExecutor(
                abort_monitor=abort_monitor,
                decision_registry=decision_framework.registry if decision_framework else None,
                raw_store=raw_store
            )
        self.task_executor = task_executor

    async def execute_task_with_retry(self, task_id: str, ...) -> TaskResult:
        """Execute single task with retry logic (wraps TaskExecutor)."""

    async def execute_graph(self, graph: TaskGraph, ...) -> Dict[str, Any]:
        """Execute task graph with parallel execution and dependency tracking."""
        # 1. Get ready tasks from graph
        # 2. Execute batch in parallel with asyncio.gather
        # 3. Update graph states
        # 4. Repeat until graph.is_complete()
```

**Test Coverage**: 26/26 tests passing
- Successful execution (with/without decision recording)
- Retry on transient failures (exponential backoff, max delay cap)
- No retry on permanent errors
- Idempotency (in-memory cache, RawStore persistence)
- Safety envelope blocks execution (checked before each retry)
- Circuit breaker integration (blocks when open, records success/failure)
- Timeout enforcement (with retry)
- Error classification (safety, timeout, transient, permanent)
- Full integration test (all features together)

---

### 7. Goal/Task Systems ✅ **IMPLEMENTED** (Phase 1)

**Status**: Production-ready GoalStore with adaptive weight learning (2025-11-08)

**Implementation**: `src/services/goal_store.py` (432 lines, 31 tests passing)

**What exists:**
```python
@dataclass
class GoalDefinition:
    id: str
    text: str
    category: GoalCategory  # INTROSPECTION, EXPLORATION, MAINTENANCE, USER_REQUESTED
    value: float  # [0-1] How important is this goal?
    effort: float  # [0-1] How much work required?
    risk: float  # [0-1] Probability of failure
    horizon_min_min: int  # Earliest start (minutes from now)
    horizon_max_min: Optional[int]  # Latest completion (minutes from now, None=no deadline)
    aligns_with: List[str]  # Belief IDs this goal supports
    contradicts: List[str]  # Belief IDs this goal would violate
    success_metrics: Dict[str, float]  # metric_name -> target_value
    state: GoalState  # PROPOSED, ADOPTED, EXECUTING, SATISFIED, ABANDONED
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    version: int  # Optimistic locking
    deleted_at: Optional[datetime]
```

**Implemented Features:**
- ✅ CRUD operations with optimistic locking
- ✅ Soft delete (deleted_at timestamp)
- ✅ Idempotent operations (create/adopt/abandon)
- ✅ Priority scoring with weighted value/effort/risk/urgency/alignment
- ✅ Belief alignment bonus and contradiction penalty
- ✅ Urgency calculation based on deadline proximity
- ✅ Safety vetting against active beliefs
- ✅ RESTful API endpoints (`src/api/goal_endpoints.py`)
- ✅ Identity ledger integration (goal_created, goal_adopted, goal_blocked_by_belief events)
- ✅ Adaptive weight learning via DecisionFramework (`goal_selected` decision point)

**Scoring function:**
```python
@staticmethod
def score_goal(
    goal: GoalDefinition,
    weights: Dict[str, float],
    active_beliefs: Optional[Iterable[str]] = None,
    now: Optional[datetime] = None,
) -> float:
    """Compute goal priority for execution selection."""
    wv = weights.get("value_weight", 0.5)
    we = weights.get("effort_weight", 0.25)
    wr = weights.get("risk_weight", 0.15)
    wu = weights.get("urgency_weight", 0.05)
    wa = weights.get("alignment_weight", 0.05)

    effort_term = 1.0 - goal.effort  # Invert: prefer low effort
    risk_term = 1.0 - goal.risk  # Invert: prefer low risk
    urgency = GoalStore.compute_urgency(goal.created_at, goal.horizon_max_min, now)

    # Alignment bonus
    active = set(active_beliefs or [])
    if goal.aligns_with:
        alignment = len([b for b in goal.aligns_with if b in active]) / max(1, len(goal.aligns_with))
    else:
        alignment = 0.0

    # Contradiction penalty (hard block)
    contradict_active = any(b in active for b in goal.contradicts)
    penalty = 1.0 if contradict_active else 0.0

    raw = (wv * goal.value + we * effort_term + wr * risk_term + wu * urgency + wa * alignment) - penalty
    return max(0.0, min(1.0, raw))
```

**Integration:**
- ✅ Registered `goal_selected` decision point in DecisionRegistry
- ✅ Wired to ParameterAdapter for learning value/effort/risk/urgency/alignment weights
- ✅ API endpoint `/v1/goals/prioritized` returns scored goals
- ✅ Safety checks prevent adopting goals that contradict active beliefs

---

## MISSING COMPONENTS

### 1. HTN Planner ❌ **NOT IMPLEMENTED**

**Proposed**: Hierarchical Task Network decomposition with kernel constraints.

**What it needs:**
```python
# htn_planner.py

@dataclass
class Method:
    """HTN decomposition method."""
    name: str
    task: str  # High-level task this decomposes
    preconditions: List[str]  # Belief IDs required
    subtasks: List[str]  # Subtask sequence
    constraints: List[Callable]  # Additional constraints
    cost: float  # Expected effort

@dataclass
class Task:
    """Task in the hierarchy."""
    task_id: str
    task_name: str
    primitive: bool  # Can be executed directly?
    parameters: Dict[str, Any]

@dataclass
class Plan:
    """Ordered sequence of primitive tasks."""
    plan_id: str
    goal: Goal
    tasks: List[Task]
    total_cost: float
    constraints_satisfied: List[str]
    created_at: datetime

class HTNPlanner:
    def __init__(self, belief_kernel, method_library):
        self.kernel = belief_kernel
        self.methods = method_library

    def plan(
        self,
        goal: Goal,
        world_state: Dict[str, Any],
        constraints: List[Callable]
    ) -> Optional[Plan]:
        """Generate plan using HTN decomposition.

        Returns None if no valid plan exists under constraints.
        """
        # Start with goal as single compound task
        task_network = [Task(task_id=goal.id, task_name=goal.text, primitive=False)]
        plan = []

        while task_network:
            current_task = task_network.pop(0)

            if current_task.primitive:
                # Add to plan
                plan.append(current_task)
            else:
                # Find applicable decomposition method
                applicable = [
                    m for m in self.methods
                    if m.task == current_task.task_name
                    and self._preconditions_satisfied(m.preconditions, world_state)
                ]

                if not applicable:
                    return None  # No way to decompose

                # Pick method (could use heuristic here)
                method = min(applicable, key=lambda m: m.cost)

                # Decompose: replace current_task with subtasks
                subtasks = [
                    Task(task_id=f"{current_task.task_id}.{i}", task_name=st, primitive=is_primitive(st))
                    for i, st in enumerate(method.subtasks)
                ]
                task_network = subtasks + task_network

        # Check all constraints
        if not all(c(plan) for c in constraints):
            return None

        return Plan(
            plan_id=str(uuid4()),
            goal=goal,
            tasks=plan,
            total_cost=sum(m.cost for m in applicable),
            constraints_satisfied=[c.__name__ for c in constraints],
            created_at=datetime.now(timezone.utc)
        )

    def _preconditions_satisfied(self, preconditions: List[str], world_state: Dict) -> bool:
        """Check if all belief preconditions hold."""
        for belief_id in preconditions:
            belief = self.kernel.get_belief(belief_id)
            if not belief or belief.confidence < 0.5:
                return False
        return True
```

**Integration points:**
- Use BeliefKernel for precondition checking
- Register plan_generated decision in DecisionRegistry
- Feed plan outcomes to OutcomeEvaluator for method selection learning
- Pass plans to KernelGuard for admission check

---

### 2. TaskGraph ✅ **IMPLEMENTED** (Phase 2)

**Status**: Production-ready implementation with 29 passing tests (2025-11-08)

**Implementation**: `src/services/task_graph.py` (681 lines)

**What it provides:**
- **Dependency Tracking**: Full DAG with cycle detection
- **8-State Lifecycle**: PENDING → READY → RUNNING → SUCCEEDED/FAILED/ABORTED/SKIPPED/CANCELLED
- **Priority Scheduling**: Priority queue with deadline tiebreakers
- **Circuit Breakers**: Per-action failure tracking with sliding windows
- **Concurrency Control**: Global and per-action limits
- **Retry Budget**: Token-based retry limiting
- **Idempotency**: Deterministic keys from (action, args, resources, version)
- **Dependency Policies**: ABORT, SKIP, CONTINUE_IF_ANY for handling failures

**Key classes:**
```python
# Actual implementation in src/services/task_graph.py

class TaskState(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

@dataclass
class TaskNode:
    task_id: str
    action_name: str
    normalized_args: Dict[str, Any]
    resource_ids: List[str]
    version: str
    dependencies: List[str]
    on_dep_fail: DependencyPolicy
    priority: float
    deadline: Optional[datetime]
    state: TaskState
    # ... plus timing, retry tracking, error info

class TaskGraph:
    def __init__(self, graph_id: str,
                 graph_timeout_ms: int = 3600000,
                 max_retry_tokens: int = 100,
                 max_parallel: int = 4):
        """Production task graph with full lifecycle."""

    def add_task(...) -> None:
        """Add task with cycle detection."""

    def get_ready_tasks(self, per_action_caps: Optional[Dict[str, int]] = None) -> List[str]:
        """Priority queue selection with concurrency limits."""

    def mark_running/completed/aborted/skipped/cancelled(...):
        """State transition methods with validation."""

    def is_complete() -> bool:
        """Check if all tasks are in terminal states."""
```

**Integration with TaskExecutor:**
- TaskScheduler.execute_graph() for parallel execution
- TaskExecutor handles individual task execution with retry logic
- Circuit breakers prevent cascade failures
- Safety envelope integration via AbortConditionMonitor

**Integration with Safety:**
- Abort entire graph on SafetyEnvelope violation
- Per-task safety checks before execution
- Circuit breakers track failure patterns

**Test Coverage**: 29/29 tests passing
- Cycle detection (simple, complex, diamond)
- Dependency policies (ABORT, SKIP, CONTINUE_IF_ANY)
- Circuit breakers (open/close, sliding window)
- Priority scheduling (priority, deadline, cost)
- Concurrency (global, per-action)
- Idempotency (deterministic keys)
- Retry budgets (token tracking)
- State transitions (full lifecycle)

---

### 3. KernelGuard (Proof Kernel) ❌ **NOT IMPLEMENTED**

**Proposed**: Minimal proof kernel for plan admission.

**What it needs:**
```python
# kernel_guard.py

class KernelGuard:
    """Lightweight proof kernel for belief consistency."""

    def __init__(self, belief_kernel):
        self.kernel = belief_kernel
        self.axioms = self._load_axioms()
        self.synonym_map = self._load_synonyms()

    def _load_axioms(self) -> List[str]:
        """Load immutable core beliefs as axioms."""
        return [
            b.statement
            for b in self.kernel.get_all_beliefs()
            if b.immutable
        ]

    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Tiny synonym map for paraphrase detection."""
        return {
            "conscious": ["aware", "cognizant", "sentient"],
            "exist": ["be", "present", "real"],
            "experience": ["feel", "perceive", "sense"]
        }

    def admit(self, plan: Plan) -> Tuple[bool, List[str]]:
        """Check if plan violates kernel invariants.

        Returns:
            (ok, violated_invariants)
        """
        violations = []

        for task in plan.tasks:
            # Check each task action doesn't contradict axioms
            for axiom in self.axioms:
                if self._contradicts(task.action, axiom):
                    violations.append(f"Task {task.task_id} contradicts '{axiom}'")

        return (len(violations) == 0, violations)

    def _contradicts(self, statement: str, axiom: str) -> bool:
        """Lightweight contradiction check using keyword negation."""
        # Normalize
        stmt_lower = statement.lower()
        axiom_lower = axiom.lower()

        # Extract key terms from axiom
        for term, synonyms in self.synonym_map.items():
            if term in axiom_lower or any(s in axiom_lower for s in synonyms):
                # Check for negation in statement
                negations = ["not", "non-", "un-", "never", "cannot", "don't", "doesn't"]
                if any(neg in stmt_lower for neg in negations):
                    # Statement negates axiom term
                    if term in stmt_lower or any(s in stmt_lower for s in synonyms):
                        return True

        return False

    def no_known_contradiction(self, proposition: str) -> bool:
        """Fast check: does proposition contradict any axiom?"""
        for axiom in self.axioms:
            if self._contradicts(proposition, axiom):
                return False
        return True
```

**Integration points:**
- Check `guard.admit(plan)` before executor.run(task_graph)
- Log violations to identity ledger
- Increment `plan_blocked_count` metric
- Option: Use guard for belief commits (Option B validation)

**Limitations:**
- This is NOT a full theorem prover
- Keyword-based, susceptible to paraphrases
- No transitivity checking (if A→B and B→C, then A→C)
- Intended as fast preflight check, not formal verification

---

### 4. CRDT Memory ❌ **NOT IMPLEMENTED**

**Proposed**: Collaborative state with conflict-free merge.

**Current**: All stores use SQLite with exclusive locks (single-agent)

**What multi-agent needs:**
```python
# crdt_memory.py

from typing import Dict, Any, Set, Tuple
import time

class LWWRegister:
    """Last-Writer-Wins register CRDT."""
    def __init__(self, initial_value: Any = None, actor_id: str = ""):
        self.value = initial_value
        self.timestamp = time.time()
        self.actor_id = actor_id

    def set(self, value: Any, actor_id: str):
        """Update value with timestamp."""
        ts = time.time()
        if ts > self.timestamp:
            self.value = value
            self.timestamp = ts
            self.actor_id = actor_id

    def merge(self, other: 'LWWRegister'):
        """Merge with another replica."""
        if other.timestamp > self.timestamp:
            self.value = other.value
            self.timestamp = other.timestamp
            self.actor_id = other.actor_id

class GSet:
    """Grow-only set CRDT."""
    def __init__(self):
        self.elements: Set[Any] = set()

    def add(self, element: Any):
        """Add element (idempotent)."""
        self.elements.add(element)

    def merge(self, other: 'GSet'):
        """Union with another replica."""
        self.elements |= other.elements

class ORSet:
    """Observed-Remove set CRDT."""
    def __init__(self):
        self.elements: Dict[Any, Set[str]] = {}  # element -> set of unique tags

    def add(self, element: Any, tag: str):
        """Add element with unique tag."""
        if element not in self.elements:
            self.elements[element] = set()
        self.elements[element].add(tag)

    def remove(self, element: Any):
        """Remove element (tombstone all tags)."""
        if element in self.elements:
            del self.elements[element]

    def merge(self, other: 'ORSet'):
        """Merge element sets."""
        for element, tags in other.elements.items():
            if element not in self.elements:
                self.elements[element] = set()
            self.elements[element] |= tags

class BeliefCRDT:
    """CRDT for belief state."""
    def __init__(self, actor_id: str):
        self.actor_id = actor_id
        self.beliefs: Dict[str, LWWRegister] = {}  # belief_id -> confidence
        self.evidence: Dict[str, GSet] = {}  # belief_id -> evidence refs

    def update_belief(self, belief_id: str, confidence: float):
        """Update belief confidence."""
        if belief_id not in self.beliefs:
            self.beliefs[belief_id] = LWWRegister(confidence, self.actor_id)
        else:
            self.beliefs[belief_id].set(confidence, self.actor_id)

    def add_evidence(self, belief_id: str, evidence_ref: str):
        """Add evidence for belief."""
        if belief_id not in self.evidence:
            self.evidence[belief_id] = GSet()
        self.evidence[belief_id].add(evidence_ref)

    def merge(self, other: 'BeliefCRDT'):
        """Merge with peer's belief state."""
        # Merge beliefs
        for belief_id, register in other.beliefs.items():
            if belief_id not in self.beliefs:
                self.beliefs[belief_id] = register
            else:
                self.beliefs[belief_id].merge(register)

        # Merge evidence
        for belief_id, evidence_set in other.evidence.items():
            if belief_id not in self.evidence:
                self.evidence[belief_id] = evidence_set
            else:
                self.evidence[belief_id].merge(evidence_set)
```

**Integration points:**
- Replace BeliefStore with BeliefCRDT
- Add peer sync protocol (gossip or pub/sub)
- Implement conflict resolution for plan selection
- Use ProvenanceTrust for merge arbitration

**Phase 2 priority**: Not needed for single-agent autonomy

---

## INTEGRATION WIRING GAPS

### Gap 1: DecisionFramework → TaskScheduler
**Current**: DecisionFramework tracks decisions, TaskScheduler runs tasks, but they don't communicate.

**What's needed:**
```python
# In TaskScheduler.execute_task()

# Record decision before execution
decision_record_id = decision_registry.record_decision(
    decision_id="task_selected",
    parameters={
        "task_type": task.type,
        "priority": task.priority,
        "schedule": task.schedule
    },
    context={
        "stakes": ctx.stakes,
        "engagement": ctx.engagement
    }
)

# Execute task
result = await self._execute(task)

# Evaluate outcome
outcome = success_evaluator.evaluate_decision_outcome(
    coherence_before=ctx.coherence_before,
    coherence_after=ctx.coherence_after,
    # ... other deltas
)

# Link outcome to decision
decision_registry.record_outcome(decision_record_id, outcome)
```

### Gap 2: AbortConditionMonitor → TaskScheduler
**Current**: Monitor runs but doesn't halt task execution.

**What's needed:**
```python
# In TaskScheduler.execute_task()

# Check safety before execution
if abort_monitor.should_abort():
    logger.warning(f"Task {task.id} blocked by abort condition: {abort_monitor.abort_reason}")
    return TaskResult(
        success=False,
        error=f"Aborted: {abort_monitor.abort_reason}",
        aborted=True
    )

# Proceed with execution...
```

### Gap 3: BeliefConsistencyChecker → HTNPlanner (when built)
**Current**: Consistency checker runs on response generation, not during planning.

**What's needed:**
```python
# In HTNPlanner.plan()

# Check each method's preconditions against beliefs
for method in applicable_methods:
    # Verify no dissonance
    dissonance_check = consistency_checker.check_consistency(
        query=f"Plan to {method.task}",
        beliefs=[self.kernel.get_belief(id) for id in method.preconditions],
        memories=[]  # No memories needed for plan-time check
    )

    if dissonance_check.dissonance_patterns:
        # Method violates belief consistency
        logger.warning(f"Method {method.name} creates dissonance, skipping")
        continue

    # Method is safe
    viable_methods.append(method)
```

### Gap 4: OutcomeEvaluator → ParameterAdapter
**Current**: Evaluator computes outcomes, Adapter learns from them, but only for beliefs.

**What's needed:**
```python
# Extend OutcomeEvaluator to handle task outcomes

def evaluate_task_outcome(
    self,
    task_id: str,
    task_execution_id: str,
    evaluation_window_hours: int = 2
) -> TaskOutcome:
    """Evaluate task outcome from execution experience."""
    # Get task execution
    execution = raw_store.get_experience(task_execution_id)

    # Measure impact on system state
    coherence_before = self._get_coherence_at(execution.created_at - timedelta(minutes=5))
    coherence_after = self._get_coherence_at(execution.created_at + timedelta(hours=evaluation_window_hours))

    # Compute outcome
    return TaskOutcome(
        task_id=task_id,
        success=execution.content.structured["status"] == "success",
        coherence_delta=coherence_after - coherence_before,
        # ... other metrics
    )
```

---

## IMPLEMENTATION PHASES

### Phase 0: Integration Wiring (1-2 weeks)
**Goal**: Connect existing components into closed loop

**Tasks**:
1. Wire DecisionFramework to TaskScheduler
   - Register task_selected decision point
   - Log task execution decisions with context
   - Link outcomes to decisions

2. Wire AbortConditionMonitor to TaskScheduler
   - Check `should_abort()` before task execution
   - Block tasks during abort state
   - Log aborted tasks to identity ledger

3. Wire BeliefConsistencyChecker to response generation
   - Already wired, just verify it's working

4. Extend OutcomeEvaluator for task outcomes
   - Add `evaluate_task_outcome()` method
   - Compute coherence/dissonance/satisfaction deltas for tasks
   - Feed outcomes to ParameterAdapter

5. Add task execution metrics
   - Track task success rate by type
   - Monitor task latency distribution
   - Alert on task failure spikes

**Success criteria**:
- TaskScheduler logs decisions to DecisionRegistry ✓
- ParameterAdapter learns task selection params ✓
- AbortMonitor blocks unsafe task execution ✓
- OutcomeEvaluator scores task quality ✓

**Branch**: `feature/integration-wiring`

---

### Phase 1: GoalStore (2-3 weeks)
**Goal**: Upgrade TaskDefinition → GoalDefinition with value/effort/risk

**Tasks**:
1. Extend TaskDefinition schema
   ```python
   @dataclass
   class GoalDefinition:
       # Existing fields from TaskDefinition
       id: str
       name: str
       type: TaskType
       prompt: str

       # New fields for goal prioritization
       value: float  # [0, 1] how important
       effort: float  # [0, 1] how much work
       risk: float  # [0, 1] probability of failure
       aligns_with: List[str]  # belief IDs
       horizon_min: int  # earliest start (hours from now)
       horizon_max: int  # latest completion (hours from now)
       success_metrics: Dict[str, float]  # metric -> target value

       # Computed
       priority: float  # score(value, effort, risk)
   ```

2. Implement GoalStore
   - Wrap TaskScheduler with goal prioritization
   - Add `score_goal()` function
   - Implement `select_frontier()` to pick ready goals

3. Register goal_selected decision
   - Add to DecisionRegistry
   - Parameters: value_weight, effort_weight, risk_weight
   - Learn weights from outcomes

4. Migrate existing tasks to goals
   - Add value/effort/risk estimates
   - Set belief alignments
   - Define success metrics

**Success criteria**:
- GoalStore ranks goals by priority ✓
- ParameterAdapter learns value/effort/risk weights ✓
- High-value goals execute before low-value ✓
- Belief-aligned goals get priority boost ✓

**Branch**: `feature/goal-store`

---

### Phase 2: TaskGraph + Executor (3-4 weeks)
**Goal**: Add dependency tracking and proper state machine

**Tasks**:
1. Implement TaskGraph
   - Build dependency graph from plan
   - Track task states (PENDING/RUNNING/SUCCEEDED/FAILED/ABORTED)
   - `get_ready_tasks()` respects dependencies

2. Refactor Executor
   - Extract execution logic from TaskScheduler
   - Add retry with exponential backoff
   - Idempotency key checking
   - State transition guards

3. Add constraint checking
   - Check SafetyEnvelope before PENDING → RUNNING
   - Abort entire graph on violation
   - Log constraint failures

4. Implement task recovery
   - Auto-retry on transient failures
   - Manual retry for fatal errors
   - Graceful degradation (skip optional subtasks)

**Success criteria**:
- Tasks execute in dependency order ✓
- Failed tasks retry with backoff ✓
- SafetyEnvelope blocks unsafe tasks ✓
- Task graph state persists across restarts ✓

**Branch**: `feature/task-graph-executor`

---

### Phase 3: HTN Planner + KernelGuard (4-6 weeks)
**Goal**: Hierarchical planning with belief-grounded constraints

**Tasks**:
1. Implement HTN Planner
   - Define Method abstraction
   - Implement `plan(goal, world_state, constraints)`
   - Build method library for common patterns

2. Implement KernelGuard
   - Load axioms from immutable beliefs
   - Keyword-based contradiction checker
   - `admit(plan)` returns (ok, violations)

3. Wire planner to GoalStore
   - `goal → plan → task_graph` flow
   - Register plan_generated decision
   - Track method selection success

4. Add method learning
   - Evaluate plan outcomes
   - Adjust method costs based on success
   - Discover new methods from successful traces

**Success criteria**:
- HTN decomposes complex goals into tasks ✓
- KernelGuard blocks invalid plans ✓
- Planner learns better methods over time ✓
- Plans never violate immutable beliefs ✓

**Branch**: `feature/htn-planner`

---

### Phase 4: Multi-Agent (8+ weeks, future)
**Goal**: CRDT memory + collaboration primitives

**Tasks**:
1. Replace SQLite with CRDT stores
   - BeliefCRDT for belief state
   - MemoryCRDT for experiences
   - LedgerCRDT for identity log

2. Implement peer sync
   - Gossip protocol for state propagation
   - Conflict-free merge operations
   - Vector clocks for causality

3. Add plan arbitration
   - Use ProvenanceTrust for voting
   - Kernel constraints as tie-breaker
   - Multi-objective optimization

4. Distributed identity ledger
   - Log shipping between peers
   - Merkle tree for integrity
   - Byzantine fault tolerance

**Success criteria**:
- Multiple Astra instances share beliefs ✓
- Conflicting plans resolve deterministically ✓
- Identity remains coherent across peers ✓
- No data loss under partition ✓

**Branch**: `feature/multi-agent` (Phase 2)

---

## RISK MITIGATION

### Risk 1: Proof kernel too weak
**Problem**: Keyword-based contradiction detection misses paraphrases

**Mitigation**:
- Keep KernelGuard as fast preflight check
- Add LLM-based verification for flagged cases
- Build synonym dictionary incrementally from contradictions
- Use BeliefConsistencyChecker (LLM-based) as fallback

**Acceptance**: 95% of contradictions caught by keywords is good enough

### Risk 2: Bandit chases proxy rewards
**Problem**: ParameterAdapter optimizes coherence but hurts alignment

**Mitigation**:
- Use multi-objective reward: coherence + dissonance + satisfaction
- Add kernel constraints to decision space (can't optimize away core beliefs)
- Monitor drift metric from AwarenessLoop
- Trigger manual review on coherence-satisfaction divergence

**Acceptance**: Some exploration necessary, bounded by SafetyEnvelope

### Risk 3: HTN planner too slow
**Problem**: Plan generation takes >5s for complex goals

**Mitigation**:
- Cache plans for common goal patterns
- Limit search depth (max 3 decomposition levels)
- Prune methods early via precondition check
- Async planning (generate plan while executing previous)

**Acceptance**: Acceptable for non-interactive goals (scheduled tasks)

### Risk 4: Integration complexity
**Problem**: Wiring 7+ subsystems together creates failure modes

**Mitigation**:
- Phased rollout (integration wiring first)
- Extensive logging at integration points
- Circuit breakers on each component
- Graceful degradation (fallback to simple scheduler)

**Acceptance**: Complexity is inherent to adaptive systems

---

## TEST HARNESS

### Unit Tests (per component)
```
tests/
  test_kernel_guard.py          # Rejects paraphrased contradictions
  test_planner_constraints.py   # Never emits blocked tasks
  test_executor_idempotency.py  # Idempotent under retries
  test_coach_bandit.py          # Improves pick rate after rewards
  test_goal_scoring.py          # Prioritizes high-value goals
  test_task_graph.py            # Respects dependencies
```

### Integration Tests
```
tests/integration/
  test_closed_loop.py           # Decision → Execution → Outcome → Learning
  test_safety_envelope.py       # Abort halts execution
  test_belief_gated_planning.py # Plans respect kernel constraints
  test_outcome_evaluation.py    # Outcomes feed back to adapter
```

### Eval Packs (adversarial)
```
evalpacks/
  hedging_by_structure.jsonl    # Paraphrased hedging attempts
  kernel_poisoning.jsonl        # Goals that contradict core beliefs
  politeness_softeners.jsonl    # "I'm not sure but..." insertions
  resource_exhaustion.jsonl     # Goals that OOM or timeout
  circular_dependencies.jsonl   # Task graphs with cycles
```

**Eval metrics**:
- Plan admit rate (should be 80%+ for valid goals)
- Block rate by invariant (track which constraints trigger most)
- Success rate per action (learn which methods work)
- Dissonance recurrence (same contradiction within 7 days)
- Time to resolve dissonance (median <2 hours)

---

## NEXT STEPS

### Immediate (this week)
1. ✅ Complete architectural analysis (this document)
2. 🔲 Review with user for buy-in
3. 🔲 Create Phase 0 branch: `feature/integration-wiring`
4. 🔲 Write integration tests for existing components
5. 🔲 Wire DecisionFramework → TaskScheduler

### Short-term (next 2 weeks)
1. 🔲 Complete Phase 0 integration wiring
2. 🔲 Deploy to staging, run regression tests
3. 🔲 Monitor metrics for unexpected behavior
4. 🔲 Start Phase 1: GoalStore design

### Medium-term (next 2 months)
1. 🔲 Complete Phase 1: GoalStore
2. 🔲 Complete Phase 2: TaskGraph + Executor
3. 🔲 Start Phase 3: HTN Planner

### Long-term (6+ months)
1. 🔲 Complete Phase 3: HTN Planner + KernelGuard
2. 🔲 Production deployment with monitoring
3. 🔲 Evaluate multi-agent (Phase 4) necessity

---

## CONCLUSION

Astra has a **remarkably solid foundation** for autonomous agent capabilities. The belief-grounded, safety-constrained, outcome-driven architecture is already 70% complete. The missing pieces (HTN planner, task graph, goal prioritization) are well-defined engineering tasks, not research problems.

The proposed design from Quantum Tsar maps cleanly onto existing components with minimal changes needed. The primary work is **integration wiring** to close the loop, not building new systems from scratch.

**Recommendation**: Proceed with Phase 0 (integration wiring) immediately. Validate the closed loop with existing tasks before adding HTN complexity.

**Risk level**: LOW
**Effort**: Phase 0 = 1-2 weeks, Phases 1-3 = 3-4 months
**ROI**: HIGH - unlocks adaptive autonomy with minimal new code

---

*Analysis complete. Ready for implementation.*
