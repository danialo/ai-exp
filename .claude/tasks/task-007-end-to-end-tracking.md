# Task #7: End-to-End Task Tracking for Auditability

## Objective

Implement comprehensive end-to-end tracking for task execution to enable full auditability of:
- What task was executed and why
- What context (memories/beliefs) informed the task
- What output the task produced
- What side effects occurred (belief updates, file changes, etc.)
- How task execution contributes to goals

## Current State Analysis

### What Exists ✅
1. **Task Scheduler System** (`src/services/task_scheduler.py`)
   - Executes scheduled tasks (reflection, goal assessment, etc.)
   - Stores results in JSON files (`persona_space/tasks/results/`)
   - Tracks: task_id, start/end times, success/failure, response text

2. **Experience Store with Parent/Child Links** (`src/memory/raw_store.py`)
   - DAG structure for experience relationships
   - Parent field on every experience
   - Supports multi-parent derivations

3. **Identity Ledger** (`src/services/identity_ledger.py`)
   - SHA-256 chained append-only log
   - Tracks belief changes, anchor updates, dissonance events
   - Daily rotated NDJSON.gz files

4. **Session Tracking** (`src/services/session_tracker.py`)
   - Links experiences to conversation sessions
   - Tracks session lifecycle

### Critical Gaps ❌

**Gap 1: Task Executions Not Stored as Experiences**
- Task results live in separate JSON files
- No parent/child links to retrieved context
- Cannot query task history as memories
- Missing from experience graph

**Gap 2: No Task Lifecycle Audit Trail**
- Can't trace who created a task
- No record of task configuration changes
- No provenance for task creation/modification

**Gap 3: No Task-to-Goal Linking**
- Tasks exist but no formal goal model
- Can't track progress toward goals
- No success criteria or outcome evaluation

**Gap 4: Cross-System Traceability Incomplete**
- No unified correlation ID across systems
- Hard to follow: user request → task execution → belief update → outcome

**Gap 5: No Task Quality Metrics**
- Can't evaluate task effectiveness
- No feedback loop for task optimization

## Implementation Plan

### Phase 1: Link Task Executions to Experiences (HIGH PRIORITY)

**Goal**: Every task execution creates a TASK_EXECUTION experience with full correlation, idempotency, and failure semantics.

**Key Changes from Initial Design**:
- Use `TASK_EXECUTION` type (not OBSERVATION) to avoid query ambiguity
- Add correlation fields (trace_id, span_id) for distributed tracing
- Separate `parents` (immediate inputs) vs `causes` (causal triggers)
- Full execution status model with retry tracking
- Idempotent inserts on idempotency_key
- Always create experience on failure
- SQLite indexes for queryability
- PII scrubbing before persistence

**Implementation**:

1. **Add New Experience Type** (`src/memory/models.py`)
   ```python
   class ExperienceType(str, Enum):
       # ... existing types ...
       TASK_EXECUTION = "task_execution"  # NEW

   class CaptureMethod(str, Enum):
       # ... existing methods ...
       SCHEDULED_TASK = "scheduled_task"  # NEW
       MANUAL_TASK = "manual_task"  # NEW
   ```

2. **Add `causes` Field to Experience Model** (`src/memory/models.py`)
   ```python
   class ExperienceModel(BaseModel):
       # ... existing fields ...
       parents: List[str] = []  # Immediate inputs (retrieved memories)
       causes: List[str] = []   # NEW: Causal triggers (prior tasks, goals)
   ```

3. **Modify Task Scheduler** (`src/services/task_scheduler.py`)
   - Wrap execution with precise timing
   - Generate trace_id, span_id, idempotency_key
   - Capture all retrieval metadata
   - Track files written, tool calls, script runs
   - Create experience on both success AND failure
   - Run PII scrubber on content

4. **Add Idempotent Insert** (`src/memory/raw_store.py`)
   ```python
   def append_experience_idempotent(self, experience: ExperienceModel, idempotency_key: str) -> str:
       """Insert experience only if idempotency_key not seen before."""
       # Check if key exists
       # If exists, return existing experience_id
       # If not, insert and return new experience_id
   ```

5. **Add Query Helpers** (`src/memory/raw_store.py`)
   ```python
   def list_task_executions(self, task_id: Optional[str] = None, limit: int = 20):
       """List task executions, optionally filtered by task_id."""

   def get_by_trace_id(self, trace_id: str) -> Optional[ExperienceModel]:
       """Get experience by trace_id."""
   ```

6. **Add SQLite Indexes** (database migration or startup script)
   ```sql
   CREATE INDEX IF NOT EXISTS ix_experiences_type_ts
       ON experiences(type, created_at DESC);

   CREATE INDEX IF NOT EXISTS ix_experiences_task
       ON experiences(json_extract(content, '$.structured.task_id'), created_at DESC);

   CREATE INDEX IF NOT EXISTS ix_experiences_task_slug
       ON experiences(json_extract(content, '$.structured.task_slug'), ts DESC);

   CREATE INDEX IF NOT EXISTS ix_experiences_trace
       ON experiences(json_extract(content, '$.structured.trace_id'));

   CREATE INDEX IF NOT EXISTS ix_experiences_idempotency
       ON experiences(json_extract(content, '$.structured.idempotency_key'));
   ```

7. **Full Experience Structure** (Concrete Field Contract):
```python
{
  "id": "task_exec_daily_reflection_2025-11-04T18:29:12Z_9f1c",
  "type": "TASK_EXECUTION",  # NEW TYPE
  "content": {
    "text": "short result summary or full response",
    "structured": {
      # Schema versioning
      "schema_version": 1,  # EXPERIENCE_SCHEMA_VERSION for TASK_EXECUTION

      # Core task identity
      "task_id": "daily_reflection",  # UUID or stable ID
      "task_slug": "daily_reflection",  # Human-readable slug for queries
      "task_name": "Daily Self-Reflection",
      "task_type": "reflection",  # reflection|assessment|ingest|custom
      "scheduled_vs_manual": "scheduled",

      # Execution status
      "status": "success",  # success|failed|partial
      "started_at_iso": "2025-11-04T18:29:12Z",  # UTC ISO-8601 string
      "ended_at_iso": "2025-11-04T18:29:13Z",
      "started_at_ts": 1762284552.123,  # Float seconds for SQL
      "ended_at_ts": 1762284553.165,
      "duration_ms": 1042,  # max(ended - started, 0)

      # Correlation and idempotency
      "trace_id": "550e8400-e29b-41d4-a716-446655440000",  # UUIDv4, stable across retries
      "span_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",  # UUIDv4, unique per attempt
      "attempt": 1,  # 1-n for retries
      "retry_of": null,  # span_id of prior attempt if retry
      "idempotency_key": "sha256(task_id+scheduled_at+attempt)",

      # Task configuration
      "task_config_digest": "sha256:abc123...",  # Proof of config used

      # Retrieval provenance
      "retrieval": {
        "memory_count": 3,
        "query": "recent tensions and reflections",
        "filters": {"type": "OBSERVATION"},
        "latency_ms": 28,
        "source": ["experiences", "beliefs"]
      },

      # Resource and side-effects ledger
      "io": {
        "files_written": ["data/reflections/2025-11-04.json"],
        "artifact_ids": ["art_abc123"],
        "tool_calls": ["toolrun_9f2"],  # Reference to tool_call table
        "script_runs": ["scr_733"]  # Reference to script_run table
      },

      # Embeddings
      "embedding_skipped": true,  # Phase 1: true, future: false for backfill

      # Scheduler identity
      "provenance_runner": {
        "name": "scheduler",
        "version": "1.0.0",
        "pid": 351425,
        "host": "d-vps"
      },

      # Permissions (even if stub)
      "grants": {
        "scopes": [],
        "impersonation": false
      },

      # Error details (only on failure)
      "error": null  # or {"type": "ValueError", "message": "...", "stack_hash": "abc123"}
    }
  },
  "provenance": {
    "actor": "AGENT",
    "method": "SCHEDULED_TASK",
    "sources": []
  },
  "parents": ["exp_123", "exp_456", "exp_789"],  # Retrieved memories (immediate inputs)
  "causes": ["goal_improve_self_awareness"],  # Optional: goals or prior tasks that triggered this
  "session_id": "task:daily_reflection:2025-11-04T18:29:12Z:9f1c",  # Structured format
  "created_at": "2025-11-04T18:29:12Z",
  "ts": 1730744952.123,  # Float seconds for SQL
  "meta": {"scrubbed": true}  # PII scrubbing flag
}
```

**Files to Create**:
- `src/pipeline/task_experience.py` - Helper to create task execution experiences
- `scripts/backfill_task_executions.py` - Backfill existing task results

**Files to Modify**:
- `src/services/task_scheduler.py` - Wrap execution, create experiences
- `src/memory/models.py` - Add TASK_EXECUTION type, add causes field
- `src/memory/raw_store.py` - Add idempotent insert, query helpers, indexes
- `src/pipeline/ingest.py` - Share PII scrubber logic

**Testing**:

**Invariants**:
1. Creating TASK_EXECUTION with same idempotency_key is no-op
2. If status="success", duration_ms >= 0 and ended_at_ts >= started_at_ts
3. If status="failed", error.type and error.message are non-null
4. If retrieval.memory_count > 0, len(parents) == memory_count, else both zero
5. trace_id constant across retries, span_id unique per attempt
6. attempt >= 1, and attempt == 1 when retry_of is null
7. retry_of holds span_id (not trace_id) of prior attempt
8. idempotency_key uniqueness enforced by insert path
9. Always create experience on failure with error details
10. tool_calls and script_runs reference existing tables or are empty

**Test Cases**:
1. Execute scheduled task → verify success experience created
2. Force task failure → verify failed experience with error details
3. Re-run with same schedule window → verify no duplicate (idempotency)
4. Verify parents length matches retrieval.memory_count
5. Verify trace_id, span_id, attempt populated
6. Query by task_id → returns all runs of that task
7. Query by trace_id → returns specific execution
8. Verify indexes exist and perform well

**Backfill Plan**:
- Script scans `persona_space/tasks/results/*.json`
- Creates TASK_EXECUTION experiences with meta.backfilled=true
- Maintains mapping table: legacy_result_file → experience_id
- Honors idempotency_key (no duplicates on re-run)
- Dry-run mode with counts before actual insert
- Idempotent execution (safe to re-run)

---

### Phase 2: Task Lifecycle Ledger (HIGH PRIORITY)

**Goal**: Track task creation, modification, enable/disable events in audit log.

**Implementation Options**:

**Option A: Extend Identity Ledger**
- Add new event types: `task_created`, `task_modified`, `task_executed`, `task_disabled`
- Reuse existing SHA-256 chain and NDJSON.gz format

**Option B: Create Separate Task Ledger**
- New file: `src/services/task_ledger.py`
- Similar structure to identity ledger
- Stores in `data/tasks/ledger-YYYYMMDD.ndjson.gz`

**Recommendation**: Use Option A (extend identity ledger) for unified audit trail.

**Event Structure**:
```python
{
  "ts": 1730721600.0,
  "schema": 2,
  "event": "task_executed",
  "task_id": "daily_reflection",
  "experience_ref": "task_exec_daily_reflection_2025-11-04T12:00:00Z",
  "meta": {
    "execution_duration_ms": 3200,
    "success": true,
    "error": null,
    "retrieved_memory_count": 5
  },
  "prev": "<sha256 of previous entry>"
}
```

**Files to Modify**:
- `src/services/identity_ledger.py` - Add task event types
- `src/services/task_scheduler.py` - Log events to ledger

**Testing**:
- Create/modify/execute/disable tasks
- Verify ledger entries created
- Verify SHA-256 chain integrity
- Query ledger for task history

---

### Phase 3: Unified Trace Context (HIGH PRIORITY)

**Goal**: Correlation IDs that flow through session → experiences → task executions → belief updates.

**Implementation**:

1. **Create TraceContext Model** (`src/services/trace_context.py`):
```python
from dataclasses import dataclass
from typing import Optional
import uuid

@dataclass
class TraceContext:
    """Unified trace context for cross-system tracking."""
    correlation_id: str  # UUID for entire trace
    session_id: str
    task_id: Optional[str] = None
    parent_experience_ids: list[str] = None

    def __post_init__(self):
        if self.parent_experience_ids is None:
            self.parent_experience_ids = []

    @classmethod
    def create(cls, session_id: str, task_id: Optional[str] = None):
        """Create new trace context with generated correlation ID."""
        return cls(
            correlation_id=str(uuid.uuid4()),
            session_id=session_id,
            task_id=task_id,
        )
```

2. **Thread-Local Context Storage**:
```python
import threading

_trace_context = threading.local()

def set_trace_context(ctx: TraceContext):
    _trace_context.value = ctx

def get_trace_context() -> Optional[TraceContext]:
    return getattr(_trace_context, 'value', None)
```

3. **Modify Systems to Use Trace Context**:
- Task scheduler: Set trace context before execution
- Ingestion pipeline: Include correlation_id in experience metadata
- Belief store: Include correlation_id in delta log
- Identity ledger: Include correlation_id in events

**Files to Create**:
- `src/services/trace_context.py`

**Files to Modify**:
- `src/services/task_scheduler.py`
- `src/pipeline/ingest.py`
- `src/services/belief_store.py`
- `src/services/identity_ledger.py`

**Testing**:
- Execute task with trace context
- Verify correlation_id flows through all systems
- Query experiences by correlation_id
- Reconstruct full execution trace from correlation_id

---

### Phase 4: Goal Model and Task-Goal Linking (MEDIUM PRIORITY)

**Goal**: Formal goal tracking with success criteria and task assignments.

**Implementation**:

1. **Create Goal Model** (`src/models/goal.py`):
```python
from datetime import datetime
from typing import Optional, List, Dict
from sqlmodel import SQLModel, Field, JSON, Column
from sqlalchemy import JSON as SAJSON

class Goal(SQLModel, table=True):
    """Represents a goal that Astra is working toward."""
    id: str = Field(primary_key=True)  # "goal_improve_self_awareness"
    statement: str  # "Improve self-awareness through daily reflection"
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    priority: int = Field(default=5)  # 1-10 scale
    active: bool = Field(default=True)

    # Success criteria (measurable)
    success_criteria: Dict = Field(default={}, sa_column=Column(SAJSON))
    # Example: {"sim_self_increase": 0.2, "timeframe_days": 14}

    # Progress tracking
    progress_metrics: Dict = Field(default={}, sa_column=Column(SAJSON))
    # Example: {"current_sim_self": 0.15, "baseline_sim_self": 0.05}

    # Task assignments
    linked_tasks: List[str] = Field(default=[], sa_column=Column(SAJSON))
    # Example: ["daily_reflection", "goal_assessment"]

    # Completion
    completed: bool = Field(default=False)
    completed_at: Optional[datetime] = None
```

2. **Create Goal Store** (`src/services/goal_store.py`):
```python
class GoalStore:
    """Manages goals and tracks progress."""

    def create_goal(self, goal: Goal) -> str:
        """Create new goal and log to ledger."""
        pass

    def update_progress(self, goal_id: str, metrics: dict):
        """Update progress metrics for goal."""
        pass

    def link_task(self, goal_id: str, task_id: str):
        """Link task to goal."""
        pass

    def evaluate_success(self, goal_id: str) -> bool:
        """Check if success criteria met."""
        pass

    def get_active_goals(self) -> List[Goal]:
        """Get all active goals."""
        pass
```

3. **Modify Task Scheduler**:
- Add goal_id field to tasks
- After task execution, update goal progress
- Log goal progress to ledger

**Files to Create**:
- `src/models/goal.py`
- `src/services/goal_store.py`

**Files to Modify**:
- `src/services/task_scheduler.py`
- Database schema (add `goal` table)

**Testing**:
- Create goal with success criteria
- Link tasks to goal
- Execute tasks and verify progress updates
- Evaluate goal completion

---

### Phase 5: Post-Task Experience Hooks (MEDIUM PRIORITY)

**Goal**: Automatically integrate task outputs into experience graph.

**Implementation**:

1. **Create Post-Task Hook System**:
```python
class TaskExecutionHook:
    """Hooks that run after task execution."""

    async def on_task_complete(
        self,
        task_id: str,
        response: str,
        retrieved_experiences: List[str],
        trace_context: TraceContext,
    ):
        """Process task output and create derived experiences."""
        # 1. Create OBSERVATION experience
        exp_id = await self._create_task_experience(...)

        # 2. Extract self-claims
        await self._extract_self_claims(response, exp_id)

        # 3. Detect belief updates
        await self._detect_belief_updates(response, exp_id)

        # 4. Update relevant indices
        await self._update_indices(exp_id)
```

2. **Integrate with Task Scheduler**:
- After task execution, call hooks
- Pass trace context through hooks

**Files to Create**:
- `src/services/task_hooks.py`

**Files to Modify**:
- `src/services/task_scheduler.py`

**Testing**:
- Execute task that generates self-claims
- Verify self-claim extracted and indexed
- Verify belief updates detected
- Verify experience created with proper parents

---

### Phase 6: Task Quality Analytics (LOW PRIORITY)

**Goal**: Measure and optimize task effectiveness.

**Implementation**:

1. **Define Quality Metrics**:
```python
class TaskQualityMetrics:
    """Metrics for evaluating task effectiveness."""

    # Engagement metrics
    user_feedback_count: int  # Explicit feedback on task output
    positive_feedback_ratio: float

    # Impact metrics
    beliefs_influenced: int  # How many beliefs linked to this task
    goal_progress_contributed: float  # Impact on goal metrics

    # Quality metrics
    coherence_score: float  # How coherent is task output
    novelty_score: float  # How novel/insightful

    # Efficiency metrics
    execution_duration_ms: int
    token_usage: int
    cost: float
```

2. **Create Analytics Pipeline**:
- Aggregate metrics across task executions
- Identify high/low performing tasks
- Generate recommendations

**Files to Create**:
- `src/services/task_analytics.py`

**Testing**:
- Run analytics on historical task executions
- Verify metrics calculated correctly
- Generate task quality report

---

## Acceptance Criteria

### Phase 1 Complete When:
- [ ] Task executions create OBSERVATION experiences
- [ ] Parent links connect to retrieved context
- [ ] Can query task history via raw_store
- [ ] Task metadata stored in structured content

### Phase 2 Complete When:
- [ ] Task lifecycle events logged to ledger
- [ ] SHA-256 chain integrity maintained
- [ ] Can reconstruct task history from ledger
- [ ] Task creation/modification/execution tracked

### Phase 3 Complete When:
- [ ] Correlation IDs flow through all systems
- [ ] Can trace: session → task → belief update
- [ ] Trace context accessible in all subsystems
- [ ] Can reconstruct full execution path from correlation_id

### Phase 4 Complete When:
- [ ] Goal model implemented and tested
- [ ] Tasks can be linked to goals
- [ ] Goal progress updates automatically
- [ ] Success criteria evaluation works

### Phase 5 Complete When:
- [ ] Task hooks automatically process output
- [ ] Self-claims extracted from task responses
- [ ] Belief updates detected and logged
- [ ] Indices updated post-task

### Phase 6 Complete When:
- [ ] Quality metrics defined and calculated
- [ ] Analytics pipeline generates reports
- [ ] Can identify high/low performing tasks
- [ ] Recommendations generated for optimization

## Implementation Order

**Day 1**: Phase 1 (Task → Experience linking)
**Day 2**: Phase 2 (Task lifecycle ledger)
**Day 3**: Phase 3 (Trace context)
**Day 4**: Phase 4 (Goal model)
**Day 5**: Phase 5 (Post-task hooks)
**Day 6**: Phase 6 (Analytics)

## Files to Create/Modify

### Create:
- `src/services/trace_context.py`
- `src/models/goal.py`
- `src/services/goal_store.py`
- `src/services/task_hooks.py`
- `src/services/task_analytics.py`

### Modify:
- `src/services/task_scheduler.py` (major changes)
- `src/pipeline/ingest.py` (add correlation_id)
- `src/services/belief_store.py` (add correlation_id)
- `src/services/identity_ledger.py` (add task events)
- `src/memory/models.py` (add SCHEDULED_TASK to CaptureMethod)
- Database schema (add goal table)

## Testing Strategy

1. **Unit Tests**:
   - Test experience creation from task execution
   - Test ledger event generation
   - Test trace context propagation
   - Test goal progress calculation

2. **Integration Tests**:
   - Execute end-to-end task with full tracing
   - Verify all audit trails created
   - Reconstruct execution path from correlation_id
   - Test goal-task-outcome chain

3. **Regression Tests**:
   - Verify existing task functionality unchanged
   - Verify ledger integrity maintained
   - Verify experience store not corrupted

## Success Metrics

After implementation, should be able to:
1. **Trace any task execution back to its trigger** (user request, schedule, goal)
2. **Reconstruct full context** used to inform task (memories, beliefs)
3. **Track all side effects** of task execution (beliefs updated, files written)
4. **Measure progress** toward goals via task contributions
5. **Audit task quality** and identify optimization opportunities
6. **Answer questions like**:
   - "What task led to this belief being formed?"
   - "What memories informed this reflection?"
   - "How did this task contribute to goal progress?"
   - "Which tasks are most effective for self-awareness?"

## Risks and Mitigations

**Risk 1: Performance Impact**
- Creating experiences for every task could slow execution
- **Mitigation**: Batch writes, async processing

**Risk 2: Storage Growth**
- More audit data = more disk usage
- **Mitigation**: Daily rotation, compression (already exists)

**Risk 3: Complexity**
- More moving parts = harder to maintain
- **Mitigation**: Thorough documentation, tests, gradual rollout

**Risk 4: Breaking Changes**
- Modifying task_scheduler could break existing tasks
- **Mitigation**: Backwards compatibility, feature flags, extensive testing

## Notes

- Leverage existing infrastructure (ledgers, experience store, session tracking)
- Follow existing patterns (SHA-256 chains, NDJSON.gz, parent/child links)
- Maintain backwards compatibility where possible
- Prioritize phases by impact (1-3 are critical, 4-6 are nice-to-have)
- Consider building Phase 1 as MVP, then iterating
