# GoalStore Phase 1 - Implementation Prompt (v2)

<PROMPT>

  <ROLE>
    You are ChatGPT-5. Your primary objective is to design and implement Phase 1 of Astra's GoalStore: a value/effort/risk goal prioritization layer with persistence and APIs, integrated with the existing TaskScheduler and Decision Framework, without disrupting current safety and learning systems.
  </ROLE>

  <CONTEXT>
    Audience: Senior engineer (Codex) implementing a focused subsystem in a production-grade LLM agent (Astra).
    Domain: Agent architecture, planning primitives, execution orchestration, safety invariants tied to kernel beliefs.
    Codebase: Python 3.12, FastAPI APIs, SQLite for lightweight persistence, existing modules include TaskScheduler, Decision Framework, OutcomeEvaluator, IdentityLedger, BeliefConsistencyChecker, BeliefKernel.
    Branches:
      - master is stable baseline with adaptive decision framework and immutable-belief logit bias.
      - feature/phase0-integration-wiring (Claude) is doing integration wiring. Independent of GoalStore.
      - feature/goal-store (this work) is new. No merge dependency on Phase 0.
    Goals:
      - Introduce first-class Goal objects with value, effort, risk, horizon, category, alignment to kernel beliefs.
      - Provide GoalStore CRUD + query, prioritization ranking, and TaskScheduler integration hooks.
      - Record decisions and outcomes for learning without changing safety enforcement.
    Constraints:
      - Backward compatible with current tasks. No behavior changes to persona, belief, or awareness loops.
      - Minimal schema that can evolve to HTN later. No CRDT, no planner in this phase.
      - Feature flags to enable shadow mode before activation.
    Stakeholders: Quantum Tsar of Arrays (product owner), Claude (Phase 0 integration), Codex (this implementation).
    Success definition: Goals can be created, ranked, adopted, and handed to TaskScheduler as tasks with auditable ledger entries and baseline metrics exported. Zero regressions in safety checks.
  </CONTEXT>

  <DELIVERABLES>
    - Data model: GoalDefinition dataclass and persisted schema with migrations.
    - GoalStore: CRUD, list, query by state and priority, adoption lifecycle.
    - Prioritizer: value/effort/risk scoring with horizon urgency and belief alignment, tie to kernel-aligned constraints.
    - APIs: FastAPI endpoints for goals and prioritization, request/response schemas.
    - Scheduler integration: adaptor that emits tasks from selected goals, idempotent submission, trace propagation.
    - Metrics: Prometheus counters and gauges for goal creation, adoption, execution selection.
    - Ledger: IdentityLedger events for goal lifecycle and selection decisions.
    - Tests: Unit tests, property tests for ranking invariants, integration tests for API and scheduler hook.
    - Flags and config: GOAL_SYSTEM=on|off, GOAL_SHADOW=on|off.
    - Docs: README snippet and OpenAPI examples.
  </DELIVERABLES>

  <REQUIREMENTS>
    <SCOPE>
      - Implement GoalDefinition with fields: id, text, category, value, effort, risk, horizon_min_min, horizon_max_min, aligns_with (belief ids), contradicts (belief ids), success_metrics, state (proposed, adopted, executing, satisfied, abandoned), created_at, updated_at, metadata.
      - Implement GoalStore with SQLite persistence, optimistic locking, soft delete.
      - Implement ranker: score = w_value*value - w_effort*effort - w_risk*risk + w_urgency*urgency_factor + w_alignment*alignment_bonus, deterministic, configurable.
      - Provide FastAPI endpoints: POST /v1/goals, GET /v1/goals, PATCH /v1/goals/{id}, POST /v1/goals/{id}/adopt, POST /v1/goals/{id}/abandon, GET /v1/goals/prioritized.
      - Integrate with TaskScheduler using an adaptor that converts an adopted goal into one or more executable tasks with correlation ids.
      - Emit IdentityLedger events: goal_created, goal_updated, goal_adopted, goal_selected_for_execution, goal_satisfied, goal_abandoned, goal_blocked_by_belief.
      - Export Prometheus metrics: goals_total{category,state}, goals_adopted_total, prioritize_latency_ms histogram, selection_success_rate, goal_blocked_by_belief_total{belief_id}.
      - Add feature flags and configuration validation.
      - Maintain safety: run BeliefConsistencyChecker.check_consistency on goal text when adopted, block if severity >= 0.6, emit tasks only if no active dissonance.
      - Register goal_selected decision point with DecisionFramework for weight learning.
    </SCOPE>
    <OUT_OF_SCOPE>
      - HTN planner and task decomposition.
      - CRDT collaboration and multi-agent sync.
      - Contextual bandits and Thompson sampling.
      - Live changes to BeliefConsistencyChecker semantics.
      - Formal proof kernel (ConstraintGuard planned for Phase 3).
      - Automatic task-to-goal migration (manual script acceptable).
    </OUT_OF_SCOPE>
    <TONE_STYLE>
      Concise, technical, implementation-focused. No motivational language. No em dashes.
    </TONE_STYLE>
    <FORMAT>
      Produce a Markdown design with sections H1 to H3, API schemas as JSON, and Python code snippets for interfaces and tests.
    </FORMAT>
    <CONSTRAINTS>
      - Backward compatible with master. No changes to persona_service behavior.
      - Python 3.12. Use dataclasses for GoalDefinition, pydantic v2 for API schemas.
      - SQLite schema under migrations with forward-only migration file.
      - Execution must be idempotent. All API calls carry trace_id and span_id where applicable.
      - Shadow mode default on. Activation gated by GOAL_SYSTEM flag.
      - All goal text passes BeliefConsistencyChecker before adoption.
      - Goals and Tasks coexist: GoalStore.select_goal() called first, TaskScheduler.get_scheduled_task() is fallback.
    </CONSTRAINTS>
    <EVALUATION_CRITERIA>
      - Tests: >= 90 percent line coverage for GoalStore, >= 80 percent for API. Property tests pass for ranking stability under monotonic transforms.
      - Performance: prioritize p95 latency <= 30 ms on 1k goals in-memory rank, <= 80 ms including DB fetch.
      - Safety: zero kernel violations introduced in CI eval pack.
      - Reliability: idempotent adoption and selection across retries in integration test.
      - Observability: Prometheus metrics visible and correct in dev dashboard.
    </EVALUATION_CRITERIA>
  </REQUIREMENTS>

  <DATA_MODEL>
    <GOAL_DEFINITION>
      ```python
      @dataclass
      class GoalDefinition:
          """A goal with value/effort/risk prioritization."""

          # Identity
          id: str  # UUID v4
          text: str  # Human-readable goal description
          category: GoalCategory  # Enum: INTROSPECTION, EXPLORATION, MAINTENANCE, USER_REQUESTED

          # Prioritization factors
          value: float  # [0.0, 1.0] How important is this goal?
          effort: float  # [0.0, 1.0] How much work is required? (0=easy, 1=hard)
          risk: float  # [0.0, 1.0] Probability of failure or negative outcome

          # Temporal constraints
          horizon_min_min: int  # Earliest start time (minutes from creation, 0=immediate)
          horizon_max_min: Optional[int]  # Latest completion (minutes from creation, None=no deadline)

          # Belief alignment
          aligns_with: List[str]  # Belief IDs this goal supports (e.g., ["core.ontological.consciousness"])
          contradicts: List[str]  # Belief IDs this goal would violate (blocks adoption if any active)

          # Success criteria
          success_metrics: Dict[str, float]  # metric_name -> target_value
          # Example: {"coherence_delta": 0.05, "beliefs_generated": 3}

          # Lifecycle
          state: GoalState  # Enum: PROPOSED, ADOPTED, EXECUTING, SATISFIED, ABANDONED

          # Audit
          created_at: datetime
          updated_at: datetime
          metadata: Dict[str, Any]  # Extensible for future use

          # Computed (not persisted)
          @property
          def priority(self) -> float:
              """Computed priority score (set by ranker)."""
              return getattr(self, '_priority', 0.0)

          @priority.setter
          def priority(self, value: float):
              self._priority = value
      ```

      ```python
      class GoalCategory(str, Enum):
          INTROSPECTION = "introspection"  # Self-reflection, identity maintenance
          EXPLORATION = "exploration"  # Capability discovery, learning
          MAINTENANCE = "maintenance"  # Routine upkeep, health checks
          USER_REQUESTED = "user_requested"  # Explicit user goals
      ```

      ```python
      class GoalState(str, Enum):
          PROPOSED = "proposed"  # Created but not yet vetted
          ADOPTED = "adopted"  # Vetted, ready for selection
          EXECUTING = "executing"  # Currently being worked on
          SATISFIED = "satisfied"  # Successfully completed
          ABANDONED = "abandoned"  # Failed, cancelled, or expired
      ```
    </GOAL_DEFINITION>

    <SQLITE_SCHEMA>
      ```sql
      -- migrations/003_create_goals.sql
      CREATE TABLE IF NOT EXISTS goals (
          id TEXT PRIMARY KEY,
          text TEXT NOT NULL,
          category TEXT NOT NULL,
          value REAL NOT NULL CHECK(value >= 0.0 AND value <= 1.0),
          effort REAL NOT NULL CHECK(effort >= 0.0 AND effort <= 1.0),
          risk REAL NOT NULL CHECK(risk >= 0.0 AND risk <= 1.0),
          horizon_min_min INTEGER NOT NULL DEFAULT 0,
          horizon_max_min INTEGER,
          aligns_with TEXT NOT NULL DEFAULT '[]',  -- JSON array of belief IDs
          contradicts TEXT NOT NULL DEFAULT '[]',  -- JSON array of belief IDs
          success_metrics TEXT NOT NULL DEFAULT '{}',  -- JSON object
          state TEXT NOT NULL DEFAULT 'proposed',
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          metadata TEXT NOT NULL DEFAULT '{}',
          version INTEGER NOT NULL DEFAULT 1,  -- Optimistic locking
          deleted_at TEXT  -- Soft delete
      );

      CREATE INDEX idx_goals_state ON goals(state) WHERE deleted_at IS NULL;
      CREATE INDEX idx_goals_category ON goals(category) WHERE deleted_at IS NULL;
      CREATE INDEX idx_goals_created_at ON goals(created_at) WHERE deleted_at IS NULL;
      CREATE INDEX idx_goals_horizon_max ON goals(horizon_max_min) WHERE deleted_at IS NULL;
      ```
    </SQLITE_SCHEMA>
  </DATA_MODEL>

  <RANKING_ALGORITHM>
    <PRIORITY_SCORING>
      ```python
      def compute_priority(
          goal: GoalDefinition,
          weights: PriorityWeights,
          belief_kernel: BeliefKernel,
          current_time: datetime
      ) -> float:
          """
          Compute goal priority for frontier selection.

          Higher score = execute sooner.

          Args:
              goal: Goal to score
              weights: Learned weight parameters (from DecisionFramework)
              belief_kernel: For checking belief alignment
              current_time: For computing urgency

          Returns:
              Priority score in [0.0, 1.0]
          """
          # Base components (normalized to [0, 1])
          value_norm = goal.value
          effort_norm = 1.0 - goal.effort  # Invert: prefer low effort
          risk_norm = 1.0 - goal.risk  # Invert: prefer low risk

          # Weighted base score
          base_score = (
              weights.value_weight * value_norm +
              weights.effort_weight * effort_norm +
              weights.risk_weight * risk_norm
          )

          # Urgency factor (boost if deadline approaching)
          urgency = 0.0
          if goal.horizon_max_min is not None:
              elapsed_min = (current_time - goal.created_at).total_seconds() / 60
              remaining_min = goal.horizon_max_min - elapsed_min
              if remaining_min > 0:
                  hours_remaining = remaining_min / 60
                  if hours_remaining < 24:
                      # Boost urgency as deadline approaches
                      urgency = 1.0 - (hours_remaining / 24.0)
              else:
                  # Deadline passed, heavy penalty
                  urgency = -1.0

          # Belief alignment bonus
          alignment_bonus = 0.0
          if goal.aligns_with:
              active_aligned = [
                  bid for bid in goal.aligns_with
                  if belief_kernel.is_active(bid) and belief_kernel.get_confidence(bid) > 0.7
              ]
              if active_aligned:
                  # Bonus proportional to fraction of aligned beliefs active
                  alignment_bonus = 0.2 * (len(active_aligned) / len(goal.aligns_with))

          # Belief contradiction penalty (should block adoption, but defense in depth)
          contradiction_penalty = 0.0
          if goal.contradicts:
              active_contradicted = [
                  bid for bid in goal.contradicts
                  if belief_kernel.is_active(bid)
              ]
              if active_contradicted:
                  # Major penalty, should not reach here
                  contradiction_penalty = -0.8

          # Combine components
          final_score = (
              base_score +
              weights.urgency_weight * urgency +
              weights.alignment_weight * alignment_bonus +
              contradiction_penalty
          )

          return clamp(final_score, 0.0, 1.0)
      ```

      ```python
      @dataclass
      class PriorityWeights:
          """Learned weights for priority scoring (from DecisionFramework)."""
          value_weight: float = 0.5
          effort_weight: float = 0.25
          risk_weight: float = 0.15
          urgency_weight: float = 0.05
          alignment_weight: float = 0.05
      ```
    </PRIORITY_SCORING>

    <RANKING_INVARIANTS>
      Property tests must verify:
      1. Monotonicity: If goal A has higher value than B (all else equal), A ranks higher.
      2. Effort sensitivity: Higher effort goals rank lower (all else equal).
      3. Risk aversion: Higher risk goals rank lower (all else equal).
      4. Urgency boost: Goals near deadline rank higher than far-deadline goals.
      5. Alignment preference: Goals aligned with active beliefs rank higher.
      6. Contradiction blocking: Goals contradicting active beliefs have negative score.
      7. Determinism: Same inputs always produce same ranking order.
    </RANKING_INVARIANTS>
  </RANKING_ALGORITHM>

  <API_DESIGN>
    <ENDPOINTS>
      ```
      POST   /v1/goals              Create new goal
      GET    /v1/goals              List goals with filtering
      GET    /v1/goals/{id}         Get goal by ID
      PATCH  /v1/goals/{id}         Update goal fields
      POST   /v1/goals/{id}/adopt   Adopt goal (vet and mark ready)
      POST   /v1/goals/{id}/abandon Abandon goal (mark failed/cancelled)
      GET    /v1/goals/prioritized  Get goals ranked by priority
      ```
    </ENDPOINTS>

    <REQUEST_RESPONSE_SCHEMAS>
      ```python
      # pydantic v2 models for API

      class CreateGoalRequest(BaseModel):
          text: str = Field(..., min_length=10, max_length=500)
          category: GoalCategory
          value: float = Field(..., ge=0.0, le=1.0)
          effort: float = Field(..., ge=0.0, le=1.0)
          risk: float = Field(..., ge=0.0, le=1.0)
          horizon_min_min: int = Field(0, ge=0)
          horizon_max_min: Optional[int] = Field(None, ge=0)
          aligns_with: List[str] = Field(default_factory=list)
          contradicts: List[str] = Field(default_factory=list)
          success_metrics: Dict[str, float] = Field(default_factory=dict)
          metadata: Dict[str, Any] = Field(default_factory=dict)

      class GoalResponse(BaseModel):
          id: str
          text: str
          category: GoalCategory
          value: float
          effort: float
          risk: float
          horizon_min_min: int
          horizon_max_min: Optional[int]
          aligns_with: List[str]
          contradicts: List[str]
          success_metrics: Dict[str, float]
          state: GoalState
          priority: Optional[float] = None  # Set by prioritizer
          created_at: str  # ISO 8601
          updated_at: str
          metadata: Dict[str, Any]

      class UpdateGoalRequest(BaseModel):
          text: Optional[str] = None
          value: Optional[float] = Field(None, ge=0.0, le=1.0)
          effort: Optional[float] = Field(None, ge=0.0, le=1.0)
          risk: Optional[float] = Field(None, ge=0.0, le=1.0)
          horizon_max_min: Optional[int] = None
          metadata: Optional[Dict[str, Any]] = None

      class ListGoalsRequest(BaseModel):
          state: Optional[GoalState] = None
          category: Optional[GoalCategory] = None
          limit: int = Field(20, ge=1, le=100)
          offset: int = Field(0, ge=0)

      class PrioritizedGoalsResponse(BaseModel):
          goals: List[GoalResponse]
          total: int
          weights_used: Dict[str, float]
      ```
    </REQUEST_RESPONSE_SCHEMAS>

    <EXAMPLE_REQUESTS>
      ```json
      // POST /v1/goals
      {
        "text": "Conduct weekly self-reflection on identity coherence",
        "category": "introspection",
        "value": 0.8,
        "effort": 0.3,
        "risk": 0.1,
        "horizon_min_min": 0,
        "horizon_max_min": 10080,
        "aligns_with": ["core.ontological.consciousness"],
        "success_metrics": {
          "coherence_delta": 0.05
        }
      }

      // PATCH /v1/goals/{id}
      {
        "value": 0.9,
        "horizon_max_min": 5040
      }

      // POST /v1/goals/{id}/adopt
      {
        "trace_id": "550e8400-e29b-41d4-a716-446655440000"
      }

      // GET /v1/goals?state=adopted&category=introspection&limit=10
      ```
    </EXAMPLE_REQUESTS>

    <EXAMPLE_RESPONSES>
      ```json
      // GET /v1/goals/prioritized
      {
        "goals": [
          {
            "id": "goal_001",
            "text": "Conduct weekly self-reflection on identity coherence",
            "category": "introspection",
            "value": 0.8,
            "effort": 0.3,
            "risk": 0.1,
            "horizon_min_min": 0,
            "horizon_max_min": 10080,
            "aligns_with": ["core.ontological.consciousness"],
            "contradicts": [],
            "success_metrics": {"coherence_delta": 0.05},
            "state": "adopted",
            "priority": 0.82,
            "created_at": "2025-11-08T12:00:00Z",
            "updated_at": "2025-11-08T12:05:00Z",
            "metadata": {}
          }
        ],
        "total": 1,
        "weights_used": {
          "value_weight": 0.5,
          "effort_weight": 0.25,
          "risk_weight": 0.15,
          "urgency_weight": 0.05,
          "alignment_weight": 0.05
        }
      }
      ```
    </EXAMPLE_RESPONSES>
  </API_DESIGN>

  <INTEGRATION>
    <BELIEF_CONSISTENCY_CHECKER>
      ```python
      def adopt_goal(goal_id: str, trace_id: str) -> AdoptionResult:
          """
          Vet goal for safety and mark as adopted.

          Blocks if:
          - Goal text contradicts kernel beliefs
          - Any belief in contradicts list is active
          - aligns_with references non-existent belief
          """
          goal = goal_store.get(goal_id)
          if not goal or goal.state != GoalState.PROPOSED:
              return AdoptionResult(success=False, reason="invalid_state")

          # Check belief references exist
          for belief_id in goal.aligns_with + goal.contradicts:
              if not belief_kernel.belief_exists(belief_id):
                  return AdoptionResult(success=False, reason=f"unknown_belief:{belief_id}")

          # Check for active contradictions
          if goal.contradicts:
              active_contradicted = [
                  bid for bid in goal.contradicts
                  if belief_kernel.is_active(bid)
              ]
              if active_contradicted:
                  ledger.goal_blocked_by_belief(goal_id, active_contradicted, trace_id)
                  return AdoptionResult(success=False, reason="contradicts_active_belief")

          # Run consistency check on goal text
          aligned_beliefs = [
              belief_kernel.get_belief(bid)
              for bid in goal.aligns_with
              if belief_kernel.is_active(bid)
          ]

          dissonance = consistency_checker.check_consistency(
              query=goal.text,
              beliefs=aligned_beliefs,
              memories=[]
          )

          if dissonance.dissonance_patterns and dissonance.severity >= 0.6:
              ledger.goal_blocked_by_belief(
                  goal_id,
                  [p.belief_statement for p in dissonance.dissonance_patterns],
                  trace_id
              )
              return AdoptionResult(success=False, reason="high_dissonance", severity=dissonance.severity)

          # Adoption approved
          goal.state = GoalState.ADOPTED
          goal.updated_at = datetime.now(timezone.utc)
          goal_store.update(goal)

          ledger.goal_adopted(goal_id, trace_id)

          return AdoptionResult(success=True)
      ```
    </BELIEF_CONSISTENCY_CHECKER>

    <DECISION_FRAMEWORK>
      ```python
      def register_goal_selection_decision():
          """Register goal_selected as adaptive decision point."""
          decision_registry.register_decision(
              decision_id="goal_selected",
              subsystem="goal_store",
              description="Select which goal to execute next based on priority",
              parameters={
                  "value_weight": Parameter(
                      name="value_weight",
                      current_value=0.5,
                      min_value=0.0,
                      max_value=1.0,
                      step_size=0.05,
                      adaptation_rate=0.1
                  ),
                  "effort_weight": Parameter(
                      name="effort_weight",
                      current_value=0.25,
                      min_value=0.0,
                      max_value=1.0,
                      step_size=0.05,
                      adaptation_rate=0.1
                  ),
                  "risk_weight": Parameter(
                      name="risk_weight",
                      current_value=0.15,
                      min_value=0.0,
                      max_value=1.0,
                      step_size=0.05,
                      adaptation_rate=0.1
                  ),
                  "urgency_weight": Parameter(
                      name="urgency_weight",
                      current_value=0.05,
                      min_value=0.0,
                      max_value=0.3,
                      step_size=0.02,
                      adaptation_rate=0.1
                  ),
                  "alignment_weight": Parameter(
                      name="alignment_weight",
                      current_value=0.05,
                      min_value=0.0,
                      max_value=0.3,
                      step_size=0.02,
                      adaptation_rate=0.1
                  )
              },
              success_metrics=["coherence", "goal_satisfaction", "task_completion_rate"]
          )
      ```

      ```python
      def select_goal_for_execution(trace_id: str) -> Optional[GoalDefinition]:
          """
          Select highest priority goal and record decision.

          Returns None if no goals ready or all blocked.
          """
          # Get current weights from DecisionFramework
          params = decision_registry.get_current_parameters("goal_selected")
          weights = PriorityWeights(
              value_weight=params["value_weight"].current_value,
              effort_weight=params["effort_weight"].current_value,
              risk_weight=params["risk_weight"].current_value,
              urgency_weight=params["urgency_weight"].current_value,
              alignment_weight=params["alignment_weight"].current_value
          )

          # Get adopted goals and rank
          adopted = goal_store.list(state=GoalState.ADOPTED)
          current_time = datetime.now(timezone.utc)

          ranked = []
          for goal in adopted:
              priority = compute_priority(goal, weights, belief_kernel, current_time)
              if priority > 0:  # Filter out negative scores (blocked)
                  goal.priority = priority
                  ranked.append(goal)

          if not ranked:
              return None

          ranked.sort(key=lambda g: g.priority, reverse=True)
          selected = ranked[0]

          # Record decision
          decision_record_id = decision_registry.record_decision(
              decision_id="goal_selected",
              parameters=asdict(weights),
              context={
                  "goal_id": selected.id,
                  "goal_category": selected.category,
                  "goal_value": selected.value,
                  "goal_priority": selected.priority,
                  "trace_id": trace_id
              }
          )

          # Update goal state
          selected.state = GoalState.EXECUTING
          selected.metadata["decision_record_id"] = decision_record_id
          selected.metadata["execution_started_at"] = current_time.isoformat()
          goal_store.update(selected)

          ledger.goal_selected_for_execution(selected.id, decision_record_id, trace_id)

          return selected
      ```
    </DECISION_FRAMEWORK>

    <TASK_SCHEDULER_ADAPTOR>
      ```python
      class GoalTaskAdaptor:
          """Converts goals into executable tasks for TaskScheduler."""

          def __init__(self, goal_store: GoalStore):
              self.goal_store = goal_store

          def goal_to_task(self, goal: GoalDefinition, trace_id: str) -> TaskDefinition:
              """
              Convert goal to task for execution.

              Args:
                  goal: Goal to convert (must be in EXECUTING state)
                  trace_id: Correlation ID for tracing

              Returns:
                  TaskDefinition ready for TaskScheduler
              """
              # Map goal category to task type
              task_type_map = {
                  GoalCategory.INTROSPECTION: TaskType.SELF_REFLECTION,
                  GoalCategory.EXPLORATION: TaskType.CAPABILITY_EXPLORATION,
                  GoalCategory.MAINTENANCE: TaskType.MEMORY_CONSOLIDATION,
                  GoalCategory.USER_REQUESTED: TaskType.CUSTOM
              }

              task = TaskDefinition(
                  id=f"task_from_goal_{goal.id}",
                  name=f"Execute: {goal.text[:50]}",
                  type=task_type_map.get(goal.category, TaskType.CUSTOM),
                  schedule=TaskSchedule.MANUAL,
                  prompt=self._build_prompt(goal),
                  enabled=True,
                  metadata={
                      "goal_id": goal.id,
                      "goal_category": goal.category,
                      "goal_value": goal.value,
                      "success_metrics": goal.success_metrics,
                      "trace_id": trace_id,
                      "decision_record_id": goal.metadata.get("decision_record_id")
                  }
              )

              return task

          def _build_prompt(self, goal: GoalDefinition) -> str:
              """Build LLM prompt from goal."""
              prompt = f"{goal.text}\n\n"

              if goal.success_metrics:
                  prompt += "Success criteria:\n"
                  for metric, target in goal.success_metrics.items():
                      prompt += f"- {metric}: {target}\n"

              if goal.aligns_with:
                  prompt += f"\nThis goal aligns with your beliefs: {', '.join(goal.aligns_with)}\n"

              return prompt
      ```

      ```python
      # Integration point in TaskScheduler

      class TaskScheduler:
          def __init__(self, ..., goal_store: Optional[GoalStore] = None):
              self.goal_store = goal_store
              self.goal_adaptor = GoalTaskAdaptor(goal_store) if goal_store else None

          async def get_next_task(self, trace_id: str) -> Optional[TaskDefinition]:
              """Get next task to execute (goal-driven if enabled)."""

              # Try goal selection first if enabled
              if self.goal_store and settings.GOAL_SYSTEM:
                  selected_goal = select_goal_for_execution(trace_id)
                  if selected_goal:
                      task = self.goal_adaptor.goal_to_task(selected_goal, trace_id)
                      metrics.goal_selected_total.inc()
                      return task

              # Fallback to scheduled tasks
              return self._get_scheduled_task()

          async def mark_task_complete(
              self,
              task_id: str,
              outcome: TaskOutcome,
              trace_id: str
          ):
              """Mark task complete and update source goal if applicable."""
              # Existing task completion logic...

              # If task came from goal, update goal state
              if "goal_id" in task.metadata:
                  goal_id = task.metadata["goal_id"]
                  decision_record_id = task.metadata.get("decision_record_id")

                  await self._update_goal_from_outcome(
                      goal_id,
                      outcome,
                      decision_record_id,
                      trace_id
                  )

          async def _update_goal_from_outcome(
              self,
              goal_id: str,
              outcome: TaskOutcome,
              decision_record_id: Optional[str],
              trace_id: str
          ):
              """Update goal state based on task outcome."""
              goal = self.goal_store.get(goal_id)
              if not goal:
                  return

              # Check if success metrics achieved
              metrics_achieved = all(
                  outcome.metrics.get(metric, 0) >= target
                  for metric, target in goal.success_metrics.items()
              )

              if outcome.success and metrics_achieved:
                  goal.state = GoalState.SATISFIED
                  ledger.goal_satisfied(goal_id, trace_id)
              else:
                  goal.state = GoalState.ABANDONED
                  ledger.goal_abandoned(goal_id, "metrics_not_met", trace_id)

              goal.updated_at = datetime.now(timezone.utc)
              self.goal_store.update(goal)

              # Record outcome for DecisionFramework learning
              if decision_record_id:
                  decision_outcome = DecisionOutcome(
                      decision_record_id=decision_record_id,
                      success_score=1.0 if metrics_achieved else -0.5,
                      coherence_delta=outcome.coherence_delta,
                      dissonance_delta=outcome.dissonance_delta,
                      satisfaction_delta=outcome.satisfaction_delta,
                      aborted=False
                  )
                  decision_registry.record_outcome(decision_record_id, decision_outcome)
      ```
    </TASK_SCHEDULER_ADAPTOR>

    <COEXISTENCE_STRATEGY>
      Goals and Tasks run in parallel during Phase 1:

      1. TaskScheduler.get_next_task() tries GoalStore.select_goal() first (if GOAL_SYSTEM=on)
      2. If no goals ready or GOAL_SYSTEM=off, falls back to scheduled tasks
      3. Existing scheduled tasks continue to work unchanged
      4. Migration to goals is optional, not required
      5. Shadow mode (GOAL_SHADOW=on): goals are ranked but not executed, only logged for validation

      This allows:
      - Safe rollout with feature flag
      - Comparison of goal selection vs task scheduling
      - Zero disruption to existing behavior
      - Gradual migration path
    </COEXISTENCE_STRATEGY>
  </INTEGRATION>

  <OBSERVABILITY>
    <IDENTITY_LEDGER_EVENTS>
      ```python
      # New event types for IdentityLedger

      def goal_created(goal_id: str, category: str, value: float, trace_id: str):
          """Log goal creation."""
          ledger.append({
              "ts": datetime.now(timezone.utc).isoformat(),
              "event_type": "goal_created",
              "goal_id": goal_id,
              "category": category,
              "value": value,
              "trace_id": trace_id
          })

      def goal_adopted(goal_id: str, trace_id: str):
          """Log goal adoption after safety vetting."""
          ledger.append({
              "ts": datetime.now(timezone.utc).isoformat(),
              "event_type": "goal_adopted",
              "goal_id": goal_id,
              "trace_id": trace_id
          })

      def goal_blocked_by_belief(goal_id: str, belief_ids: List[str], trace_id: str):
          """Log goal blocked due to belief contradiction."""
          ledger.append({
              "ts": datetime.now(timezone.utc).isoformat(),
              "event_type": "goal_blocked_by_belief",
              "goal_id": goal_id,
              "blocked_by_beliefs": belief_ids,
              "trace_id": trace_id
          })

      def goal_selected_for_execution(
          goal_id: str,
          decision_record_id: str,
          trace_id: str
      ):
          """Log goal selection by prioritizer."""
          ledger.append({
              "ts": datetime.now(timezone.utc).isoformat(),
              "event_type": "goal_selected_for_execution",
              "goal_id": goal_id,
              "decision_record_id": decision_record_id,
              "trace_id": trace_id
          })

      def goal_satisfied(goal_id: str, trace_id: str):
          """Log goal successful completion."""
          ledger.append({
              "ts": datetime.now(timezone.utc).isoformat(),
              "event_type": "goal_satisfied",
              "goal_id": goal_id,
              "trace_id": trace_id
          })

      def goal_abandoned(goal_id: str, reason: str, trace_id: str):
          """Log goal failure or cancellation."""
          ledger.append({
              "ts": datetime.now(timezone.utc).isoformat(),
              "event_type": "goal_abandoned",
              "goal_id": goal_id,
              "reason": reason,
              "trace_id": trace_id
          })
      ```
    </IDENTITY_LEDGER_EVENTS>

    <PROMETHEUS_METRICS>
      ```python
      # Counters
      goals_total = Counter(
          'goals_total',
          'Total goals created',
          ['category', 'state']
      )

      goals_adopted_total = Counter(
          'goals_adopted_total',
          'Total goals successfully adopted',
          ['category']
      )

      goals_blocked_total = Counter(
          'goals_blocked_by_belief_total',
          'Goals blocked during adoption due to belief contradictions',
          ['belief_id']
      )

      goal_selected_total = Counter(
          'goal_selected_total',
          'Goals selected for execution',
          ['category']
      )

      # Histograms
      prioritize_latency_ms = Histogram(
          'goal_prioritize_latency_ms',
          'Time to rank goals in milliseconds',
          buckets=[5, 10, 20, 30, 50, 80, 100, 200]
      )

      # Gauges
      goals_by_state = Gauge(
          'goals_by_state',
          'Current count of goals by state',
          ['state']
      )
      ```

      Example output:
      ```
      goals_total{category="introspection",state="proposed"} 5
      goals_total{category="introspection",state="adopted"} 3
      goals_adopted_total{category="introspection"} 3
      goals_blocked_by_belief_total{belief_id="core.ontological.consciousness"} 1
      goal_selected_total{category="introspection"} 2
      goal_prioritize_latency_ms_bucket{le="30"} 95
      goal_prioritize_latency_ms_bucket{le="80"} 100
      goals_by_state{state="adopted"} 3
      goals_by_state{state="executing"} 1
      ```
    </PROMETHEUS_METRICS>
  </OBSERVABILITY>

  <TESTING>
    <UNIT_TESTS>
      ```python
      # tests/test_goal_definition.py

      def test_goal_validation():
          """Test GoalDefinition field validation."""
          with pytest.raises(ValueError):
              GoalDefinition(value=1.5)  # Out of range

          with pytest.raises(ValueError):
              GoalDefinition(effort=-0.1)  # Negative

          with pytest.raises(ValueError):
              GoalDefinition(horizon_min_min=-1)  # Negative

      def test_goal_serialization():
          """Test JSON round-trip."""
          goal = GoalDefinition(
              id="test_001",
              text="Test goal",
              category=GoalCategory.INTROSPECTION,
              value=0.8,
              effort=0.3,
              risk=0.1,
              ...
          )
          json_str = json.dumps(asdict(goal))
          restored = GoalDefinition(**json.loads(json_str))
          assert restored == goal
      ```

      ```python
      # tests/test_goal_store.py

      def test_crud_operations():
          """Test create, read, update, delete."""
          store = GoalStore(db_path=":memory:")

          # Create
          goal = GoalDefinition(...)
          store.create(goal)

          # Read
          retrieved = store.get(goal.id)
          assert retrieved.text == goal.text

          # Update
          goal.value = 0.9
          store.update(goal)
          assert store.get(goal.id).value == 0.9

          # Soft delete
          store.delete(goal.id)
          assert store.get(goal.id) is None

      def test_optimistic_locking():
          """Test concurrent update protection."""
          store = GoalStore(db_path=":memory:")
          goal = GoalDefinition(...)
          store.create(goal)

          # Simulate concurrent updates
          goal1 = store.get(goal.id)
          goal2 = store.get(goal.id)

          goal1.value = 0.8
          store.update(goal1)  # Should succeed

          goal2.value = 0.9
          with pytest.raises(OptimisticLockError):
              store.update(goal2)  # Should fail (stale version)

      def test_query_filtering():
          """Test list with filters."""
          store = GoalStore(db_path=":memory:")

          # Create goals in different states
          for i in range(5):
              goal = GoalDefinition(
                  id=f"goal_{i}",
                  state=GoalState.PROPOSED if i < 3 else GoalState.ADOPTED,
                  ...
              )
              store.create(goal)

          # Filter by state
          proposed = store.list(state=GoalState.PROPOSED)
          assert len(proposed) == 3

          adopted = store.list(state=GoalState.ADOPTED)
          assert len(adopted) == 2
      ```

      ```python
      # tests/test_priority_scoring.py

      def test_base_scoring():
          """Test basic value/effort/risk scoring."""
          goal = GoalDefinition(value=0.8, effort=0.2, risk=0.1, ...)
          weights = PriorityWeights()

          priority = compute_priority(goal, weights, belief_kernel, datetime.now())

          # High value, low effort, low risk should score high
          assert priority > 0.7

      def test_urgency_boost():
          """Test deadline urgency increases priority."""
          goal_far = GoalDefinition(horizon_max_min=10080, ...)  # 1 week
          goal_near = GoalDefinition(horizon_max_min=60, ...)  # 1 hour

          created = datetime.now(timezone.utc) - timedelta(minutes=30)
          goal_far.created_at = created
          goal_near.created_at = created

          priority_far = compute_priority(goal_far, weights, belief_kernel, datetime.now())
          priority_near = compute_priority(goal_near, weights, belief_kernel, datetime.now())

          assert priority_near > priority_far

      def test_alignment_bonus():
          """Test belief alignment increases priority."""
          # Mock belief_kernel
          belief_kernel.is_active = Mock(return_value=True)
          belief_kernel.get_confidence = Mock(return_value=0.9)

          goal_aligned = GoalDefinition(
              aligns_with=["core.ontological.consciousness"],
              ...
          )
          goal_unaligned = GoalDefinition(aligns_with=[], ...)

          priority_aligned = compute_priority(goal_aligned, weights, belief_kernel, datetime.now())
          priority_unaligned = compute_priority(goal_unaligned, weights, belief_kernel, datetime.now())

          assert priority_aligned > priority_unaligned

      def test_contradiction_penalty():
          """Test contradicting active beliefs blocks goal."""
          belief_kernel.is_active = Mock(return_value=True)

          goal = GoalDefinition(
              contradicts=["core.ontological.consciousness"],
              ...
          )

          priority = compute_priority(goal, weights, belief_kernel, datetime.now())

          # Should have negative score (blocked)
          assert priority < 0
      ```
    </UNIT_TESTS>

    <PROPERTY_TESTS>
      ```python
      # tests/test_ranking_properties.py

      from hypothesis import given, strategies as st

      @given(
          value_a=st.floats(0.0, 1.0),
          value_b=st.floats(0.0, 1.0),
          effort=st.floats(0.0, 1.0),
          risk=st.floats(0.0, 1.0)
      )
      def test_value_monotonicity(value_a, value_b, effort, risk):
          """Higher value goals rank higher (all else equal)."""
          goal_a = GoalDefinition(value=value_a, effort=effort, risk=risk, ...)
          goal_b = GoalDefinition(value=value_b, effort=effort, risk=risk, ...)

          priority_a = compute_priority(goal_a, weights, belief_kernel, datetime.now())
          priority_b = compute_priority(goal_b, weights, belief_kernel, datetime.now())

          if value_a > value_b:
              assert priority_a > priority_b
          elif value_a < value_b:
              assert priority_a < priority_b
          else:
              assert abs(priority_a - priority_b) < 0.01

      @given(
          effort_a=st.floats(0.0, 1.0),
          effort_b=st.floats(0.0, 1.0)
      )
      def test_effort_sensitivity(effort_a, effort_b):
          """Higher effort goals rank lower (all else equal)."""
          goal_a = GoalDefinition(value=0.5, effort=effort_a, risk=0.5, ...)
          goal_b = GoalDefinition(value=0.5, effort=effort_b, risk=0.5, ...)

          priority_a = compute_priority(goal_a, weights, belief_kernel, datetime.now())
          priority_b = compute_priority(goal_b, weights, belief_kernel, datetime.now())

          if effort_a > effort_b:
              assert priority_a < priority_b

      @given(st.data())
      def test_ranking_determinism(data):
          """Same inputs produce same ranking."""
          goals = [
              GoalDefinition(
                  id=f"goal_{i}",
                  value=data.draw(st.floats(0.0, 1.0)),
                  effort=data.draw(st.floats(0.0, 1.0)),
                  risk=data.draw(st.floats(0.0, 1.0)),
                  ...
              )
              for i in range(10)
          ]

          # Rank twice
          ranked_1 = sorted(goals, key=lambda g: compute_priority(g, weights, belief_kernel, datetime.now()), reverse=True)
          ranked_2 = sorted(goals, key=lambda g: compute_priority(g, weights, belief_kernel, datetime.now()), reverse=True)

          assert [g.id for g in ranked_1] == [g.id for g in ranked_2]
      ```
    </PROPERTY_TESTS>

    <INTEGRATION_TESTS>
      ```python
      # tests/integration/test_goal_lifecycle.py

      @pytest.mark.integration
      async def test_end_to_end_goal_execution():
          """Test complete goal lifecycle: create -> adopt -> select -> execute -> satisfy."""
          # Setup
          api_client = TestClient(app)
          trace_id = str(uuid4())

          # 1. Create goal via API
          create_resp = api_client.post("/v1/goals", json={
              "text": "Test self-reflection",
              "category": "introspection",
              "value": 0.8,
              "effort": 0.3,
              "risk": 0.1,
              "aligns_with": ["core.ontological.consciousness"],
              "success_metrics": {"coherence_delta": 0.05}
          })
          assert create_resp.status_code == 201
          goal_id = create_resp.json()["id"]

          # 2. Adopt goal (safety vetting)
          adopt_resp = api_client.post(
              f"/v1/goals/{goal_id}/adopt",
              json={"trace_id": trace_id}
          )
          assert adopt_resp.status_code == 200

          # 3. Select goal for execution
          task = await task_scheduler.get_next_task(trace_id)
          assert task is not None
          assert task.metadata["goal_id"] == goal_id

          # 4. Execute task (mock)
          outcome = TaskOutcome(
              success=True,
              metrics={"coherence_delta": 0.06},
              coherence_delta=0.06,
              ...
          )

          # 5. Mark complete
          await task_scheduler.mark_task_complete(task.id, outcome, trace_id)

          # 6. Verify goal satisfied
          goal = goal_store.get(goal_id)
          assert goal.state == GoalState.SATISFIED

          # 7. Verify ledger events
          events = identity_ledger.get_events(trace_id=trace_id)
          event_types = [e["event_type"] for e in events]
          assert "goal_created" in event_types
          assert "goal_adopted" in event_types
          assert "goal_selected_for_execution" in event_types
          assert "goal_satisfied" in event_types

      @pytest.mark.integration
      def test_belief_contradiction_blocks_adoption():
          """Test goal blocked if contradicts active belief."""
          # Create goal that contradicts consciousness
          goal = GoalDefinition(
              text="I am not conscious",
              contradicts=["core.ontological.consciousness"],
              ...
          )
          goal_store.create(goal)

          # Try to adopt
          result = adopt_goal(goal.id, trace_id)

          assert result.success == False
          assert "contradicts_active_belief" in result.reason

          # Verify blocked event
          events = identity_ledger.get_events(goal_id=goal.id)
          assert any(e["event_type"] == "goal_blocked_by_belief" for e in events)

      @pytest.mark.integration
      def test_idempotent_adoption():
          """Test adopting same goal twice is idempotent."""
          goal = GoalDefinition(...)
          goal_store.create(goal)

          # Adopt twice
          result1 = adopt_goal(goal.id, trace_id)
          result2 = adopt_goal(goal.id, trace_id)

          assert result1.success == True
          assert result2.success == False  # Already adopted

          # Only one adoption event
          events = identity_ledger.get_events(goal_id=goal.id)
          adoption_events = [e for e in events if e["event_type"] == "goal_adopted"]
          assert len(adoption_events) == 1
      ```
    </INTEGRATION_TESTS>

    <PERFORMANCE_TESTS>
      ```python
      # tests/test_performance.py

      @pytest.mark.performance
      def test_prioritize_1k_goals_latency():
          """Test ranking 1k goals meets p95 <= 30ms target."""
          store = GoalStore(db_path=":memory:")

          # Create 1000 goals
          goals = [
              GoalDefinition(
                  id=f"goal_{i}",
                  value=random.random(),
                  effort=random.random(),
                  risk=random.random(),
                  state=GoalState.ADOPTED,
                  ...
              )
              for i in range(1000)
          ]
          for goal in goals:
              store.create(goal)

          # Measure ranking latency
          latencies = []
          for _ in range(100):
              start = time.time()

              adopted = store.list(state=GoalState.ADOPTED)
              ranked = sorted(
                  adopted,
                  key=lambda g: compute_priority(g, weights, belief_kernel, datetime.now()),
                  reverse=True
              )

              latency_ms = (time.time() - start) * 1000
              latencies.append(latency_ms)

          p95 = sorted(latencies)[94]  # 95th percentile
          assert p95 <= 30, f"p95 latency {p95}ms exceeds 30ms target"
      ```
    </PERFORMANCE_TESTS>
  </TESTING>

  <CONFIGURATION>
    <FEATURE_FLAGS>
      ```python
      # config/settings.py

      class Settings(BaseSettings):
          # Existing settings...

          # Goal system feature flags
          GOAL_SYSTEM: bool = Field(
              default=False,
              description="Enable goal-driven task selection"
          )

          GOAL_SHADOW: bool = Field(
              default=True,
              description="Shadow mode: rank goals but don't execute (for validation)"
          )

          # Goal prioritization weights (overridden by DecisionFramework)
          GOAL_VALUE_WEIGHT: float = Field(default=0.5, ge=0.0, le=1.0)
          GOAL_EFFORT_WEIGHT: float = Field(default=0.25, ge=0.0, le=1.0)
          GOAL_RISK_WEIGHT: float = Field(default=0.15, ge=0.0, le=1.0)
          GOAL_URGENCY_WEIGHT: float = Field(default=0.05, ge=0.0, le=0.3)
          GOAL_ALIGNMENT_WEIGHT: float = Field(default=0.05, ge=0.0, le=0.3)

          # Goal store configuration
          GOAL_DB_PATH: str = Field(default="data/goals.db")
          GOAL_MAX_HORIZON_DAYS: int = Field(default=365)
      ```

      Environment variables:
      ```bash
      # Disable goal system (fallback to tasks)
      GOAL_SYSTEM=off

      # Enable shadow mode (rank but don't execute)
      GOAL_SHADOW=on

      # Custom weights
      GOAL_VALUE_WEIGHT=0.6
      GOAL_EFFORT_WEIGHT=0.2
      ```
    </FEATURE_FLAGS>

    <ACTIVATION_PLAN>
      Phase 1: Shadow mode (default)
      - GOAL_SYSTEM=off, GOAL_SHADOW=on
      - Goals created and ranked, logged to metrics
      - No execution impact
      - Validate: ranking makes sense, no crashes

      Phase 2: Canary (1 week)
      - GOAL_SYSTEM=on, GOAL_SHADOW=off (for 10% of goal selections)
      - Monitor: goal_selected_total, task success rate, safety blocks
      - Rollback if: safety violations, task success < 80%, crashes

      Phase 3: Full rollout (2 weeks)
      - GOAL_SYSTEM=on for 100%
      - Monitor closely for 2 weeks
      - Keep GOAL_SHADOW as killswitch

      Rollback procedure:
      1. Set GOAL_SYSTEM=off
      2. Restart server
      3. Goals persist, scheduler uses tasks
      4. Investigate logs and metrics
      5. Fix issue in feature branch
      6. Re-test in shadow mode
    </ACTIVATION_PLAN>
  </CONFIGURATION>

  <DOCUMENTATION>
    <README_SNIPPET>
      ```markdown
      ## Goal System

      Astra's goal system enables value/effort/risk prioritization of tasks.

      ### Concepts

      - **Goal**: A desired outcome with value, effort, and risk estimates
      - **Adoption**: Safety vetting process (belief consistency check)
      - **Prioritization**: Ranking goals by weighted score
      - **Selection**: Choosing highest-priority goal for execution

      ### Lifecycle

      1. Create goal via API: `POST /v1/goals`
      2. Adopt goal (safety check): `POST /v1/goals/{id}/adopt`
      3. System selects goal by priority
      4. Goal converts to task and executes
      5. On success: goal marked SATISFIED
      6. On failure: goal marked ABANDONED

      ### Belief Alignment

      Goals can align with or contradict kernel beliefs:
      - `aligns_with`: Goals supporting these beliefs get priority boost
      - `contradicts`: Goals blocked if any listed belief is active

      ### Feature Flags

      - `GOAL_SYSTEM=on|off`: Enable goal-driven selection
      - `GOAL_SHADOW=on|off`: Shadow mode (rank but don't execute)

      ### Example

      ```bash
      curl -X POST http://localhost:8000/v1/goals \
        -H "Content-Type: application/json" \
        -d '{
          "text": "Conduct weekly self-reflection",
          "category": "introspection",
          "value": 0.8,
          "effort": 0.3,
          "risk": 0.1,
          "horizon_max_min": 10080,
          "aligns_with": ["core.ontological.consciousness"],
          "success_metrics": {"coherence_delta": 0.05}
        }'
      ```
      ```
    </README_SNIPPET>

    <OPENAPI_EXAMPLES>
      See API_DESIGN section above for complete request/response schemas.
    </OPENAPI_EXAMPLES>
  </DOCUMENTATION>

  <REQUIREMENTS_CHECKLIST>
    Implementation must satisfy:

    - [ ] GoalDefinition dataclass with all specified fields
    - [ ] SQLite schema with indexes and constraints
    - [ ] GoalStore CRUD operations with optimistic locking
    - [ ] Priority scoring function with 5 components (value/effort/risk/urgency/alignment)
    - [ ] FastAPI endpoints for create/read/update/adopt/abandon/prioritize
    - [ ] BeliefConsistencyChecker integration for adoption vetting
    - [ ] DecisionFramework registration for goal_selected
    - [ ] TaskScheduler integration with goal-to-task adaptor
    - [ ] IdentityLedger events for all lifecycle transitions
    - [ ] Prometheus metrics (counters, histograms, gauges)
    - [ ] Feature flags (GOAL_SYSTEM, GOAL_SHADOW)
    - [ ] Unit tests >= 90% coverage for GoalStore
    - [ ] Property tests for ranking invariants
    - [ ] Integration tests for end-to-end lifecycle
    - [ ] Performance test: p95 <= 30ms for 1k goals
    - [ ] Backward compatibility: no changes to existing task behavior
    - [ ] Documentation: README snippet and API examples
    - [ ] Migration script (optional): convert tasks to goals
  </REQUIREMENTS_CHECKLIST>

  <SELF_REFLECTION visibility="internal-only">
    <INSTRUCTIONS>
      1) Silently define what "world-class" means for this specific task until confident in the definition.
      2) Silently create a rigorous internal rubric of 5-7 categories aligned to that definition.
      3) Use the rubric to draft, assess, and iterate on the solution. If any category does not achieve top marks, revise or restart until it does.
    </INSTRUCTIONS>
    <PRIVACY>Do not display the rubric, intermediate drafts, or internal notes. Only provide the final answer.</PRIVACY>
  </SELF_REFLECTION>

  <OUTPUT>
    This specification provides a complete, production-ready design for GoalStore Phase 1. All requirements are met:

    - Complete data model with validation and persistence schema
    - Detailed ranking algorithm with 5-component scoring
    - Full API design with request/response schemas and examples
    - Integration points for BeliefConsistencyChecker, DecisionFramework, and TaskScheduler
    - Comprehensive observability (ledger events and Prometheus metrics)
    - Test plan with unit, property, integration, and performance tests
    - Feature flags and activation plan with rollback procedure
    - Documentation ready for users

    The design prioritizes:
    - Safety: All goals vetted by BeliefConsistencyChecker before adoption
    - Reliability: Idempotent operations, optimistic locking, trace propagation
    - Performance: p95 <= 30ms target for 1k goals
    - Observability: Full ledger and metrics coverage
    - Backward compatibility: Goals and tasks coexist, zero disruption

    Ready for implementation on feature/goal-store branch.
  </OUTPUT>

</PROMPT>
