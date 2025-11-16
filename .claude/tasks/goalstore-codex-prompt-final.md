# GoalStore Phase 1 - Implementation Specification

<PROMPT>

  <ROLE>
    You are ChatGPT-5. Your objective is to design and implement Phase 1 of Astra's GoalStore: a value/effort/risk prioritization layer with persistence and APIs, integrated with TaskScheduler and Decision Framework, without disrupting safety or learning.
  </ROLE>

  <CONTEXT>
    Audience: Senior engineer (Codex).
    Stack: Python 3.12, FastAPI, SQLite, Redis present.
    Existing: TaskScheduler, Decision Framework, OutcomeEvaluator, IdentityLedger, BeliefConsistencyChecker, BeliefKernel.
    Branches: master stable; feature/phase0-integration-wiring (independent); feature/goal-store (this work).
    Success: Create, rank, adopt, and hand goals to TaskScheduler with ledger and metrics. No safety regressions.
  </CONTEXT>

  <DELIVERABLES>
    - Data model: GoalDefinition dataclass + SQLite migration.
    - GoalStore: CRUD, list, query by state, soft delete, optimistic locking.
    - Prioritizer: score(value, effort, risk, urgency, alignment), deterministic.
    - APIs: create/read/update/adopt/abandon/prioritized.
    - Scheduler integration: goal→task adaptor, idempotent submission, trace propagation.
    - Observability: Prometheus metrics, IdentityLedger events.
    - Tests: unit, property, integration, performance.
    - Flags: GOAL_SYSTEM, GOAL_SHADOW. Docs: README snippet and OpenAPI examples.
  </DELIVERABLES>

  <REQUIREMENTS>
    <SCOPE>
      - GoalDefinition fields: id, text, category, value, effort, risk, horizon_min_min, horizon_max_min?, aligns_with[], contradicts[], success_metrics{}, state, created_at, updated_at, metadata{}.
      - Ranker: final = wv*value + we*(1-effort) + wr*(1-risk) + wu*urgency + wa*alignment + penalty(contradiction). Clamp to [0,1].
      - APIs: POST /v1/goals, GET /v1/goals, GET /v1/goals/{id}, PATCH /v1/goals/{id}, POST /v1/goals/{id}/adopt, POST /v1/goals/{id}/abandon, GET /v1/goals/prioritized.
      - Adoption vetting: BeliefConsistencyChecker on goal text; block if severity ≥ 0.6 or contradicts active beliefs.
      - Decision Framework: register goal_selected parameters for weights; record outcomes.
      - TaskScheduler: prefer selected goal if GOAL_SYSTEM=on, else fallback to scheduled tasks.
      - Metrics: goals_total{category,state}, goals_adopted_total, goal_selected_total, goal_blocked_by_belief_total{belief_id}, goal_prioritize_latency_ms, goals_by_state{state}.
      - Ledger events: goal_created, goal_updated, goal_adopted, goal_selected_for_execution, goal_satisfied, goal_abandoned, goal_blocked_by_belief.
    </SCOPE>
    <OUT_OF_SCOPE>
      HTN planner, CRDTs, contextual bandits, proof kernel changes, auto task→goal migration.
    </OUT_OF_SCOPE>
    <FORMAT>
      Output a Markdown design with H1–H3 sections, JSON API schemas, and Python interface snippets.
    </FORMAT>
    <CONSTRAINTS>
      Python 3.12; dataclasses for model, pydantic v2 for API. Forward-only migration. Idempotent operations with trace_id/span_id. Shadow first. Backward compatible with persona and awareness.
    </CONSTRAINTS>
    <EVALUATION_CRITERIA>
      Tests: ≥90% GoalStore, ≥80% API. p95 prioritize ≤30 ms for 1k goals, ≤80 ms with DB fetch. Zero safety violations in CI pack. Idempotent adoption and selection. Metrics visible in dev dashboard.
    </EVALUATION_CRITERIA>
  </REQUIREMENTS>

  <INTEGRATION_DETAILS>
    <BELIEF_SAFETY>
      - Use BeliefConsistencyChecker.check_consistency(query=goal.text, beliefs=aligned_beliefs, memories=[])
      - Block adoption if severity ≥ 0.6
      - Block if any belief in contradicts[] is active
      - Emit goal_blocked_by_belief event with belief_ids
      - Note: Formal ConstraintGuard (proof kernel) planned for Phase 3
    </BELIEF_SAFETY>

    <DECISION_FRAMEWORK_REGISTRATION>
      ```python
      decision_registry.register_decision(
          decision_id="goal_selected",
          subsystem="goal_store",
          parameters={
              "value_weight": Parameter(current=0.5, min=0.0, max=1.0, step=0.05),
              "effort_weight": Parameter(current=0.25, min=0.0, max=1.0, step=0.05),
              "risk_weight": Parameter(current=0.15, min=0.0, max=1.0, step=0.05),
              "urgency_weight": Parameter(current=0.05, min=0.0, max=0.3, step=0.02),
              "alignment_weight": Parameter(current=0.05, min=0.0, max=0.3, step=0.02)
          },
          success_metrics=["coherence", "goal_satisfaction", "task_completion_rate"]
      )
      ```
    </DECISION_FRAMEWORK_REGISTRATION>

    <SCHEDULER_COEXISTENCE>
      Phase 1: Goals and Tasks run in parallel
      - TaskScheduler.get_next_task() tries GoalStore.select_goal() first if GOAL_SYSTEM=on
      - Falls back to scheduled tasks if no goals ready or GOAL_SYSTEM=off
      - Shadow mode (GOAL_SHADOW=on): goals ranked but not executed, only logged
      - Existing scheduled tasks continue unchanged
      - No migration required
    </SCHEDULER_COEXISTENCE>

    <HORIZON_SCORING>
      ```python
      # Urgency factor based on deadline proximity
      if goal.horizon_max_min:
          elapsed_min = (current_time - goal.created_at).total_seconds() / 60
          remaining_min = goal.horizon_max_min - elapsed_min
          if remaining_min > 0:
              hours_remaining = remaining_min / 60
              urgency = 0.0 if hours_remaining > 24 else (1.0 - hours_remaining/24)
          else:
              urgency = -1.0  # Deadline passed, penalty
      else:
          urgency = 0.0  # No deadline
      ```
    </HORIZON_SCORING>

    <ERROR_HANDLING>
      Adoption validation:
      - Reject if goal.text contradicts any belief in aligns_with (severity check)
      - Reject if any belief in contradicts[] is active
      - Warn if aligns_with[] references unknown belief_id
      - Reject if horizon_max < current_time

      Selection safety:
      - Re-check BeliefConsistencyChecker before task emission
      - Propagate trace_id through entire flow
      - Log all blocked goals to IdentityLedger with reason
    </ERROR_HANDLING>
  </INTEGRATION_DETAILS>

  <DATA_MODEL_EXAMPLES>
    ```python
    # GoalCategory enum
    class GoalCategory(str, Enum):
        INTROSPECTION = "introspection"
        EXPLORATION = "exploration"
        MAINTENANCE = "maintenance"
        USER_REQUESTED = "user_requested"

    # GoalState enum
    class GoalState(str, Enum):
        PROPOSED = "proposed"
        ADOPTED = "adopted"
        EXECUTING = "executing"
        SATISFIED = "satisfied"
        ABANDONED = "abandoned"
    ```

    Example goal JSON:
    ```json
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
      "state": "proposed",
      "created_at": "2025-11-08T12:00:00Z",
      "updated_at": "2025-11-08T12:00:00Z",
      "metadata": {}
    }
    ```
  </DATA_MODEL_EXAMPLES>

  <RANKING_INVARIANTS>
    Property tests must verify:
    1. Value monotonicity: Higher value → higher priority (all else equal)
    2. Effort sensitivity: Higher effort → lower priority
    3. Risk aversion: Higher risk → lower priority
    4. Urgency boost: Near deadline → higher priority
    5. Alignment preference: Aligned with active beliefs → bonus
    6. Contradiction blocking: Contradicts active belief → negative score
    7. Determinism: Same inputs → same ranking order
  </RANKING_INVARIANTS>

  <ACTIVATION_PLAN>
    Phase 1: Shadow (default)
    - GOAL_SYSTEM=off, GOAL_SHADOW=on
    - Goals created and ranked, logged to metrics
    - No execution impact
    - Validate: ranking sensible, no crashes

    Phase 2: Canary (1 week)
    - GOAL_SYSTEM=on for 10% of selections
    - Monitor: goal_selected_total, task success, safety blocks
    - Rollback if: violations, success < 80%, crashes

    Phase 3: Full rollout (2 weeks)
    - GOAL_SYSTEM=on for 100%
    - Monitor for 2 weeks
    - Keep GOAL_SHADOW as killswitch

    Rollback: Set GOAL_SYSTEM=off, restart server, scheduler uses tasks
  </ACTIVATION_PLAN>

  <SELF_REFLECTION visibility="internal-only">
    <INSTRUCTIONS>
      1) Define "world-class" for this task.
      2) Build a 5–7 category rubric.
      3) Iterate until all categories pass.
    </INSTRUCTIONS>
    <PRIVACY>Do not reveal rubric or drafts.</PRIVACY>
  </SELF_REFLECTION>

  <OUTPUT>
    Provide the final design only, meeting all sections above. Include:
    - Data model and migration.
    - Priority function and invariants.
    - API endpoints with request/response models and examples.
    - Scheduler and Decision Framework integration notes.
    - Metrics and ledger mapping.
    - Test plan with sample cases.
    - Activation and rollback with flags.
    Add a short checklist confirming requirements met.
  </OUTPUT>

</PROMPT>
