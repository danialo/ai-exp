# TODO List - Astra AI Experience

**Last Updated**: 2025-11-11

## In Progress

- [ ] **Code generation quality improvements**
  - Current: LLM generates generic test code instead of specific implementations
  - Need: Better prompt engineering for CodeGenerator
  - Impact: execute_goal pipeline is functional but output quality needs tuning

## Ready to Start

## Backlog

- [ ] **Token usage tracking & budget controls**
  - Add usage tracking to LLMService.generate_with_tools()
  - Create api_usage database table
  - Track tokens/cost by feature (chat, introspection, beliefs)
  - Implement daily/monthly budget limits
  - Add usage dashboard endpoint
  - See TODO_TOKEN_TRACKING.md for full implementation plan
  - Current spend: ~$24/month (healthy, but using estimates)

- [x] **Complete decision audit logging** (COMPLETE)
  - [x] Add logging to form_belief_from_pattern() - AdaptiveBeliefLifecycleManager
  - [x] Add logging to consider_promotion() - AdaptiveBeliefLifecycleManager
  - [x] Add logging to consider_deprecation() - AdaptiveBeliefLifecycleManager
  - [x] Add abort logging to abort_condition_monitor.py - decision_aborted_event()
  - [x] Add adaptation logging to parameter_adapter.py - parameter_adapted_event()

- [ ] **MCP integration (Phase 2A)**
  - Tooling + tests landed in `src/mcp/task_execution_server.py`
  - Install `modelcontextprotocol` in active runtime and smoke test
  - Add CLI/service entrypoint to expose the server in deployment
  - Spec remains in `.claude/tasks/prompt-008-mcp-task-execution.md`

- [ ] **Identity boundary definition**
  - Define what constitutes "identity" vs "state"
  - Implement boundary protections
  - Document identity evolution rules

- [ ] **Instant vs slow reasoning paths**
  - Fast path for simple queries
  - Slow path for complex reasoning
  - Automatic path selection

## Completed ✅

- [x] **Phase 3 Production Fixes (COMPLETE)** - Bug fixes and verification (2025-11-11)
  - Fixed logger initialization bug in `src/services/llm.py` (UnboundLocalError)
  - Added defensive None checks in `belief_consistency_checker.py`
  - Added warning to `/api/chat` endpoint redirecting to `/api/persona/chat`
  - Verified end-to-end execute_goal functionality in production
  - Confirmed Astra autonomously calls execute_goal and generates code
  - Full pipeline tested: Tool call → HTN planning → Code gen → File creation → Test execution
  - Documentation updated in `.claude/tasks/PHASE3-COMPLETE.md`
- [x] **Phase 1: GoalStore (COMPLETE)** - Goal prioritization with adaptive learning (2025-11-08)
  - GoalStore implementation (432 lines, 31 tests) - Value/effort/risk scoring, belief alignment
  - RESTful API endpoints (270 lines) - CRUD + prioritization + adoption safety checks
  - Decision point registration - goal_selected wired to ParameterAdapter for weight learning
  - TaskScheduler integration (150+ lines, 13 integration tests) - Goal-to-task conversion, autonomous selection
  - Comprehensive usage guide - Examples, patterns, best practices
  - **Total**: 850+ lines of production code, 44 passing tests, full API coverage + autonomous execution
- [x] **Phase 2: TaskGraph + Executor (COMPLETE)** - Production-ready task execution system (2025-11-08)
  - **Week 1**: TaskGraph (681 lines, 29 tests) - Dependency tracking, cycle detection, priority scheduling
  - **Week 2**: TaskExecutor (540 lines, 26 tests) - Retry logic, idempotency, safety checks
  - **Week 3**: TaskScheduler integration (10/11 integration tests) - Parallel execution, dependency tracking
  - **Week 4**: Documentation + regression tests - 65 total tests passing, comprehensive docs
  - **Total**: 1,220+ lines of production code, 65 passing tests, 91% integration test coverage
- [x] **Phase 3: HTN Planner (COMPLETE)** - Hierarchical task decomposition for complex goals (2025-11-09)
  - HTN Planner core (474 lines, 24 unit tests) - Method decomposition, cost-based selection, precondition checking
  - TaskGraph integration (plan_to_task_graph) - Converts HTN plans to executable graphs
  - Full pipeline integration (12 integration tests) - GoalStore → HTN Planner → TaskGraph
  - Decision point registration - plan_generated wired to ParameterAdapter for method learning
  - **Total**: 474 lines of production code, 36 passing tests, 90% architecture complete
- [x] **Adaptive Decision Framework (Phases 1-4)** - Complete implementation
- [x] **Adaptive Framework E2E Testing (COMPLETE)** - Production validation (2025-11-09)
  - 12 end-to-end integration tests covering full decision loop
  - Verified decision recording (goal_selected, plan_generated)
  - Verified DecisionRegistry persistence
  - Verified ParameterAdapter integration
  - Verified default weights and framework enabled flag
  - **Total**: 12 passing tests validating ~3,700 lines of framework code
- [x] **End-to-end task tracking for auditability (Phase 1)** - Full correlation system
- [x] **Wire adaptive framework into app.py** - Integrated with feature flag
- [x] **HTTPS setup with self-signed certificates** - Working on port 8443

## Notes

- Adaptive Decision Framework is fully implemented (~3,700 lines) and tested end-to-end
- Framework is enabled in production (DECISION_FRAMEWORK_ENABLED=true)
- All decision points registered: goal_selected, plan_generated
- Identity ledger has been enhanced to support decision framework events
- 12 integration tests validate full adaptive loop: record → evaluate → adapt
