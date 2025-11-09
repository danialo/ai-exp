# TODO List - Astra AI Experience

**Last Updated**: 2025-11-08

## In Progress

- No active work items

## Ready to Start

- [ ] **Test adaptive framework end-to-end**
  - Enable DECISION_FRAMEWORK_ENABLED in .env
  - Run system with framework active
  - Verify decisions are recorded
  - Verify outcomes are evaluated
  - Verify parameters adapt

## Backlog

- [ ] **Token usage tracking & budget controls**
  - Add usage tracking to LLMService.generate_with_tools()
  - Create api_usage database table
  - Track tokens/cost by feature (chat, introspection, beliefs)
  - Implement daily/monthly budget limits
  - Add usage dashboard endpoint
  - See TODO_TOKEN_TRACKING.md for full implementation plan
  - Current spend: ~$24/month (healthy, but using estimates)

- [ ] **Complete decision audit logging**
  - [x] Add logging to form_belief_from_pattern()
  - [ ] Add logging to consider_promotion()
  - [ ] Add logging to consider_deprecation()
  - [ ] Add abort logging to abort_condition_monitor.py
  - [ ] Add adaptation logging to parameter_adapter.py

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
- [x] **End-to-end task tracking for auditability (Phase 1)** - Full correlation system
- [x] **Wire adaptive framework into app.py** - Integrated with feature flag
- [x] **HTTPS setup with self-signed certificates** - Working on port 8443

## Notes

- Adaptive Decision Framework is fully implemented (~3,700 lines) but not yet tested in production
- All framework code is behind DECISION_FRAMEWORK_ENABLED flag (default: false)
- Identity ledger has been enhanced to support decision framework events
- Ready for end-to-end integration testing once audit logging is complete
