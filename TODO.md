# TODO List - Astra AI Experience

**Last Updated**: 2025-11-14

## In Progress

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

- [ ] **Identity boundary definition**
  - Define what constitutes "identity" vs "state"
  - Implement boundary protections
  - Document identity evolution rules

- [ ] **Instant vs slow reasoning paths**
  - Fast path for simple queries
  - Slow path for complex reasoning
  - Automatic path selection

## Completed ✅

- [x] **Belief System & Agent Router Fixes (COMPLETE)** - Critical bug fixes for coherence and routing (2025-11-14)
  - **Belief reconciliation infinite loop fixed**: Meta-disclaimer filter blocks LLM safety responses ("I do not possess consciousness") from being treated as self-claims
  - **120-minute cooldown**: Prevents repeated dissonance events for same belief (was triggering every 3 minutes)
  - **Charitable interpretation**: Enhanced prompts recognize implicit consciousness claims ("I'm here" → presence = consciousness)
  - **Sensitive log separation**: Created `meta_cognitive/` directory for dissonance analysis and memory rewrites (not accessible via read_logs tool)
  - **Agent router improvements**: Word boundary matching prevents false positives ("building" no longer matches "build")
  - **Capability question routing**: Questions like "Can you see your source code?" now route to Astra instead of CoderAgent
  - **Codebase access tools**: Added `list_source_files()` tool for file discovery, improved error messages
  - **HTTP 500 fixes**: Fixed async/await bugs in router.py and coder_agent.py, UnboundLocalError in app.py
  - **Files modified**: belief_consistency_checker.py, persona_service.py, router.py, coder_agent.py, app.py
  - **Result**: Dissonance events reduced from every 3min to max once per 2 hours, memory rewrites dramatically reduced, coherence protected

- [x] **Astra's Autonomous Workspace (COMPLETE)** - Dedicated workspace with CodeAgent-5 prompts (2025-11-12)
  - **Workspace structure**: /home/d/astra-workspace/ with projects/, templates/, logs/
  - **ProjectManager service** (344 lines): Creates/manages timestamped projects with metadata
  - **CodeAgent-5 prompts**: Structured prompt template with ROLE, CONTEXT, REQUIREMENTS, QUALITY_GUARDS sections
  - **Workspace integration**: execute_goal now creates projects in workspace instead of tests/generated/
  - **Project lifecycle**: created → in_progress → completed/failed with full metadata tracking
  - **Improved code quality**: Prompts now include codebase context, evaluation criteria, security guards
  - **Testing**: Verified project creation and metadata tracking work correctly
  - Addresses: "LLM generates generic test code instead of specific implementations"
- [x] **MCP Autonomous Scheduling (COMPLETE)** - Full MCP server with scheduling and introspection (2025-11-12)
  - **9 MCP tools**: Task introspection (tasks_list, tasks_by_trace, tasks_last_failed, astra.health), scheduling (create/modify/pause/resume/list), desires (record/list/reinforce)
  - **ScheduleService** (565 lines): NDJSON+index persistence, cron scheduling with croniter, 3-tier safety model, per-day budget enforcement
  - **DesireStore** (349 lines): Vague wish tracking with strength decay, deterministic IDs, automatic reinforcement
  - **MCP tools** (507 lines): Complete tool handlers for schedule and desire management
  - **67 tests passing**: ScheduleService (27 tests), DesireStore (26 tests), MCP tools (14 tests)
  - **Comprehensive documentation** (2,080+ lines across 11 files): Quick start, complete guide, API reference, architecture deep dive
  - **Stdio transport**: bin/mcp wrapper script, Claude Desktop integration, on-demand startup pattern
  - **Production-ready**: Tier 0 (read-only) and Tier 1 (local writes with budgets) fully operational
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
- [x] **Complete decision audit logging (COMPLETE)**
  - Added logging to form_belief_from_pattern() - AdaptiveBeliefLifecycleManager
  - Added logging to consider_promotion() - AdaptiveBeliefLifecycleManager
  - Added logging to consider_deprecation() - AdaptiveBeliefLifecycleManager
  - Added abort logging to abort_condition_monitor.py - decision_aborted_event()
  - Added adaptation logging to parameter_adapter.py - parameter_adapted_event()

## Notes

- **MCP Server**: Production-ready with 9 tools, 67 tests passing, comprehensive documentation
  - Start with: `bin/mcp`
  - Configure Claude Desktop: Add command path to `~/.config/Claude/claude_desktop_config.json`
  - Documentation: `docs/MCP_QUICKSTART.md`, `docs/MCP_COMPLETE_GUIDE.md`
- Adaptive Decision Framework is fully implemented (~3,700 lines) and tested end-to-end
- Framework is enabled in production (DECISION_FRAMEWORK_ENABLED=true)
- All decision points registered: goal_selected, plan_generated
- Identity ledger has been enhanced to support decision framework events
- 12 integration tests validate full adaptive loop: record → evaluate → adapt
