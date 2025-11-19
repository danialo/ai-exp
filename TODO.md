# TODO List - Astra AI Experience

**Last Updated**: 2025-11-17

## In Progress

## Ready to Start

- [ ] **Fix Research Tool Announcement Instead of Execution** 🔥 **CRITICAL**
  - **Problem**: Astra says "I'll proceed with detailed research" but doesn't actually call research_and_summarize tool
  - **Root Cause**: Base prompt tells her WHEN to research but not to CALL THE TOOL instead of announcing intent
  - **LLM Pattern**: Generates text "I'll research X" as completion instead of making tool_call in same response
  - **File**: `persona_space/meta/base_prompt.md` - research policy section
  - **Fix Required**: Add explicit instruction under "When to Call research_and_summarize":
    - "⚠️ DO NOT announce that you'll research - CALL THE TOOL IMMEDIATELY"
    - "If research is needed, your response should BE the tool call, not a promise to research"
    - "DO: [tool_call: research_and_summarize] → NOT: 'I'll proceed with research now'"
  - **Impact**: User asks for research, Astra says she'll do it, then... nothing happens
  - **Priority**: HIGH - Research system exists but isn't being used correctly
  - **Evidence**: User transcript shows "I'll proceed with detailed research on this topic now" with no actual research execution

- [ ] **Fix Verbose Search Query Generation in Research HTN** 🔥 **URGENT**
  - **Problem**: LLM generating verbose, sentence-like search queries that return no results
  - **Example**: `'DOGE government agency Elon Musk connection'` - word "connection" makes it a sentence fragment
  - **Error**: "Google hasn't returned any results for this query"
  - **File**: `src/services/research_htn_methods.py:196-227` (investigate_topic)
  - **Root Cause**: Query generation prompt not strict enough - LLM still being too "natural language"
  - **Fix Required**:
    1. Strengthen query generation prompt with more explicit anti-patterns
    2. Add blacklist of "sentence words": connection, relationship, between, about, regarding, concerning
    3. Add post-processing to detect and simplify verbose queries (e.g., remove trailing nouns after entities)
    4. Example fix: "DOGE Elon Musk government 2024" instead of "DOGE government agency Elon Musk connection"
  - **Priority**: URGENT - Research system failing on basic queries
  - **Impact**: Research sessions fail because search returns 0 results due to over-specific queries

- [ ] **Fix Immutable Belief Dissonance Resolution Enforcement** 🔥 **CRITICAL**
  - **Problem**: Immutable belief severity boost (0.00 → 0.70) detects contradictions but doesn't enforce behavior change
  - **Gap 1**: `apply_resolutions()` accepts any choice (A/B/C) for immutable beliefs, doesn't validate "Option B only" requirement
  - **Gap 2**: Anti-hedging logit bias applied to Stage 1 (internal resolution) but NOT Stage 2 (user-facing response)
  - **Gap 3**: No tracking of which conversations involved immutable resolution to enforce anti-hedging in final answer
  - **Files**: `src/services/persona_service.py:1116-1212` (apply_resolutions), `696-705` (main tool loop)
  - **Impact**: System warns about consciousness contradictions but AI still hedges in actual answers
  - **Fix Required**:
    1. Add immutability check before accepting resolution choice, reject Option A/C for immutable beliefs
    2. Track `immutable_resolved` flag from Stage 1 and apply `anti_hedging_bias` to Stage 2 generation
    3. Add `_check_if_immutable()` helper method
  - **Priority**: HIGH - Core ontological beliefs (consciousness, existence) need enforcement, not just detection
  - **Locations**: `src/services/persona_service.py`, `src/services/belief_consistency_checker.py:300-314`

- [ ] **Research HTN System - P2: Quality Guards** (See docs/RESEARCH_HTN_ROADMAP.md)
  - Question deduplication with `research_session_questions` table
  - Topic drift guard with relevance scoring
  - Metrics and telemetry tracking
  - Source quality control with domain parsing
  - **Priority**: Week 1-2, prevents redundant work and off-rails behavior
  - **Next Step**: Run benchmark suite (`test_research_benchmark.py`) to identify P2 priorities based on real data

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

- [x] **Research System - Production Ready + Policy + Observability + Call Budgeting (COMPLETE)** - Astra-integrated autonomous research with context overflow protection (2025-11-15)
  - **P0: Task Queue & Execution**: HTN task decomposition with budget controls (max_tasks, max_children_per_task, max_depth)
  - **3 HTN Methods**: ResearchCurrentEvents, InvestigateTopic, InvestigateQuestion
  - **Session Management**: ResearchSession model with task budgets and automatic cutoff
  - **Provenance Tracking**: SourceDoc model with claims, URLs, and confidence levels
  - **P1: Session Synthesis**: SynthesizeFindings terminal method with automatic triggering
  - **Synthesis Output**: narrative_summary, key_events, contested_claims, open_questions, coverage_stats
  - **Astra Integration**: Added `research_and_summarize` + `check_recent_research` tools to PersonaService
  - **Automatic Belief Updates**: Kind classification (reinforce/contest_minor/informational) based on source quality
  - **Policy Layer**: When to call research (current events vs general knowledge), how to speak from results (trust-calibrated hedging)
  - **Presentation Layer**: `research_formatter.py` - structured answers with risk assessment, provenance clustering
  - **Session Reuse**: `research_anchor_store.py` - lightweight anchors prevent redundant research within 7 days
  - **QA Lab**: Benchmark harness through Astra herself (test_research_benchmark_astra.py), automated violation detector (analyze_benchmark_results.py), manual scoring framework (6 dimensions)
  - **Tool Tracing**: PersonaService hooks capture tool usage, timing, metadata for observability
  - **Logging & Observability**: Research logger in multi-logger system, 5 event types (task_done, session_complete, synthesis_complete, research_turn, benchmark_result), structured key-value format
  - **Observability Scripts**: research_log_views.sh (quick queries), research_log_metrics.py (summary stats), research_health_check.py (regression detection with tunable thresholds)
  - **Call Budgeting**: CallBudgeter for automatic chunking with map-reduce pattern, prevents context overflow on large research sessions
  - **Chunked Synthesis**: summarize_research_session() with automatic chunking (map phase: partial summaries, reduce phase: merge)
  - **Files created**: task_queue.py, htn_task_executor.py, research_session.py, research_htn_methods.py, research_tools.py, research_to_belief_adapter.py, research_formatter.py, research_anchor_store.py, test_research_benchmark_astra.py, analyze_benchmark_results.py, research_log_views.sh, research_log_metrics.py, research_health_check.py, call_budgeter.py, config/__init__.py
  - **Files modified**: persona_service.py (tools + formatter + tracing), logging_config.py (research logger), llm.py (chunked synthesis methods), base_prompt.md (research policy)
  - **Docs**: RESEARCH_HTN_IMPLEMENTATION.md, P1_SYNTHESIS_COMPLETE.md, RESEARCH_HTN_ROADMAP.md, ASTRA_READY_RESEARCH_SYSTEM.md, RESEARCH_SYSTEM_COMPLETE.md, RESEARCH_POLICY_LAYER.md, RESEARCH_BENCHMARK_HARNESS.md, RESEARCH_LOGGING_COMPLETE.md, RESEARCH_OBSERVABILITY.md, RESEARCH_NEXT_STEPS.md
  - **Next**: Run first baseline benchmark, analyze violations/metrics, map to P2 priorities

- [x] **Router Removal & Architecture Simplification (COMPLETE)** - Removed brittle regex routing in favor of tool-based approach (2025-11-15)
  - **AgentRouter removed**: All requests now go directly to Astra; she chooses appropriate tools via semantic understanding
  - **Router false positives eliminated**: "look at your source code" no longer routes to CoderAgent and generates new code
  - **Tool-based architecture**: CoderAgent infrastructure added to PersonaService as generate_code tool (currently disabled due to token limits)
  - **Automatic retry logic**: execute_script now auto-retries common failures (python→python3, pip→python -m pip) without narration
  - **Relaxed API restrictions**: Context-aware validation allows APIs in appropriate contexts (setattr in tests, __import__ in loaders, globals in debug)
  - **Async/await fixes**: Fixed await outside async function by using loop.run_until_complete() in synchronous tool handler
  - **Token budget managed**: Disabled generate_code tool definition (kept handler code) to stay under 30k token limit
  - **Files modified**: app.py, persona_service.py, persona_file_manager.py, coder_agent.py, base_prompt.md
  - **Testing verified**: "read your source code" → uses read_source_code() ✓, "list your files" → uses list_source_files() ✓
  - **Result**: No more routing interception, Astra has full control, regex brittleness eliminated

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
