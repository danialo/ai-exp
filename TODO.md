# TODO List - Astra AI Experience

**Last Updated**: 2025-12-08

## In Progress

## Ready to Start

- [ ] **Fix Astra's Incorrect Beliefs About Source Code Access** 🐛 **BUG**
  - **Problem**: Astra has contradictory beliefs about source code access from the backfill:
    - ❌ "i am currently unable to directly access my source code"
    - ❌ "i am unable to modify my source code myself"
    - ✅ "i can access my source code files through my tools"
    - ✅ "i can take a look at my source code"
  - **Reality**: She HAS `read_source_code` and `list_source_files` tools - she CAN access her code
  - **Fix Options**:
    1. Delete the incorrect belief nodes from `belief_nodes` table
    2. Update her base prompt to clarify her actual capabilities
    3. Add a "capability correction" experience to override the false beliefs
  - **Priority**: HIGH - Core self-knowledge should be accurate


- [ ] **Research HTN System - P2: Quality Guards** (See docs/RESEARCH_HTN_ROADMAP.md)
  - Question deduplication with `research_session_questions` table
  - Topic drift guard with relevance scoring
  - Metrics and telemetry tracking
  - Source quality control with domain parsing
  - **Priority**: Week 1-2, prevents redundant work and off-rails behavior
  - **Next Step**: Run benchmark suite (`test_research_benchmark.py`) to identify P2 priorities based on real data

- [ ] **Astra File Activity Attribution & Isolation** 🔧 **ARCHITECTURE**
  - **Problem**: Cannot distinguish Astra's file operations from user/system changes at filesystem level
  - **Current State**: All files owned by user `d`, actions_log.json only tracks tool calls, not comprehensive
  - **Issues**:
    1. persona_space/ mixing identity (beliefs, reflections) with execution artifacts (venvs, test files)
    2. No way to audit "who created this file" - Astra vs user vs Claude Code vs system init
    3. File operations not visible in standard filesystem tools (ls, git status)
  - **Proposed Solutions**:
    1. **Dedicated User Account**: Create `astra` user, run persona operations as that user
       - Pros: Clear filesystem ownership, standard Unix permissions
       - Cons: Complexity in service setup, may need sudo for user switching
    2. **Dedicated Service/Container**: Run Astra operations in isolated service
       - Pros: Clear boundary, can log all I/O, easier to sandbox
       - Cons: More complex architecture, networking/IPC overhead
    3. **Enhanced Logging**: Expand actions_log.json to capture ALL file operations
       - Pros: Simple, no architecture change
       - Cons: Doesn't solve filesystem attribution, easy to bypass
    4. **Separate Workspace**: Move execution artifacts to dedicated workspace/ or leverage existing astra-workspace/
       - Pros: Clean separation of identity vs execution
       - Cons: Doesn't solve attribution, just organization
  - **Files Affected**:
    - `src/services/persona_file_manager.py` - file operation handlers
    - `src/services/persona_service.py` - tool execution
    - `persona_space/` - directory structure and gitignore rules
  - **Priority**: MEDIUM - Affects observability, debugging, and system clarity
  - **Impact**: Better attribution, cleaner persona_space/, easier debugging of "who made this change"
  - **Related**: persona_space cleanup (old October files, duplicate venvs, test artifacts)

## Backlog

- [ ] **Fix Astra's Ability to See and Add to Her Own Code** 🔧 **ARCHITECTURE**
  - **Problem**: Astra cannot view or modify her own codebase effectively
  - **Current State**: Has `read_source_code` and `list_source_files` tools but limited capability
  - **Needed**:
    1. Better code navigation - understand file relationships and dependencies
    2. Ability to propose code changes (not just read)
    3. Safe sandbox for testing self-modifications before applying
    4. Version control awareness - see what changed, create branches
  - **Safety Considerations**:
    - Changes should be reviewed before applying
    - Rollback capability required
    - Core identity files (beliefs, pledges) should have extra protection
  - **Priority**: MEDIUM - Key for self-improvement and debugging capabilities

- [ ] **Wire Up Memory Pruning System** 🔧 **MAINTENANCE**
  - **Problem**: Memory grows indefinitely - nothing is pruned or decayed
  - **Current State**:
    - 6,357 experiences (19.9 MB), growing ~130-180/day
    - `MemoryDecayCalculator` exists but `recalculate_all_decay()` never called
    - `MemoryPruner` exists but not wired into `IntegrationLayer`
    - 96.5% of experiences have no decay metrics
    - All `decay_factor` values stuck at 1.0 (no decay applied)
    - All `access_count` values at 0 (never tracked)
  - **Fix Required**:
    1. Wire `MemoryPruner` into `IntegrationLayer` in app.py
    2. Schedule periodic `recalculate_all_decay()` runs (e.g., daily)
    3. Call `record_access()` when experiences are retrieved
    4. Initialize decay metrics for existing experiences
  - **Projected Growth (no fix)**: ~73,000 experiences in 1 year
  - **Priority**: MEDIUM - Not urgent but will become problem at scale

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

- [x] **Fix Immutable Belief Dissonance Resolution Enforcement (COMPLETE)** - 2025-12-08
  - Added immutability validation in `apply_resolutions()` - Options A/C now rejected for immutable beliefs
  - Added `_immutable_dissonance_active` flag to track when anti-hedging enforcement is needed
  - Anti-hedging bias now merged with logit_bias when flag is active (enforces committed responses)
  - Flag cleared after response generation to affect only the immediate next response
  - Files: `persona_service.py` (apply_resolutions, process_with_persona), `app.py` (resolution logging)
  - Gap 1 ✅: Immutable beliefs now require Option B only
  - Gap 2 ✅: Anti-hedging bias applied to Stage 2 via `_immutable_dissonance_active` flag
  - Gap 3 ✅: `immutable_resolved` flag tracks immutable resolutions, enables anti-hedging

- [x] **Wire Full VAD (Arousal + Dominance) into AgentMood (COMPLETE)** - 2025-12-08
  - AgentMood now tracks arousal and dominance alongside valence
  - `detect_vad()` called in app.py, `record_vad()` method added
  - Richer emotional state tracking for tone adaptation

- [x] **ResearchGate: Deterministic Research Decision Layer (COMPLETE)** - 2025-12-03
  - Fixed "announces research instead of executing" by moving decision out of model prose generation
  - ResearchGate class with fast heuristics + fallback LLM classifier
  - Integrated into PersonaService._is_research_query()
  - Tests for research orchestration
  - Files: research_gate.py (393 lines), test_research_orchestration.py (262 lines)

- [x] **Research Query Optimization (COMPLETE)** - 2025-12-03
  - Fixed verbose search query generation
  - Query scoring, fallback, and telemetry
  - research_query_utils.py, research_query_scoring.py, research_query_fallback.py
  - 16 unit tests passing
  - Fixed research anchor dict binding error

- [x] **Awareness Loop Context Budget Increase (COMPLETE)** - 2025-12-03
  - max_context_tokens: 1000 → 16000 (50% of 32k input limit)
  - max_tokens reply: 300 → 4000
  - buf_win: 32 → 64, mem_k: 5 → 30
  - Enables richer self-reflection with meaningful context

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
