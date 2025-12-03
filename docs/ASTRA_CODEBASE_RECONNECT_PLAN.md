# Astra Codebase Reconnect Plan

**Last Updated:** 2025-12-02

## 1. Objectives
- Re-establish Astra's ability to inspect and ultimately modify her source tree without regressing on prior safety incidents.
- Ship capability in two gated phases:
  1. **Phase 1 – View (Complete)**: allow `list_source_files`/`read_source_code` so Astra can regain context.
  2. **Phase 2 – Write (Planned)**: route all code changes through guarded workflows, tests, and approvals before writes hit the repo.

## 2. Current Access Controls
### 2.1 Feature Flags & Runtime Switches
- `PERSONA_MODE_ENABLED` (`config/settings.py:103`) gates the entire persona stack; view/write tools are unavailable when `false`.
- Persona-only LLM instances are built inside `app.py:658-724`, and `code_access_service`/`task_scheduler` are only instantiated when persona mode is on (`app.py:345-410`, `app.py:691-715`).

### 2.2 File-System Boundaries
- `PersonaFileManager` enforces sandboxing for self-owned files and exposes read-only `src/` mirroring (`src/services/persona_file_manager.py:22-105`).
  - `_resolve_path` explicitly rejects any attempt to write into `src/` or outside `persona_space` (`src/services/persona_file_manager.py:264-308`).
- Conversation tools `list_source_files` + `read_source_code` live inside `PersonaService` and funnel through the file manager (`src/services/persona_service.py:1406-1494`).
- `CoderAgent`’s safety prompt forbids `open(..., 'w')`, subprocess, sockets, etc., reinforcing read-only operation in LLM completions (`src/agents/prompts/coder_v1_0.txt:1-86`).

### 2.3 Write Surfaces (Currently Disabled)
- `CodeAccessService` is wired but guarded by allowed/forbidden path lists and auto-branching (`src/services/code_access.py:59-201`).
- Persona tools that lead to writes (`schedule_code_modification`, `execute_goal`) check for `code_access_service` availability and enforce manual approval or HTN execution pipelines (`src/services/persona_service.py:1612-1703`, `src/services/persona_service.py:2110-2208`). These remain dormant because we have not re-enabled write approval flow.
- `CodeModificationExecutor` ensures every `TaskGraph` write runs through `CodeAccessService` admission checks and optional syntax validation (`src/services/task_executors/code_modification.py:14-177`).

### 2.4 Telemetry & Audit Hooks
- Every scheduled code modification appends a ledger event for forensic tracking (`src/services/persona_service.py:2150-2169` + `src/services/identity_ledger.py:1-150`).
- `CodeAccessService` logs each read/write attempt and tracks modification metadata for subsequent approvals (`src/services/code_access.py:111-186`).
- Task scheduler persists CODE_MODIFY tasks to disk before execution (`src/services/task_scheduler.py`, see tool wiring at `persona_service.py:2110-2190`).

## 3. Phase 1 – View Access (Status: ✅ Restored 2025-12-02)
### 3.1 Activation Steps
1. Set `PERSONA_MODE_ENABLED=true` so persona services initialize (already applied in runtime env).
2. Ensure `code_access_service` is instantiated but only used for read-only surfaces (`app.py:345-410`).
3. Confirm Persona tools `list_source_files` and `read_source_code` render in the tool schema (`src/services/persona_service.py:1406-1494`).
4. Verify `PersonaFileManager` resolves `src/...` paths in read-only mode and denies write intents (`src/services/persona_file_manager.py:22-118`, `264-308`).

### 3.2 Verification Checklist
- Run `pytest tests/test_code_access_conversation_tools.py::TestReadSourceCode` to assert the tool plumbing works.
- Manual smoke:
  - Call `list_source_files` followed by `read_source_code` to ensure real repo visibility and friendly errors when adding `src/` prefix.
  - Attempt persona-space writes to confirm they remain scoped (`PersonaFileManager.write_file`).
- Observe `app.log` for `code_access_service: True` and absence of `modify_file` invocations.

### 3.3 Observability During Phase 1
- Monitor identity ledger for any unexpected `code_modification_*` events (should stay empty).
- Keep standard HTTP/API telemetry on to observe new `read_source_code` usage volume; consider adding a lightweight counter if needed (not critical before Phase 2).

## 4. Phase 2 – Write Access (Planned)
### 4.1 Prerequisite Guardrails
1. **Config Flagging**: introduce `ASTRA_CODE_WRITE_ENABLED` (default `false`). Gate `schedule_code_modification`, `execute_goal`, and Goal Execution Service initialization on this flag so we can flip writes atomically at runtime.
2. **Persona Prompting**: add explicit instructions reminding Astra to keep repo writes behind `schedule_code_modification`/`execute_goal` when the flag is disabled and to narrate when she requests approvals.
3. **Task Scheduler Policies**: ensure CODE_MODIFY tasks remain `MANUAL` until an operator approves; prevent unattended execution unless audits pass.
4. **Identity Ledger Coverage**: extend ledger events to include `code_modification_executed`, `tests_started`, `tests_failed`, and `tests_passed` for end-to-end traceability.
5. **HTN Research Alignment**: Astra already runs HTN planning for research (multi-step `research_and_summarize` flows). When we enable writes, ensure those research HTN plans can feed directly into code-mod goals without bypassing new guardrails—e.g., require that any research-driven change still schedules a CODE_MODIFY task and references the research session ID for auditing.
6. **MCP Enforcement Option**: Explore routing code-write approvals through the already-hardened MCP server (Tier 0/1 tooling). MCP’s stdio transport plus tool budget tracking could provide an externalized approval surface—e.g., an MCP tool that surfaces pending CODE_MODIFY tasks, captures human approval, and triggers execution via the scheduler.

### 4.2 Testing & Regression Matrix
| Layer | Test/Script | Purpose |
| --- | --- | --- |
| Unit | `pytest tests/test_code_access.py` | Validates boundary checks, file size guard, modification tracking. |
| Unit | `pytest tests/test_code_access_conversation_tools.py` | Confirms persona tools enforce access and scheduler wiring. |
| Integration | `pytest tests/integration/test_phase1_execution_engine.py` + `test_phase2_goal_execution_service.py` | Exercises `CodeModificationExecutor` & Goal Execution Service with mocked repo. |
| Research HTN | `pytest src/test_research_benchmark_astra.py -k htn` (or `pytest -m research_htn`) | Validates multi-step research planner so research-informed code goals behave before/after write flag flips. |
| System | `tests/regression_quick.sh` (while server running) | Ensures chat endpoints, persona loop, and logging stay healthy post-flag flip. |
| Targeted | Manual dry-run of `execute_goal` → `TaskGraph` → `CodeAccessService.modify_file` on a throwaway branch; verify branch isolation and diffs. |

All tests must pass twice: once with `ASTRA_CODE_WRITE_ENABLED=false` (guard ensures no writes) and once with it `true` in a sandboxed repo clone.

### 4.3 Implementation Tasks
1. **Config + Settings**
   - Add `ASTRA_CODE_WRITE_ENABLED` to `config/settings.py` with default `false` and wire it into `app.py` so we only register write tools and `GoalExecutionService` when enabled.
2. **Persona Tool Gating**
   - Update `_get_tool_definitions` in `PersonaService` to hide `schedule_code_modification`/`execute_goal` unless the new flag is on; return a friendly advisory otherwise.
   - Add runtime guardrails inside `_execute_tool` to short-circuit requests when disabled.
3. **Telemetry Enhancements**
   - Extend identity ledger events for code modifications (schedule, approval, execution, rollback) and emit `tests_started/tests_completed` events from `GoalExecutionService`.
   - Add diff + branch metadata to logs for forensic review.
4. **Approval Workflow**
   - Ensure `TaskScheduler` exposes a CLI/API (or MCP tool) to approve `CODE_MODIFY` tasks; the plan is to keep `schedule=MANUAL` until an operator toggles `enabled` and triggers execution.
   - Document the approval handshake (who reviews diffs, what logs to inspect) in `docs/OPS_CODE_WRITE.md` (to be created alongside implementation).
5. **Monitoring + Alerts**
   - Add counters/metrics for: scheduled modifications, approvals, failed tests, rollbacks. Hook into existing logging or Prometheus exporters if available.
6. **Dry-Run & Launch**
   - Use a disposable branch to run through `schedule_code_modification` → manual approval → `CodeModificationExecutor` → tests → diff review. Capture transcripts/screenshots for future incidents.
   - Only after dry-run success and regression matrix sign-off do we set `ASTRA_CODE_WRITE_ENABLED=true` in prod.

### 4.4 Go/No-Go Checklist
- ✅ Flag disabled: confirm persona gracefully refuses write tools.
- ✅ All listed tests + regression script pass while flag is off (baseline).
- ✅ Dry-run with flag on completes without `code_access` or HTN errors; identity ledger contains the full trail.
- ✅ Monitoring dashboards show expected counters and no new alerts.
- ✅ Operator approval process documented and rehearsed.

## 5. Deliverables & Owners
- **Docs**: This plan plus follow-up `OPS_CODE_WRITE.md` detailing approval steps. Owner: Safety/Infra.
- **Implementation**: Platform team to add flag gating, telemetry, and scheduler updates.
- **Verification**: QA to run regression matrix + manual dry-run; owner signs off in TODO + CHANGELOG entries.

Once the checklist is satisfied, update `TODO.md` Phase 2 entry with the evidence summary and flip the new flag to enable writes.
