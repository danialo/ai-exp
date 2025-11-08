# TODO List - Astra AI Experience

**Last Updated**: 2025-11-06

## In Progress

- [ ] **Complete decision audit logging** (CURRENT)
  - [x] Add logging to form_belief_from_pattern()
  - [ ] Add logging to consider_promotion()
  - [ ] Add logging to consider_deprecation()
  - [ ] Add abort logging to abort_condition_monitor.py
  - [ ] Add adaptation logging to parameter_adapter.py

## Ready to Start

- [ ] **Test adaptive framework end-to-end**
  - Enable DECISION_FRAMEWORK_ENABLED in .env
  - Run system with framework active
  - Verify decisions are recorded
  - Verify outcomes are evaluated
  - Verify parameters adapt

## Backlog

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

- [x] **Adaptive Decision Framework (Phases 1-4)** - Complete implementation
- [x] **End-to-end task tracking for auditability (Phase 1)** - Full correlation system
- [x] **Wire adaptive framework into app.py** - Integrated with feature flag
- [x] **HTTPS setup with self-signed certificates** - Working on port 8443

## Notes

- Adaptive Decision Framework is fully implemented (~3,700 lines) but not yet tested in production
- All framework code is behind DECISION_FRAMEWORK_ENABLED flag (default: false)
- Identity ledger has been enhanced to support decision framework events
- Ready for end-to-end integration testing once audit logging is complete
