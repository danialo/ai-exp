# Research System Logging - Complete

**Date**: 2025-11-15
**Status**: ✅ Fully integrated

## Overview

Complete logging integration for the research system across all layers: core runtime (HTN executor), tool tracing (PersonaService), and benchmark harness.

---

## Logging Architecture

### Logger Location
- **File**: `logs/research/research_system.log`
- **Logger name**: `astra.research`
- **Max size**: 20MB rotating
- **Backup count**: 5

### Log Format
All research logs use structured key-value format for easy parsing:
```
session={session_id} event={event_type} key1=value1 key2=value2 ...
```

---

## Layer 1: Core Runtime Logs

**What**: HTN task execution and session lifecycle

**Where**: `src/services/htn_task_executor.py`

### Events Logged

#### 1. Task Completion
**Event**: `task_done`
**When**: After each HTN task completes
**Data**:
- `task_id`: Unique task ID
- `type`: HTN task type (ResearchCurrentEvents, InvestigateTopic, etc.)
- `depth`: Task depth in HTN tree
- `children`: Number of child tasks enqueued

**Example**:
```
session=abc-123 event=task_done task_id=xyz-789 type=InvestigateTopic depth=1 children=3
```

#### 2. Session Completion
**Event**: `session_complete`
**When**: When research session finishes (budget exhausted or no tasks remain)
**Data**:
- `tasks_created`: Total tasks created
- `max_tasks`: Budget limit
- `budget_remaining`: Unused task slots

**Example**:
```
session=abc-123 event=session_complete tasks_created=28 max_tasks=30 budget_remaining=2
```

---

## Layer 2: Synthesis Logs

**What**: Session synthesis and aggregation

**Where**: `src/services/research_htn_methods.py` (SynthesizeFindings method)

### Events Logged

#### 3. Synthesis Completion
**Event**: `synthesis_complete`
**When**: After global narrative synthesis completes
**Data**:
- `docs`: Number of source documents processed
- `claims`: Total claims extracted
- `key_events`: Number of key events identified
- `contested_claims`: Number of contested claims found
- `open_questions`: Number of open questions generated

**Example**:
```
session=abc-123 event=synthesis_complete docs=12 claims=47 key_events=5 contested_claims=2 open_questions=3
```

---

## Layer 3: Tool Trace Logs (Per-Turn)

**What**: Research tool usage during user interactions

**Where**: `src/services/persona_service.py` (generate_response method)

### Events Logged

#### 4. Research Turn
**Event**: `research_turn`
**When**: After any turn where research tools were called
**Data**:
- `question`: User question (truncated to 120 chars, repr'd)
- `tools`: List of research tools called

**Example**:
```
session=N/A event=research_turn question='What is actually going on with the Epstein files story?' tools=['check_recent_research', 'research_and_summarize']
```

**Note**: `session=N/A` because this logs the outer persona turn, not a specific research session

---

## Layer 4: Benchmark Logs

**What**: Benchmark test results

**Where**: `src/test_research_benchmark_astra.py` (ask_astra function)

### Events Logged

#### 5. Benchmark Result
**Event**: `benchmark_result`
**When**: After each benchmark question completes
**Data**:
- `q`: Question (truncated to 80 chars, repr'd)
- `elapsed`: Total elapsed time in seconds
- `tools`: List of all tools called
- `risk`: Research risk level (high/medium/low)

**Example**:
```
benchmark_result q='What is actually going on with the Epstein files story?' elapsed=47.12 tools=['check_recent_research', 'research_and_summarize'] risk=high
```

---

## Files Modified

### 1. `src/utils/logging_config.py`
**Added**:
- Research logger to MultiFileLogger (lines 121-129)
- `log_research_event()` convenience method (lines 204-211)

**New method signature**:
```python
def log_research_event(self, event_type: str, session_id: str, data: Optional[dict] = None):
    """Log research system events."""
    logger = self.loggers['research']
    msg = f"session={session_id} event={event_type}"
    if data:
        for k, v in data.items():
            msg += f" {k}={v}"
    logger.info(msg)
```

### 2. `src/services/htn_task_executor.py`
**Added**:
- Import `get_multi_logger` (line 12)
- Task completion logging (lines 141-150)
- Session completion logging (lines 205-213)

### 3. `src/services/research_htn_methods.py`
**Added**:
- Import `get_multi_logger` (line 6)
- Synthesis completion logging (lines 257-267)

### 4. `src/services/persona_service.py`
**Added**:
- Research turn logging at end of generate_response (lines 633-645)

### 5. `src/test_research_benchmark_astra.py`
**Added**:
- Import `get_multi_logger` (line 26)
- Fixed response tuple unpacking (line 64)
- Benchmark result logging (lines 94-100)

---

## Usage Examples

### Reading Research Logs

```bash
# All research events
tail -f logs/research/research_system.log

# Filter by event type
grep "event=synthesis_complete" logs/research/research_system.log

# Filter by session
grep "session=abc-123" logs/research/research_system.log

# Benchmark results only
grep "benchmark_result" logs/research/research_system.log

# High-risk research sessions
grep "risk=high" logs/research/research_system.log
```

### Analyzing Research Performance

```bash
# Count tasks per session
grep "event=task_done" logs/research/research_system.log | awk '{print $2}' | sort | uniq -c

# Average synthesis doc count
grep "event=synthesis_complete" logs/research/research_system.log | grep -oP 'docs=\K[0-9]+' | awk '{sum+=$1; count++} END {print sum/count}'

# Tool usage patterns
grep "event=research_turn" logs/research/research_system.log | grep -oP "tools=\[.*?\]" | sort | uniq -c
```

---

## What This Enables

### 1. Performance Monitoring
- Session execution time (via task timestamps)
- Synthesis throughput (docs processed per session)
- Tool call frequency

### 2. Policy Compliance Tracking
- Tool ordering (check_recent_research before research_and_summarize)
- Risk calibration (risk level assignments)
- Budget adherence (tasks created vs max_tasks)

### 3. Quality Metrics
- Source diversity (docs per session)
- Claim density (claims per doc)
- Contested claim rate
- Open question generation

### 4. Debugging
- Task decomposition paths (depth tracking)
- Session completion triggers
- Tool call sequences
- Error patterns (via standard logger.error)

---

## Integration with Existing Logging

This research logger integrates seamlessly with the existing multi-logger system:

**Other loggers** (already in place):
1. `astra.app` → `logs/app/application.log` (general system events)
2. `astra.conversation` → `logs/conversations/conversations.log` (user-Astra chat)
3. `astra.error` → `logs/errors/errors.log` (errors from all systems)
4. `astra.tools` → `logs/tools/tool_calls.log` (all tool calls)
5. `astra.memory` → `logs/memory/memory_retrieval.log` (memory searches)
6. `astra.beliefs` → `logs/beliefs/belief_system.log` (belief updates)
7. `astra.awareness` → `logs/awareness/awareness_loop.log` (introspection)
8. `astra.performance` → `logs/performance/performance.log` (timing/costs)

**New logger**:
9. `astra.research` → `logs/research/research_system.log` (research subsystem)

All loggers share the same format, rotation policy, and convenience method pattern.

---

## Next Steps

With logging complete, you can now:

1. **Run the benchmark** and watch logs in real-time:
   ```bash
   tail -f logs/research/research_system.log &
   source venv/bin/activate
   cd /home/d/git/ai-exp
   python3 src/test_research_benchmark_astra.py
   ```

2. **Analyze violations** using both automated detector and logs:
   ```bash
   python -m src.analyze_benchmark_results data/research_benchmark_astra_results_*.json
   grep "risk=high" logs/research/research_system.log
   ```

3. **Build monitoring dashboards** from structured logs:
   - Parse key-value format
   - Track metrics over time
   - Alert on anomalies

4. **Debug policy issues** by tracing full execution:
   - Follow session lifecycle from start to synthesis
   - Verify tool ordering
   - Check budget enforcement

---

## Bottom Line

**Status**: Logging fully integrated across all research stack layers

**What's logged**:
- ✅ HTN task execution (task_done events)
- ✅ Session lifecycle (session_complete events)
- ✅ Synthesis results (synthesis_complete events)
- ✅ Per-turn tool usage (research_turn events)
- ✅ Benchmark results (benchmark_result events)

**What's next**: Run baseline benchmark and use logs + analyzer to identify P2 priorities.

**Blocker**: None - logging complete and ready to use.
