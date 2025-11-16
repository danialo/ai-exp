# TaskGraph Usage Guide

**Date**: 2025-11-08
**Status**: Production Ready (Phase 2 Complete)

## Overview

TaskGraph provides production-ready dependency tracking and parallel task execution with circuit breakers, retry logic, and safety checks. This guide shows how to use TaskGraph and TaskExecutor in your code.

## Quick Start

### Basic Graph Execution

```python
from src.services.task_graph import TaskGraph, TaskNode, DependencyPolicy
from src.services.task_scheduler import create_task_scheduler

# Create task scheduler (auto-creates TaskExecutor)
scheduler = create_task_scheduler(
    persona_space_path="persona_space",
    raw_store=raw_store,
    abort_monitor=abort_monitor
)

# Create a task graph
graph = TaskGraph(
    graph_id="my_workflow",
    max_parallel=5,  # Max concurrent tasks
    max_retry_tokens=100  # Retry budget
)

# Add tasks with dependencies
graph.add_task(
    task_id="gather_data",
    action_name="data_collection",
    normalized_args={"source": "database"},
    resource_ids=["db_connection"],
    version="1.0",
    priority=10  # Higher = higher priority
)

graph.add_task(
    task_id="process_data",
    action_name="data_processing",
    normalized_args={"algorithm": "standard"},
    resource_ids=["cpu"],
    version="1.0",
    dependencies=["gather_data"],  # Wait for gather_data
    priority=5
)

graph.add_task(
    task_id="generate_report",
    action_name="reporting",
    normalized_args={"format": "pdf"},
    resource_ids=[],
    version="1.0",
    dependencies=["process_data"],
    priority=1
)

# Execute graph with parallel execution
result = await scheduler.execute_graph(
    graph=graph,
    persona_service=persona_service,
    max_parallel=5
)

# Check results
print(f"Completed {result['statistics']['states']['succeeded']} tasks")
print(f"Failed {result['statistics']['states']['failed']} tasks")
```

## Advanced Features

### 1. Dependency Policies

Control how tasks handle failed dependencies:

```python
from src.services.task_graph import DependencyPolicy

# ABORT: Fail if any dependency fails (default)
graph.add_task(
    task_id="critical_task",
    action_name="critical_action",
    normalized_args={...},
    resource_ids=[],
    version="1.0",
    dependencies=["task_a"],
    on_dep_fail=DependencyPolicy.ABORT  # Fail if task_a fails
)

# SKIP: Skip if any dependency fails
graph.add_task(
    task_id="optional_task",
    action_name="optional_action",
    normalized_args={...},
    resource_ids=[],
    version="1.0",
    dependencies=["task_b"],
    on_dep_fail=DependencyPolicy.SKIP  # Skip if task_b fails
)

# CONTINUE_IF_ANY: Run if at least one dependency succeeds
graph.add_task(
    task_id="flexible_task",
    action_name="flexible_action",
    normalized_args={...},
    resource_ids=[],
    version="1.0",
    dependencies=["task_c", "task_d"],
    on_dep_fail=DependencyPolicy.CONTINUE_IF_ANY  # Run if either succeeds
)
```

### 2. Concurrency Control

Limit concurrent executions globally or per-action:

```python
# Global concurrency limit (via graph constructor)
graph = TaskGraph(
    graph_id="limited_graph",
    max_parallel=3  # Max 3 tasks running at once
)

# Per-action concurrency limits (via execute_graph)
result = await scheduler.execute_graph(
    graph=graph,
    persona_service=persona_service,
    max_parallel=10,  # Overall limit
    per_action_caps={
        "expensive_computation": 2,  # Max 2 concurrent expensive computations
        "database_query": 5,  # Max 5 concurrent DB queries
        "api_call": 3  # Max 3 concurrent API calls
    }
)
```

### 3. Priority and Deadlines

Tasks are executed by priority (higher first) with deadline tiebreakers:

```python
from datetime import datetime, timedelta, timezone

graph.add_task(
    task_id="urgent_task",
    action_name="urgent_action",
    normalized_args={...},
    resource_ids=[],
    version="1.0",
    priority=10.0,  # Higher priority
    deadline=datetime.now(timezone.utc) + timedelta(minutes=5)  # Urgent deadline
)

graph.add_task(
    task_id="normal_task",
    action_name="normal_action",
    normalized_args={...},
    resource_ids=[],
    version="1.0",
    priority=5.0,  # Lower priority
    deadline=datetime.now(timezone.utc) + timedelta(hours=1)
)
```

### 4. Retry Logic with TaskExecutor

```python
# Execute single task with retry
result = await scheduler.execute_task_with_retry(
    task_id="my_task",
    persona_service=persona_service,
    max_retries=5,  # Override default retries
    timeout_ms=60000  # 60 second timeout
)

if result.success:
    print(f"Task succeeded: {result.response}")
else:
    print(f"Task failed after retries: {result.error}")
```

### 5. Idempotency

TaskExecutor automatically handles idempotency using deterministic keys:

```python
# These two calls will only execute once (same idempotency key)
result1 = await executor.execute(
    node=TaskNode(
        task_id="task_1",
        action_name="my_action",
        normalized_args={"param": "value"},
        resource_ids=["resource1"],
        version="1.0"
    ),
    task_callable=my_callable
)

result2 = await executor.execute(
    node=TaskNode(
        task_id="task_2",  # Different ID
        action_name="my_action",  # Same action
        normalized_args={"param": "value"},  # Same args
        resource_ids=["resource1"],  # Same resources
        version="1.0"  # Same version
    ),
    task_callable=my_callable
)
# result2 will be fetched from cache
```

### 6. Circuit Breakers

Automatic circuit breakers prevent cascade failures:

```python
# Circuit breakers are automatic per action type
# Default settings:
# - Window size: 10 attempts
# - Failure threshold: 50%
# - Half-open timeout: 30 seconds

# Check circuit breaker state
stats = graph.get_stats()
breaker_states = stats["breaker_states"]

for action, state in breaker_states.items():
    print(f"{action}: {state['state']}")  # CLOSED, OPEN, or HALF_OPEN
    if state['state'] == 'OPEN':
        print(f"  Opens at: {state['opens_at']}")
```

### 7. Safety Envelope Integration

Tasks are automatically checked against safety constraints:

```python
from src.services.abort_condition_monitor import AbortConditionMonitor

# Create abort monitor with safety rules
abort_monitor = AbortConditionMonitor(
    raw_store=raw_store,
    ledger=identity_ledger
)

# Pass to scheduler
scheduler = create_task_scheduler(
    persona_space_path="persona_space",
    raw_store=raw_store,
    abort_monitor=abort_monitor  # Tasks checked before execution
)

# Tasks will automatically abort if safety conditions violated
result = await scheduler.execute_task_with_retry(
    task_id="safe_task",
    persona_service=persona_service
)
```

## Common Patterns

### Pattern 1: Fan-Out / Fan-In

```python
graph = TaskGraph(graph_id="fan_out_in")

# Single root task
graph.add_task(
    task_id="prepare",
    action_name="preparation",
    normalized_args={...},
    resource_ids=[],
    version="1.0"
)

# Fan out: Multiple parallel tasks depend on prepare
for i in range(5):
    graph.add_task(
        task_id=f"process_{i}",
        action_name="processing",
        normalized_args={"batch": i},
        resource_ids=[],
        version="1.0",
        dependencies=["prepare"]
    )

# Fan in: Single task depends on all parallel tasks
graph.add_task(
    task_id="aggregate",
    action_name="aggregation",
    normalized_args={...},
    resource_ids=[],
    version="1.0",
    dependencies=[f"process_{i}" for i in range(5)]
)
```

### Pattern 2: Pipeline with Stages

```python
graph = TaskGraph(graph_id="pipeline")

# Stage 1: Extraction (3 parallel sources)
sources = ["database", "api", "files"]
for source in sources:
    graph.add_task(
        task_id=f"extract_{source}",
        action_name="extraction",
        normalized_args={"source": source},
        resource_ids=[],
        version="1.0"
    )

# Stage 2: Transformation (depends on all extractions)
graph.add_task(
    task_id="transform",
    action_name="transformation",
    normalized_args={...},
    resource_ids=[],
    version="1.0",
    dependencies=[f"extract_{s}" for s in sources]
)

# Stage 3: Load (depends on transformation)
graph.add_task(
    task_id="load",
    action_name="loading",
    normalized_args={...},
    resource_ids=[],
    version="1.0",
    dependencies=["transform"]
)
```

### Pattern 3: Conditional Execution

```python
graph = TaskGraph(graph_id="conditional")

# Primary task
graph.add_task(
    task_id="primary",
    action_name="primary_action",
    normalized_args={...},
    resource_ids=[],
    version="1.0"
)

# Fallback task (runs if primary fails)
graph.add_task(
    task_id="fallback",
    action_name="fallback_action",
    normalized_args={...},
    resource_ids=[],
    version="1.0",
    dependencies=["primary"],
    on_dep_fail=DependencyPolicy.CONTINUE_IF_ANY  # Run even if primary fails
)

# Cleanup task (always runs)
graph.add_task(
    task_id="cleanup",
    action_name="cleanup_action",
    normalized_args={...},
    resource_ids=[],
    version="1.0",
    dependencies=["primary", "fallback"],
    on_dep_fail=DependencyPolicy.CONTINUE_IF_ANY  # Run regardless
)
```

## Monitoring and Debugging

### Check Graph Status

```python
# During execution, check graph state
stats = graph.get_stats()

print(f"Total tasks: {stats['total_tasks']}")
print(f"Running: {stats['running_tasks']}")
print(f"States: {stats['states']}")
print(f"Retry tokens used: {stats['retry_tokens_used']}/{graph.max_retry_tokens}")
print(f"Circuit breakers: {stats['breaker_states']}")
```

### Get Execution Results

```python
result = await scheduler.execute_graph(graph, persona_service)

# Overall statistics
print(f"Graph ID: {result['graph_id']}")
print(f"Total tasks: {result['total_tasks']}")
print(f"Elapsed: {result['elapsed_ms']}ms")
print(f"Completed at: {result['completed_at']}")

# Per-task results
for task_id, task_result in result["results"].items():
    if task_result["success"]:
        print(f"✓ {task_id}: {task_result.get('execution_time_ms')}ms")
    else:
        print(f"✗ {task_id}: {task_result.get('error')}")

# State breakdown
states = result["statistics"]["states"]
print(f"\nSucceeded: {states.get('succeeded', 0)}")
print(f"Failed: {states.get('failed', 0)}")
print(f"Aborted: {states.get('aborted', 0)}")
print(f"Skipped: {states.get('skipped', 0)}")
```

## Best Practices

1. **Use Meaningful Task IDs**: Make them descriptive and unique
   ```python
   task_id="user_registration_email_validation"  # Good
   task_id="task_42"  # Bad
   ```

2. **Set Appropriate Priorities**: Higher for time-sensitive tasks
   ```python
   priority=10.0  # Critical/urgent
   priority=5.0   # Normal
   priority=1.0   # Low priority/cleanup
   ```

3. **Choose Right Dependency Policy**: Based on business logic
   - Use `ABORT` for critical dependencies
   - Use `SKIP` for optional enhancements
   - Use `CONTINUE_IF_ANY` for fallback scenarios

4. **Limit Concurrency**: Prevent resource exhaustion
   ```python
   per_action_caps={"database_query": 10}  # Don't overwhelm DB
   ```

5. **Use Deadlines**: For time-sensitive workflows
   ```python
   deadline=datetime.now(timezone.utc) + timedelta(minutes=5)
   ```

6. **Monitor Circuit Breakers**: Check for repeated failures
   ```python
   if breaker_state["state"] == "OPEN":
       logger.warning(f"Circuit breaker open for {action}")
   ```

## Testing

Example test using TaskGraph:

```python
import pytest
from src.services.task_graph import TaskGraph, TaskNode

@pytest.mark.asyncio
async def test_my_workflow(scheduler, mock_persona_service):
    """Test my custom workflow."""
    graph = TaskGraph(graph_id="test_workflow")

    # Add tasks
    graph.add_task(task_id="step1", ...)
    graph.add_task(task_id="step2", dependencies=["step1"], ...)

    # Execute
    result = await scheduler.execute_graph(
        graph=graph,
        persona_service=mock_persona_service
    )

    # Assert
    assert result["statistics"]["states"]["succeeded"] == 2
    assert result["results"]["step2"]["success"] is True
```

## See Also

- `src/services/task_graph.py` - TaskGraph implementation
- `src/services/task_executor.py` - TaskExecutor implementation
- `src/services/task_scheduler.py` - TaskScheduler integration
- `tests/test_task_graph.py` - Unit tests (29 tests)
- `tests/test_task_executor.py` - Unit tests (26 tests)
- `tests/integration/test_task_graph_execution.py` - Integration tests (10 tests)
- `docs/PHASE2_IMPLEMENTATION_PLAN.md` - Design document
- `docs/AUTONOMOUS_AGENT_ARCHITECTURE_ANALYSIS.md` - Architecture overview
