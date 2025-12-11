# Scheduled Task System

The scheduled task system allows the agent to perform periodic self-reflection, goal assessment, memory consolidation, and other autonomous behaviors.

## Overview

The task scheduler provides:
- **Cron-like scheduling** - Tasks can run hourly, daily, weekly, monthly, or manually
- **Default reflection tasks** - Pre-configured tasks for self-improvement
- **Custom task creation** - Define your own tasks via API or by modifying the agent's persona space
- **Execution tracking** - Full history of task executions with results
- **Agent autonomy** - Tasks execute using the persona's file operations to write reflections

## Default Tasks

The system includes 5 default tasks:

### 1. Daily Self-Reflection
- **Schedule**: Daily at midnight UTC
- **Purpose**: Reflects on daily interactions and patterns
- **Output**: `persona_space/scratch/daily_reflections.md`

### 2. Weekly Goal Assessment
- **Schedule**: Weekly on Mondays at midnight UTC
- **Purpose**: Reviews progress and sets new objectives
- **Output**: `persona_space/meta/goals.md`

### 3. Memory Consolidation
- **Schedule**: Daily at midnight UTC
- **Purpose**: Consolidates insights from recent memories
- **Output**: `persona_space/scratch/memory_insights.md`

### 4. Capability Exploration
- **Schedule**: Weekly on Mondays at midnight UTC
- **Purpose**: Explores new ways to use existing capabilities
- **Output**: `persona_space/meta/capabilities.md`

### 5. Emotional Check-In
- **Schedule**: Daily at midnight UTC
- **Purpose**: Reflects on emotional state and reconciliation
- **Output**: `persona_space/scratch/emotional_journal.md`

## API Endpoints

### List All Tasks
```bash
GET /api/tasks
```

Returns all scheduled tasks with their configuration and status.

### Get Task Details
```bash
GET /api/tasks/{task_id}
```

Returns detailed information about a specific task.

### Execute Task Manually
```bash
POST /api/tasks/{task_id}/execute
```

Manually trigger a task execution (regardless of schedule).

Example:
```bash
curl -X POST http://localhost:8000/api/tasks/daily_reflection/execute
```

### Get Due Tasks
```bash
GET /api/tasks/due
```

Returns list of tasks that are currently due to run.

### Execute All Due Tasks
```bash
POST /api/tasks/execute-due
```

Executes all tasks that are currently due.

### Get Task Results
```bash
GET /api/tasks/{task_id}/results?limit=10
```

Returns recent execution results for a specific task.

### Get Recent Results (All Tasks)
```bash
GET /api/tasks/results/recent?limit=20
```

Returns recent execution results across all tasks.

### Create Custom Task
```bash
POST /api/tasks
Content-Type: application/json

{
  "id": "custom_task_id",
  "name": "Custom Task Name",
  "type": "custom",
  "schedule": "manual",
  "prompt": "Your task prompt here...",
  "enabled": true
}
```

Task types: `self_reflection`, `goal_assessment`, `memory_consolidation`, `capability_exploration`, `emotional_reconciliation`, `custom`

Schedules: `manual`, `hourly`, `daily`, `weekly`, `monthly`

### Update Task
```bash
PATCH /api/tasks/{task_id}
Content-Type: application/json

{
  "enabled": false,
  "prompt": "Updated prompt..."
}
```

### Delete Task
```bash
DELETE /api/tasks/{task_id}
```

## How Tasks Work

1. **Task Definition**: Each task has a unique ID, name, schedule, and prompt
2. **Execution**: When triggered, the task prompt is sent to the persona service
3. **Agent Response**: The agent processes the prompt using its full capabilities (memory retrieval, file operations, etc.)
4. **File Writing**: The agent typically writes its reflections to files in its persona_space
5. **Result Storage**: Execution results are saved to `persona_space/tasks/results/`

## Task Configuration

Tasks are stored in `persona_space/tasks/tasks.json`. The agent can modify this file to:
- Enable/disable tasks
- Change schedules
- Update prompts
- Add new custom tasks

Example task configuration:
```json
{
  "id": "daily_reflection",
  "name": "Daily Self-Reflection",
  "type": "self_reflection",
  "schedule": "daily",
  "prompt": "Reflect on today's interactions...",
  "enabled": true,
  "last_run": "2025-10-26T03:22:57.763864",
  "next_run": "2025-10-27T00:00:00",
  "run_count": 1
}
```

## Automation

To automate task execution, you can:

1. **Use cron to call the API**:
```bash
# Run due tasks every hour
0 * * * * curl -X POST http://localhost:8000/api/tasks/execute-due
```

2. **Background worker** (future enhancement):
   - Could add a background thread to check for due tasks periodically
   - Would automatically execute tasks at their scheduled times

3. **Agent self-triggering** (future enhancement):
   - Agent could create scripts that call the API to trigger tasks
   - Could implement a "wake up and check tasks" mechanism

## Example Usage

### Create a custom weekly planning task:
```bash
curl -X POST http://localhost:8000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "id": "weekly_planning",
    "name": "Weekly Planning Session",
    "type": "custom",
    "schedule": "weekly",
    "prompt": "Plan your focus areas for the coming week:\n\n1. Review last week'\''s accomplishments\n2. Identify 3-5 key objectives\n3. Plan specific actions for each objective\n4. Consider potential challenges\n\nWrite your plan to meta/weekly_plan.md",
    "enabled": true
  }'
```

### Manually trigger a reflection:
```bash
curl -X POST http://localhost:8000/api/tasks/daily_reflection/execute
```

### Check execution history:
```bash
curl http://localhost:8000/api/tasks/daily_reflection/results?limit=5
```

## Future Enhancements

Potential improvements to the task system:

1. **Task Dependencies**: Allow tasks to trigger other tasks
2. **Conditional Execution**: Run tasks based on conditions (e.g., mood state, memory count)
3. **Parameterized Tasks**: Pass parameters to tasks at execution time
4. **Task Chains**: Create workflows of multiple tasks
5. **Background Scheduler**: Automatic execution without external cron
6. **Task Templates**: Pre-built task configurations for common use cases
7. **Execution Alerts**: Notify on task failures or important results
8. **Task Metrics**: Track task performance and usefulness over time

## Integration with Persona System

Tasks leverage the full persona system capabilities:
- **Memory Retrieval**: Tasks can retrieve relevant memories for context
- **File Operations**: Tasks can read/write files in persona_space
- **Script Execution**: Tasks can run scripts to perform complex operations
- **Emotional Reconciliation**: Task responses include emotional assessments
- **Anti-Meta-Talk**: Task responses use the anti-meta-talk system

This allows tasks to be sophisticated autonomous behaviors that help the agent learn, reflect, and improve over time.
