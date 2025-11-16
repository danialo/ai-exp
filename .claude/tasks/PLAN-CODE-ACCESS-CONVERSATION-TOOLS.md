# Plan: Code Access Conversation Tools

## Goal
Enable Astra to read code and schedule modifications during conversations, while maintaining safety through approval workflows.

## Scope
Add 3 conversational tools that interface with the existing CodeAccessService:
1. `read_source_code` - Read files (immediate, read-only)
2. `schedule_code_modification` - Queue code changes (approval required)
3. `read_logs` - Read log files (immediate, read-only)

## Non-Goals
- NOT adding immediate code modification (too dangerous)
- NOT modifying CodeAccessService itself (already built)
- NOT building full notification system yet (separate task)

## Tool Specifications

### 1. read_source_code

**Purpose**: Let Astra read source files during conversations for debugging/investigation

**Function Definition**:
```python
{
    "type": "function",
    "function": {
        "name": "read_source_code",
        "description": "Read a source code file from the codebase. Use this to investigate code, understand implementations, or debug issues.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path from project root (e.g., 'src/services/goal_store.py')"
                }
            },
            "required": ["file_path"]
        }
    }
}
```

**Implementation**:
```python
async def read_source_code(file_path: str) -> dict:
    """Read source file via CodeAccessService."""
    content, error = await code_access_service.read_file(file_path)

    if error:
        return {
            "success": false,
            "error": error,
            "file_path": file_path
        }

    return {
        "success": true,
        "file_path": file_path,
        "content": content,
        "lines": len(content.split('\n'))
    }
```

**Safety**: Uses existing CodeAccessService boundaries (allowed/forbidden paths)

### 2. schedule_code_modification

**Purpose**: Let Astra queue a code change for your approval

**Function Definition**:
```python
{
    "type": "function",
    "function": {
        "name": "schedule_code_modification",
        "description": "Schedule a code modification for user approval. Creates a task that will execute after approval. Use this when you've identified a fix that needs to be applied.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "File to modify (e.g., 'src/services/task_scheduler.py')"
                },
                "new_content": {
                    "type": "string",
                    "description": "Complete new file content"
                },
                "reason": {
                    "type": "string",
                    "description": "Clear explanation of why this change is needed"
                },
                "goal_id": {
                    "type": "string",
                    "description": "Optional: ID of related goal",
                    "default": null
                }
            },
            "required": ["file_path", "new_content", "reason"]
        }
    }
}
```

**Implementation**:
```python
async def schedule_code_modification(
    file_path: str,
    new_content: str,
    reason: str,
    goal_id: Optional[str] = None
) -> dict:
    """Schedule a code modification task for approval."""

    # Check if code_access_service exists
    if not code_access_service:
        return {"success": false, "error": "Code access service not available"}

    # Check if task_scheduler exists
    if not task_scheduler:
        return {"success": false, "error": "Task scheduler not available"}

    # Verify path is allowed
    can_access, access_error = code_access_service.can_access(file_path)
    if not can_access:
        return {
            "success": false,
            "error": f"Access denied: {access_error}",
            "file_path": file_path
        }

    # Create a manual CODE_MODIFY task
    task_id = f"code_mod_{uuid4().hex[:8]}"
    task = TaskDefinition(
        id=task_id,
        name=f"Modify {file_path}",
        type=TaskType.CODE_MODIFY,
        schedule=TaskSchedule.MANUAL,  # Requires manual trigger
        prompt=reason,
        enabled=True,
        metadata={
            "file_path": file_path,
            "new_content": new_content,
            "reason": reason,
            "goal_id": goal_id,
            "requested_by": "astra",
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "status": "awaiting_approval"
        }
    )

    # Add task to scheduler
    task_scheduler.tasks[task_id] = task
    task_scheduler._save_tasks()

    # Log to identity ledger
    append_event(LedgerEvent(
        event="code_modification_scheduled",
        meta={
            "task_id": task_id,
            "file_path": file_path,
            "reason": reason,
            "goal_id": goal_id,
        }
    ))

    return {
        "success": true,
        "task_id": task_id,
        "file_path": file_path,
        "status": "awaiting_approval",
        "message": f"Code modification scheduled. Task ID: {task_id}. User approval required before execution."
    }
```

**Safety**:
- Creates MANUAL task (doesn't auto-execute)
- Logs to identity ledger
- Still respects allowed/forbidden paths

### 3. read_logs

**Purpose**: Let Astra read log files for debugging

**Function Definition**:
```python
{
    "type": "function",
    "function": {
        "name": "read_logs",
        "description": "Read application log files for debugging. Use this to investigate errors, check system behavior, or troubleshoot issues.",
        "parameters": {
            "type": "object",
            "properties": {
                "log_file": {
                    "type": "string",
                    "description": "Log file to read (e.g., 'app/astra.log', 'app/errors.log')",
                    "enum": ["app/astra.log", "app/errors.log", "awareness", "beliefs", "conversations", "errors", "memory", "performance"]
                },
                "lines": {
                    "type": "integer",
                    "description": "Number of recent lines to read (default: 100)",
                    "default": 100
                }
            },
            "required": ["log_file"]
        }
    }
}
```

**Implementation**:
```python
async def read_logs(log_file: str, lines: int = 100) -> dict:
    """Read log files for debugging."""

    # Construct full path
    log_path = Path("logs") / log_file

    # Security check - ensure path stays within logs/
    try:
        resolved = log_path.resolve()
        if not str(resolved).startswith(str(Path("logs").resolve())):
            return {"success": false, "error": "Path traversal detected"}
    except:
        return {"success": false, "error": "Invalid log path"}

    # Check if file exists
    if not log_path.exists():
        return {
            "success": false,
            "error": f"Log file not found: {log_file}",
            "available_logs": list_available_logs()
        }

    # Read last N lines
    try:
        with open(log_path, 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            content = ''.join(recent_lines)

        return {
            "success": true,
            "log_file": log_file,
            "lines_returned": len(recent_lines),
            "total_lines": len(all_lines),
            "content": content
        }
    except Exception as e:
        return {
            "success": false,
            "error": f"Failed to read log: {str(e)}"
        }
```

**Safety**:
- Restricted to logs/ directory only
- Path traversal protection
- Limited to recent N lines (prevents huge responses)

## Where Tools Are Registered

Tools need to be added to PersonaService's available tools. Need to find where tools are defined:

**Expected location**: `src/services/persona_service.py` or similar

**Changes needed**:
1. Import code_access_service, task_scheduler
2. Define the 3 tool functions
3. Add tool definitions to available_tools array
4. Wire tool calls to execute the functions

## Files to Modify

1. **src/services/persona_service.py** (or wherever tools are defined)
   - Add 3 tool definitions
   - Add 3 tool handler functions
   - Wire to code_access_service and task_scheduler

2. **Tests** (new file: `tests/test_code_access_tools.py`)
   - Test read_source_code with allowed/forbidden paths
   - Test schedule_code_modification creates task
   - Test read_logs with path traversal attempts

## Testing Plan

### Manual Testing
1. Start app
2. Chat with Astra: "Can you read src/services/goal_store.py?"
3. Verify she calls read_source_code tool
4. Verify content is returned
5. Chat: "The timeout needs to be increased to 60s"
6. Verify she calls schedule_code_modification
7. Check that task was created with MANUAL schedule
8. Chat: "Can you check the recent errors in the log?"
9. Verify she calls read_logs
10. Verify log content is returned

### Unit Tests
```python
def test_read_source_code_allowed_path()
def test_read_source_code_forbidden_path()
def test_schedule_code_modification_creates_task()
def test_schedule_code_modification_respects_boundaries()
def test_read_logs_path_traversal_prevention()
def test_read_logs_line_limit()
```

## Approval Workflow (Future)

This plan does NOT include the full approval UI/API. That's a separate task.

For now, approvals would be manual:
```bash
# List pending tasks
sqlite3 persona_space/tasks/tasks.json  # or wherever tasks are stored

# Execute approved task
# (Manual trigger mechanism TBD)
```

Future work:
- API endpoint: POST /api/v1/tasks/{task_id}/approve
- API endpoint: POST /api/v1/tasks/{task_id}/reject
- Dashboard showing pending modifications
- Notification system

## Risks

1. **Tool complexity**: PersonaService integration might be more complex than expected
2. **Task storage**: Need to verify where MANUAL tasks are persisted
3. **Approval mechanism**: No good way to approve tasks yet (future work)
4. **Log file size**: Very large logs could cause issues (mitigated by line limit)

## Rollback Plan

If this doesn't work:
```bash
git checkout claude/feature/autonomous-goal-generation
git branch -D claude/feature/code-access-conversation-tools
```

All changes in isolated branch, easy to discard.

## Estimated Effort

- Find tool registration location: 30 min
- Implement 3 tools: 2 hours
- Write tests: 1 hour
- Manual testing: 30 min

**Total: ~4 hours**

## Questions to Answer Before Starting

1. Where are conversation tools currently defined?
2. How are tool calls currently handled?
3. Where are MANUAL tasks stored?
4. Is there existing approval mechanism I'm missing?

---

## Approval Needed

Do you want me to:
- ✅ Implement as specified above?
- ❌ Change something in the plan?
- ❌ Don't implement at all?
