# execute_goal Quick Reference

**Last Updated**: 2025-11-11

## ⚠️ CRITICAL: Use the Correct Endpoint

### ✅ CORRECT: `/api/persona/chat`
```bash
curl -k -X POST https://172.239.66.45:8443/api/persona/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Use execute_goal to create a calculator","retrieve_memories":false}'
```

**Supports:**
- ✅ Tool calling (execute_goal, file operations, web search, etc.)
- ✅ Belief system integration
- ✅ Full conversation history
- ✅ Dissonance resolution
- ✅ 14 tools available

### ❌ WRONG: `/api/chat`
```bash
# DO NOT USE THIS ENDPOINT FOR TOOLS
curl -k -X POST https://172.239.66.45:8443/api/chat ...
```

**Limitations:**
- ❌ NO tool calling support
- ❌ Uses ExperienceLens (simple LLM generation only)
- ❌ Cannot execute code or use execute_goal

## How to Test execute_goal

### 1. Simple Request
```bash
curl -k -X POST https://172.239.66.45:8443/api/persona/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Use execute_goal to create a simple calculator function",
    "retrieve_memories": false
  }'
```

### 2. Watch the Logs
```bash
tail -f server.log | grep -E "execute_goal|LLM API CALL|AGENT ACTION"
```

### 3. Check Generated Files
```bash
ls -lt tests/generated/ | head -5
```

## Expected Log Output

### ✅ Success Pattern
```
🔧 LLM API CALL
Model: gpt-4o
Tools count: 14
execute_goal present: True

🔧 LLM API RESPONSE
Finish reason: tool_calls
Has tool calls: True
Tool calls count: 1
  - execute_goal()

🤖 AGENT ACTION: execute_goal(goal_text=implement_feature, timeout_ms=120000)
   ✓ Result: 🎯 Goal Execution Complete

INFO: 172.239.66.45:XXXXX - "POST /api/persona/chat HTTP/1.1" 200 OK
```

### ❌ Wrong Endpoint Pattern
```
# No "LLM API CALL" block at all
# No "execute_goal present" message
# Just a 200 OK with text response
```

## How execute_goal Works

### Full Pipeline
```
User Request
    ↓
/api/persona/chat endpoint
    ↓
PersonaService.generate_response()
    ↓
LLM.generate_with_tools() [includes execute_goal in tools list]
    ↓
GPT-4o decides to call execute_goal
    ↓
PersonaService._execute_tool("execute_goal", {...})
    ↓
GoalExecutionService.execute_goal()
    ↓
HTN Planner decomposes goal
    ↓
TaskGraph builds execution plan
    ↓
TaskExecutionEngine runs tasks
    ↓
CodeGenerator creates files (via LLM)
    ↓
Test execution
    ↓
Result returned to Astra
    ↓
Astra responds to user with outcome
```

## Tool Definition

```python
{
    "type": "function",
    "function": {
        "name": "execute_goal",
        "description": "Autonomously execute a coding goal end-to-end using HTN planning and task execution. The system will decompose the goal, create files, run tests, and return execution results.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal_text": {
                    "type": "string",
                    "description": "Goal type: implement_feature | fix_bug | refactor_code | add_tests"
                },
                "context": {
                    "type": "object",
                    "description": "Optional HTN context (file paths, dependencies, etc.)"
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Execution timeout in milliseconds (default: 60000)"
                }
            },
            "required": ["goal_text"]
        }
    }
}
```

## Troubleshooting

### Issue: No tool call happening
**Check:**
1. Are you using `/api/persona/chat`? (NOT `/api/chat`)
2. Do the logs show `execute_goal present: True`?
3. Is the server running? `ps aux | grep uvicorn`

### Issue: Server crashes (500 error)
**Check:**
1. Logs: `tail -100 logs/errors/errors.log`
2. Server log: `tail -100 server.log`
3. Common causes:
   - Logger initialization issues (fixed 2025-11-11)
   - None values in belief checker (fixed 2025-11-11)

### Issue: Code generation produces placeholders
**This is expected behavior:**
- The pipeline is working correctly
- Code quality needs prompt tuning
- See "Code generation quality improvements" in TODO.md

## Files to Check

### Source Code
- `src/services/persona_service.py` - execute_goal tool definition and handler
- `src/services/goal_execution_service.py` - HTN + TaskGraph integration
- `src/services/code_generator.py` - LLM-based code generation
- `src/services/llm.py` - OpenAI API wrapper with tool support

### Documentation
- `.claude/tasks/PHASE3-COMPLETE.md` - Full Phase 3 documentation
- `docs/EXECUTE_GOAL_QUICK_REF.md` - This file

### Logs
- `server.log` - Main application log
- `logs/errors/errors.log` - Error tracebacks
- `logs/conversations/conversations.log` - User/assistant interactions

### Generated Files
- `tests/generated/feature_*.py` - Implementation files
- `tests/generated/test_*.py` - Test files

## Known Issues

### Code Generation Quality
**Status**: In Progress
**Impact**: Pipeline works but output quality varies
**Workaround**: None yet - needs prompt engineering
**Tracking**: TODO.md "Code generation quality improvements"

### Verbose Logging Performance
**Status**: Accepted
**Impact**: Lots of console output during execution
**Workaround**: Filter logs: `tail -f server.log | grep -v "SLOW TICK"`
**Notes**: Useful for debugging, can be toggled off in production

## Success Criteria

### ✅ Working System
- execute_goal tool is exposed to GPT-4o
- GPT-4o calls the tool when asked
- HTN planner decomposes the goal
- Files are created in tests/generated/
- Tests run (even if they fail)
- Result is returned to user
- Request completes with 200 OK

### ❌ NOT Required for Success
- Tests passing (depends on code quality)
- Perfect code generation (prompt tuning needed)
- Fast execution (<10s) - 20-30s is normal

## Next Steps

1. **Improve Code Generation** (TODO.md)
   - Tune CodeGenerator prompts
   - Add more context to generation requests
   - Implement validation feedback loop

2. **Add More HTN Methods** (Phase 4)
   - Context-aware decomposition
   - Adaptive method selection
   - Belief-informed planning

3. **WebSocket Updates** (Phase 4)
   - Real-time task progress
   - TaskGraph visualization
   - Live execution monitoring
