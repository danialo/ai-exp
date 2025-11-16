# Critical Issue: Goal Generator vs Actual Capabilities

## The Problem

The goal generator I built creates goals that **Astra cannot execute** because she doesn't have access to her own source code.

### Example of Broken Flow

**What happens now:**
```
1. TaskFailureDetector sees: daily_backup task failed 4 times
2. Creates goal: "Fix recurring failures in 'daily_backup' task"
3. Goal gets created in database
4. HTN Planner selects goal
5. Tries to decompose into tasks...
   ❌ PROBLEM: Astra can't modify src/services/task_scheduler.py
   ❌ PROBLEM: Astra can't fix the backup code
   ❌ PROBLEM: Astra can't deploy changes
```

**What Astra CAN access:**
- ✅ `persona_space/` - Her workspace (beliefs, identity, logs, meta)
- ✅ `data/` - Databases (read/write)
- ✅ LLM calls - Can generate text, analyze, reason
- ✅ Task execution - Can run introspective tasks (reflection, goal assessment)

**What Astra CANNOT access:**
- ❌ `src/` - Her own source code
- ❌ `app.py` - Application logic
- ❌ Code deployment
- ❌ System modifications

## Current Task Types (What She Can Execute)

From `task_scheduler.py`:
```python
class TaskType(str, Enum):
    SELF_REFLECTION = "self_reflection"           # ✅ LLM introspection
    GOAL_ASSESSMENT = "goal_assessment"           # ✅ LLM evaluation
    MEMORY_CONSOLIDATION = "memory_consolidation" # ✅ LLM summarization
    CAPABILITY_EXPLORATION = "capability_exploration"  # ✅ LLM discovery
    EMOTIONAL_RECONCILIATION = "emotional_reconciliation"  # ✅ LLM processing
    CUSTOM = "custom"                             # ✅ LLM custom prompts
```

**All of these are LLM-based cognitive tasks, NOT code modification tasks.**

## The Mismatch

### Goals the System Creates:
1. ❌ "Fix recurring failures in 'daily_backup' task" - **Requires code access**
2. ❌ "Improve test coverage for auth module" - **Requires writing tests**
3. ❌ "Refactor complex functions in goal_store.py" - **Requires code modification**
4. ❌ "Optimize slow database queries" - **Requires schema/query changes**

### Goals Astra Could ACTUALLY Execute:
1. ✅ "Develop understanding of backup failure patterns"
2. ✅ "Create belief about system reliability requirements"
3. ✅ "Request user investigation of backup timeouts"
4. ✅ "Document observed failure patterns for user review"
5. ✅ "Generate hypothesis about backup failure root cause"

## Three Possible Solutions

### Option 1: User-Request Goals (Recommended)

**Change goal framing from action to request:**

```python
# BEFORE (broken):
text="Fix recurring failures in 'daily_backup' task"
# Implies: I will fix this myself

# AFTER (works):
text="Request user fix recurring backup task failures (4 failures, timeout errors)"
# Implies: I'm notifying you of an issue
```

**Decomposition:**
```python
tasks = [
    TaskDefinition(
        type=TaskType.CUSTOM,
        prompt="Analyze the backup task failure patterns and create a detailed report for the user"
    ),
    TaskDefinition(
        type=TaskType.CUSTOM,
        prompt="Create a belief about system reliability based on backup failures"
    ),
    # No code modification tasks!
]
```

### Option 2: Knowledge/Insight Goals

**Goals about understanding, not fixing:**

```python
GoalProposal(
    text="Understand root cause of backup task timeout pattern",
    category=GoalCategory.KNOWLEDGE,  # Not CAPABILITY
    # Tasks would be:
    # 1. Analyze failure logs
    # 2. Identify common patterns
    # 3. Generate hypothesis
    # 4. Document findings in persona_space/learning_journey/
)
```

### Option 3: Extend Capabilities (Future)

**Give Astra ACTUAL code modification capabilities:**

```python
# New task types:
class TaskType(str, Enum):
    # ... existing ...
    CODE_ANALYSIS = "code_analysis"       # Read/analyze code
    CODE_MODIFICATION = "code_modification"  # Write code changes
    TEST_GENERATION = "test_generation"   # Write tests
    FILE_EDIT = "file_edit"               # Edit specific files
```

**Requirements:**
- Sandboxed code execution
- Git branch isolation
- User approval for changes
- Rollback capability
- Clear boundaries (what code can she touch?)

**This is a MAJOR expansion of scope - not recommended for initial implementation.**

## Recommended Fix

### 1. Rename Goal Categories

Update `GoalCategory` to reflect what she can actually do:

```python
class GoalCategory(str, Enum):
    # Existing (keep these)
    EFFICIENCY = "efficiency"
    CAPABILITY = "capability"

    # Add these:
    USER_REQUEST = "user_request"    # Goals requesting user action
    INSIGHT = "insight"              # Goals about understanding
    KNOWLEDGE = "knowledge"          # Goals about learning
    SELF_IMPROVEMENT = "self_improvement"  # Goals about her own behavior
```

### 2. Update TaskFailureDetector

```python
class TaskFailureDetector(PatternDetector):
    async def detect(self) -> List[GoalProposal]:
        failures = self._get_recent_failures()

        if not failures:
            return []

        failure_counts = Counter(f["task_type"] for f in failures)
        proposals = []

        for task_type, count in failure_counts.items():
            if count >= self._failure_threshold:
                # BEFORE:
                # text = f"Fix recurring failures in '{task_type}' task"

                # AFTER:
                proposal = GoalProposal(
                    text=f"Alert user: '{task_type}' task failing repeatedly ({count} failures)",
                    category=GoalCategory.USER_REQUEST,  # Changed!
                    pattern_detected="task_failure_recurring",
                    evidence={
                        "task_type": task_type,
                        "failure_count": count,
                        "error_type": self._identify_error_pattern(failures),
                        "user_action_required": True  # New field!
                    },
                    # ... rest of fields
                )
                proposals.append(proposal)

        return proposals
```

### 3. HTN Planner Decomposition

Update planner to handle USER_REQUEST goals:

```python
def decompose_goal(self, goal: GoalDefinition) -> List[TaskDefinition]:
    if goal.category == GoalCategory.USER_REQUEST:
        return [
            TaskDefinition(
                type=TaskType.CUSTOM,
                prompt=f"Analyze this issue and create a detailed report: {goal.text}"
            ),
            TaskDefinition(
                type=TaskType.CUSTOM,
                prompt="Save findings to persona_space/learning_journey/issue_reports/"
            ),
            # Could also create a notification mechanism
        ]

    elif goal.category == GoalCategory.INSIGHT:
        return [
            TaskDefinition(
                type=TaskType.SELF_REFLECTION,
                prompt=f"Reflect on patterns related to: {goal.text}"
            ),
            TaskDefinition(
                type=TaskType.CUSTOM,
                prompt="Document insights in persona_space/learning_journey/"
            ),
        ]

    # ... other categories
```

## Alternative: Hybrid Approach

**System creates TWO types of goals:**

1. **Actionable by Astra** (insight, knowledge gathering)
   ```
   "Understand backup failure patterns" → She can execute
   ```

2. **Actionable by User** (code changes, system fixes)
   ```
   "Fix backup timeout issue" → Marked as user_action_required
   ```

**Both feed into unified goal pool, but decompose differently.**

## Impact on Current Implementation

### Files to Modify:

1. **src/services/goal_store.py**
   ```python
   # Add to GoalCategory
   USER_REQUEST = "user_request"
   INSIGHT = "insight"
   KNOWLEDGE = "knowledge"

   # Add to GoalDefinition
   user_action_required: bool = False
   ```

2. **src/services/detectors/task_failure_detector.py**
   ```python
   # Change goal framing
   text = f"Alert user: '{task_type}' task failing ({count} failures)"
   category = GoalCategory.USER_REQUEST
   ```

3. **HTN Planner decomposition logic**
   ```python
   # Add handlers for new goal categories
   # Don't try to decompose USER_REQUEST into code changes
   ```

4. **Add notification mechanism**
   ```python
   # When USER_REQUEST goal created:
   # - Log prominently
   # - Maybe email/webhook user
   # - Create issue report in persona_space/
   ```

## Testing the Fix

### Before Fix:
```bash
python3 scripts/trigger_goal_detection.py
# Creates: "Fix recurring failures in 'daily_backup' task"
# HTN Planner tries to decompose → FAILS (can't access code)
```

### After Fix:
```bash
python3 scripts/trigger_goal_detection.py
# Creates: "Alert user: 'daily_backup' task failing (4 failures)"
# HTN Planner decomposes to:
#   1. Analyze failure patterns
#   2. Generate report
#   3. Save to persona_space/learning_journey/issue_reports/backup_failures.md
#   4. Log prominently for user
# ✓ SUCCESS - All tasks executable by Astra
```

## Philosophical Question

**Should Astra be able to modify her own code?**

This is a fundamental design decision:

### Arguments FOR Code Access:
- True autonomy
- Can fix her own bugs
- Self-improving system
- Aligns with "autonomous agent" concept

### Arguments AGAINST Code Access:
- Security risk (she could break herself)
- Unpredictable behavior
- Hard to debug/rollback
- Violates separation of concerns
- User loses control

**Current recommendation: Keep code access OFF, use user-request pattern.**

## Immediate Action Required

1. ✅ Acknowledge this issue in docs
2. ⚠️ Update goal categories
3. ⚠️ Fix TaskFailureDetector framing
4. ⚠️ Add user notification mechanism
5. ⚠️ Update HTN Planner decomposition
6. ⚠️ Test end-to-end flow

Without these fixes, **the goal generation system will create unexecutable goals.**
