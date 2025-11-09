# Code Access System Design

## Goal

Enable Astra to read and modify source code to fix issues she detects, while maintaining safety and user control.

## Core Principles

1. **Git Branch Isolation** - All code changes happen in isolated branches
2. **Approval Required** - No changes merge without user approval
3. **Easy Rollback** - Can undo any change instantly
4. **Clear Boundaries** - Some files are off-limits
5. **Audit Trail** - Every code change logged to identity ledger
6. **Test Before Deploy** - Run tests before allowing merge

## Architecture

### New Task Types

```python
class TaskType(str, Enum):
    # Existing cognitive tasks
    SELF_REFLECTION = "self_reflection"
    GOAL_ASSESSMENT = "goal_assessment"
    # ... etc

    # NEW: Code access tasks
    CODE_READ = "code_read"              # Read source files
    CODE_ANALYZE = "code_analyze"        # Analyze code patterns
    CODE_MODIFY = "code_modify"          # Modify source files
    CODE_TEST = "code_test"              # Run tests
    CODE_REVIEW = "code_review"          # Review changes before commit
```

### Code Access Service

```python
class CodeAccessService:
    """Manages safe code reading and modification."""

    def __init__(
        self,
        project_root: Path,
        allowed_paths: List[str],
        forbidden_paths: List[str],
        approval_required: bool = True,
        auto_branch: bool = True,
    ):
        self.project_root = project_root
        self.allowed_paths = allowed_paths  # e.g., ["src/", "tests/"]
        self.forbidden_paths = forbidden_paths  # e.g., ["config/", ".env"]
        self.approval_required = approval_required
        self.auto_branch = auto_branch

    def can_access(self, file_path: str) -> bool:
        """Check if file is within allowed boundaries."""
        pass

    async def read_file(self, file_path: str) -> str:
        """Read source file with permission check."""
        pass

    async def modify_file(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
        reason: str,
    ) -> CodeModification:
        """Modify file in isolated git branch."""
        pass

    async def create_modification_branch(self, goal_id: str) -> str:
        """Create isolated branch for modifications."""
        pass

    async def run_tests(self, branch: str) -> TestResult:
        """Run test suite on branch."""
        pass

    async def request_approval(self, modification: CodeModification) -> ApprovalRequest:
        """Create approval request for user."""
        pass

    async def merge_modification(self, modification_id: str) -> bool:
        """Merge approved modification."""
        pass

    async def rollback_modification(self, modification_id: str) -> bool:
        """Rollback a modification."""
        pass
```

### Data Models

```python
@dataclass
class CodeModification:
    """Represents a code modification."""
    id: str  # modification_xyz
    goal_id: str  # Which goal triggered this
    branch_name: str  # astra/fix-backup-timeout
    files_modified: List[str]
    reason: str  # Why this change was made
    diff: str  # Git diff
    created_at: datetime
    status: ModificationStatus
    test_results: Optional[TestResult] = None
    approval_request_id: Optional[str] = None

class ModificationStatus(str, Enum):
    PENDING = "pending"          # Code modified, awaiting tests
    TESTING = "testing"          # Running tests
    AWAITING_APPROVAL = "awaiting_approval"  # Tests passed, needs user approval
    APPROVED = "approved"        # User approved
    MERGED = "merged"           # Merged to main
    REJECTED = "rejected"       # User rejected
    ROLLED_BACK = "rolled_back"  # Was merged but then rolled back

@dataclass
class TestResult:
    """Test execution results."""
    passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    failures: List[str]
    output: str
    duration_seconds: float

@dataclass
class ApprovalRequest:
    """Request for user to approve code change."""
    id: str
    modification_id: str
    goal_id: str
    summary: str  # Human-readable summary of changes
    diff: str
    test_results: TestResult
    created_at: datetime
    status: ApprovalStatus
    responded_at: Optional[datetime] = None
    response: Optional[str] = None  # approve/reject/request_changes

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"
```

## Safety Boundaries

### Allowed Paths (She CAN modify)
- `src/services/` - Service layer
- `src/pipeline/` - Pipeline components
- `src/utils/` - Utility functions
- `tests/` - Test files
- `scripts/` - Utility scripts
- `docs/` - Documentation

### Forbidden Paths (She CANNOT modify)
- `config/` - Configuration files
- `.env*` - Environment files
- `app.py` - Main application (too risky)
- `persona_space/` - Her own space (separate workflow)
- `.git/` - Git metadata
- `venv/` - Virtual environment

### File Size Limits
- Max file size to read: 100KB
- Max file size to modify: 50KB
- If larger, must request user assistance

## Workflow: Code Modification Flow

### Step 1: Goal Created
```
TaskFailureDetector detects: "daily_backup task failing"
Creates goal: "Fix recurring failures in 'daily_backup' task"
Goal includes: evidence, confidence, estimated effort
```

### Step 2: HTN Planner Decomposes Goal
```python
decompose_goal(goal) → [
    Task(type=CODE_READ, params={
        "files": ["src/services/task_scheduler.py"],
        "focus": "daily_backup task definition and execution"
    }),
    Task(type=CODE_ANALYZE, params={
        "analysis": "Identify why timeouts are occurring",
        "evidence": goal.metadata['evidence']
    }),
    Task(type=CODE_MODIFY, params={
        "file": "src/services/task_scheduler.py",
        "change": "Increase timeout or optimize backup logic",
        "reason": "Fix timeout errors causing backup failures"
    }),
    Task(type=CODE_TEST, params={
        "test_pattern": "test_task_scheduler*"
    }),
]
```

### Step 3: Task Execution

#### Task 1: CODE_READ
```python
# Executor calls CodeAccessService
result = await code_access.read_file("src/services/task_scheduler.py")

# LLM receives file content in prompt
prompt = f"""
You are analyzing this file to understand backup task failures.

File: src/services/task_scheduler.py
{result.content}

Task: Find the daily_backup task definition and understand why it's timing out.
Evidence: {evidence}

Provide your analysis.
"""

# LLM responds with analysis
analysis = await llm_service.generate(prompt)
# "The backup task has a 30-second timeout but the backup process
#  typically takes 35-40 seconds, causing repeated failures."
```

#### Task 2: CODE_ANALYZE
```python
# LLM proposes solution based on analysis
prompt = f"""
Based on this analysis:
{analysis}

Propose a code change to fix this issue.
Provide:
1. Exact file to modify
2. Exact lines to change
3. New code
4. Reasoning
"""

solution = await llm_service.generate(prompt)
```

#### Task 3: CODE_MODIFY
```python
# Create isolated branch
branch = await code_access.create_modification_branch(goal_id)
# → "astra/fix-backup-timeout-goal_8f2e4a9c"

# Apply modification
modification = await code_access.modify_file(
    file_path="src/services/task_scheduler.py",
    old_content=original_lines,
    new_content=modified_lines,
    reason="Increase backup task timeout from 30s to 60s",
)

# Commit change
git add src/services/task_scheduler.py
git commit -m "Fix: Increase daily_backup task timeout to 60s

The backup task was timing out after 30 seconds, but backup process
typically takes 35-40 seconds. Increased timeout to 60s to prevent
failures.

Goal: goal_8f2e4a9c
Modification: modification_xyz
Pattern detected: task_failure_recurring (4 failures in 24h)"
```

#### Task 4: CODE_TEST
```python
# Switch to modification branch
git checkout astra/fix-backup-timeout-goal_8f2e4a9c

# Run tests
test_result = await code_access.run_tests(branch)

if test_result.passed:
    # Tests passed - request approval
    approval = await code_access.request_approval(modification)
else:
    # Tests failed - mark modification as failed
    modification.status = ModificationStatus.REJECTED
    modification.test_results = test_result
    # Could create new goal to fix the tests
```

### Step 4: Approval Request Created

**Stored in:** `data/approval_requests/modification_xyz.json`

```json
{
  "id": "approval_abc123",
  "modification_id": "modification_xyz",
  "goal_id": "goal_8f2e4a9c",
  "summary": "Fix daily_backup task timeout (30s → 60s)",
  "files_modified": ["src/services/task_scheduler.py"],
  "diff": "...",
  "test_results": {
    "passed": true,
    "total_tests": 127,
    "passed_tests": 127,
    "failed_tests": 0
  },
  "branch": "astra/fix-backup-timeout-goal_8f2e4a9c",
  "status": "pending",
  "created_at": "2025-11-09T14:30:00Z"
}
```

**User notification:**
```
🤖 Astra has proposed a code change:

Goal: Fix recurring failures in 'daily_backup' task
Branch: astra/fix-backup-timeout-goal_8f2e4a9c
Files: src/services/task_scheduler.py

Change: Increase backup task timeout from 30s to 60s
Reason: Task was timing out after 30s, but backup typically takes 35-40s

Tests: ✅ All 127 tests passing

Review:
  git checkout astra/fix-backup-timeout-goal_8f2e4a9c
  git diff main

Approve:
  curl -X POST http://localhost:8000/api/v1/approvals/approval_abc123/approve

Reject:
  curl -X POST http://localhost:8000/api/v1/approvals/approval_abc123/reject
```

### Step 5: User Approval

**User reviews and approves:**
```bash
curl -X POST http://localhost:8000/api/v1/approvals/approval_abc123/approve
```

**CodeAccessService merges:**
```python
# Switch to main
git checkout main

# Merge modification branch
git merge astra/fix-backup-timeout-goal_8f2e4a9c

# Update modification status
modification.status = ModificationStatus.MERGED

# Log to identity ledger
append_event(LedgerEvent(
    event="code_modified",
    meta={
        "modification_id": "modification_xyz",
        "goal_id": "goal_8f2e4a9c",
        "files": ["src/services/task_scheduler.py"],
        "branch": "astra/fix-backup-timeout-goal_8f2e4a9c",
        "approved_by": "user",
    }
))

# Mark goal as completed
goal_store.update_goal(goal_id, state=GoalState.COMPLETED)
```

### Step 6: Rollback (if needed)

**If change causes issues:**
```bash
curl -X POST http://localhost:8000/api/v1/approvals/modification_xyz/rollback
```

**CodeAccessService reverts:**
```python
# Find merge commit
merge_commit = get_merge_commit(modification.branch_name)

# Revert the merge
git revert merge_commit -m 1

# Update status
modification.status = ModificationStatus.ROLLED_BACK

# Log to ledger
append_event(LedgerEvent(
    event="code_rollback",
    meta={
        "modification_id": "modification_xyz",
        "reason": "User-requested rollback"
    }
))
```

## Identity Ledger Events

### code_modification_requested
```json
{
  "ts": 1699538400.123,
  "event": "code_modification_requested",
  "meta": {
    "modification_id": "modification_xyz",
    "goal_id": "goal_8f2e4a9c",
    "files": ["src/services/task_scheduler.py"],
    "reason": "Fix backup task timeout"
  }
}
```

### code_modification_approved
```json
{
  "ts": 1699538500.456,
  "event": "code_modification_approved",
  "meta": {
    "modification_id": "modification_xyz",
    "approval_id": "approval_abc123",
    "approved_by": "user"
  }
}
```

### code_modification_merged
```json
{
  "ts": 1699538510.789,
  "event": "code_modification_merged",
  "meta": {
    "modification_id": "modification_xyz",
    "branch": "astra/fix-backup-timeout-goal_8f2e4a9c",
    "merge_commit": "a1b2c3d4"
  }
}
```

### code_modification_rolled_back
```json
{
  "ts": 1699538600.123,
  "event": "code_modification_rolled_back",
  "meta": {
    "modification_id": "modification_xyz",
    "reason": "User-requested rollback",
    "revert_commit": "e5f6g7h8"
  }
}
```

## API Endpoints

### List Pending Approvals
```
GET /api/v1/approvals?status=pending
```

### Get Approval Details
```
GET /api/v1/approvals/{approval_id}
```

### Approve Code Change
```
POST /api/v1/approvals/{approval_id}/approve
```

### Reject Code Change
```
POST /api/v1/approvals/{approval_id}/reject
Body: {"reason": "Tests don't cover edge cases"}
```

### Request Changes
```
POST /api/v1/approvals/{approval_id}/request_changes
Body: {"changes": "Add error handling for network failures"}
```

### Rollback Merged Change
```
POST /api/v1/modifications/{modification_id}/rollback
Body: {"reason": "Caused performance regression"}
```

### List All Modifications
```
GET /api/v1/modifications?status=merged
```

## Dashboard / UI

### Approval Queue
```
Pending Code Changes (2)

┌─────────────────────────────────────────────────────────────────┐
│ Fix daily_backup task timeout                                   │
│ Branch: astra/fix-backup-timeout-goal_8f2e4a9c                  │
│ Files: src/services/task_scheduler.py                           │
│ Tests: ✅ 127/127 passing                                        │
│                                                                  │
│ [View Diff] [Approve] [Reject]                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Add error handling to backup task                               │
│ Branch: astra/add-backup-error-handling                         │
│ Files: src/services/task_scheduler.py                           │
│ Tests: ❌ 2/127 failing                                          │
│                                                                  │
│ [View Diff] [View Test Failures]                                │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Read-Only Access
- Implement CODE_READ task type
- CodeAccessService (read-only)
- Test with goal analysis tasks
- **Goal:** Astra can read and analyze code to understand issues

### Phase 2: Isolated Modifications
- Add CODE_MODIFY task type
- Git branch creation/management
- Modification tracking
- **Goal:** Astra can make changes in isolated branches

### Phase 3: Testing Integration
- Add CODE_TEST task type
- Automated test running
- Test result analysis
- **Goal:** Changes are tested before approval

### Phase 4: Approval Workflow
- Approval request system
- API endpoints
- User notifications
- **Goal:** User can review and approve changes

### Phase 5: Merge & Rollback
- Merge approved changes
- Rollback mechanism
- Full audit trail
- **Goal:** Safe deployment with easy undo

### Phase 6: Advanced Features
- Conflict resolution
- Multi-file modifications
- Dependency analysis
- Code review comments
- **Goal:** Handle complex code changes

## Safety Mechanisms

1. **Branch Isolation** - All changes in separate branches
2. **Test Gating** - Tests must pass before approval request
3. **User Approval** - No automatic merges (initially)
4. **Easy Rollback** - One command to undo
5. **File Boundaries** - Cannot touch config/env files
6. **Size Limits** - Cannot modify very large files
7. **Audit Trail** - Every change logged
8. **Rate Limiting** - Max N code modifications per day

## Future Enhancements

- **Confidence-based auto-merge** - High confidence changes auto-merge after tests pass
- **Staged rollout** - Deploy to test environment first
- **A/B testing** - Test both versions in production
- **Automated code review** - LLM reviews its own changes
- **Learning from rollbacks** - Adjust confidence based on which changes get rolled back
- **Collaborative editing** - Astra suggests, user refines, Astra applies

## Questions to Answer

1. Should she be able to create new files or only modify existing?
2. Should modifications require tests to pass, or just run tests and report?
3. Auto-merge for high-confidence + all-tests-passing changes?
4. What happens if user doesn't respond to approval for N days?
5. Should she be able to modify multiple files in one modification?
6. Can she create new test files to test her changes?
7. Should there be a "dry run" mode for testing the workflow?

## Success Metrics

- Number of successful code modifications
- Percentage of modifications that get approved
- Percentage of modifications that get rolled back
- Time from issue detection to fix deployment
- Test pass rate before approval
- User satisfaction with proposed changes
