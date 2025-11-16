"""Code Access Service - Safe code reading and modification for Astra.

Enables Astra to read and modify source code with safety boundaries,
git branch isolation, and approval workflows.
"""

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class ModificationStatus(str, Enum):
    """Status of a code modification."""
    PENDING = "pending"                  # Code modified, awaiting tests
    TESTING = "testing"                  # Running tests
    AWAITING_APPROVAL = "awaiting_approval"  # Tests passed, needs user approval
    APPROVED = "approved"                # User approved
    MERGED = "merged"                    # Merged to main
    REJECTED = "rejected"                # User rejected
    ROLLED_BACK = "rolled_back"          # Was merged but then rolled back


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
class CodeModification:
    """Represents a code modification."""
    id: str
    goal_id: str
    branch_name: str
    files_modified: List[str]
    reason: str
    diff: str
    created_at: datetime
    status: ModificationStatus
    test_results: Optional[TestResult] = None
    approval_request_id: Optional[str] = None
    merged_at: Optional[datetime] = None
    rolled_back_at: Optional[datetime] = None


class CodeAccessService:
    """Manages safe code reading and modification."""

    def __init__(
        self,
        project_root: Path,
        allowed_paths: Optional[List[str]] = None,
        forbidden_paths: Optional[List[str]] = None,
        max_file_size_kb: int = 100,
        auto_branch: bool = True,
    ):
        """Initialize code access service.

        Args:
            project_root: Root directory of the project
            allowed_paths: Paths Astra can access (default: src/, tests/, scripts/, docs/)
            forbidden_paths: Paths Astra cannot access (default: config/, .env*, app.py)
            max_file_size_kb: Maximum file size to read/modify in KB
            auto_branch: Automatically create branches for modifications
        """
        self.project_root = Path(project_root)
        self.max_file_size_bytes = max_file_size_kb * 1024
        self.auto_branch = auto_branch

        # Default allowed paths
        self.allowed_paths = allowed_paths or [
            "src/services/",
            "src/pipeline/",
            "src/utils/",
            "src/memory/",
            "tests/",
            "scripts/",
            "docs/",
            "/home/d/astra-workspace/",  # Astra's autonomous workspace
        ]

        # Default forbidden paths
        self.forbidden_paths = forbidden_paths or [
            "config/",
            ".env",
            ".env.local",
            ".env.production",
            "app.py",  # Main app - too risky
            "persona_space/",  # Her own space - separate workflow
            ".git/",
            "venv/",
            "__pycache__/",
        ]

        # Track modifications
        self.modifications: dict[str, CodeModification] = {}

        logger.info(f"CodeAccessService initialized: project_root={project_root}")
        logger.info(f"Allowed paths: {self.allowed_paths}")
        logger.info(f"Forbidden paths: {self.forbidden_paths}")

    def can_access(self, file_path: str) -> tuple[bool, Optional[str]]:
        """Check if file is within allowed boundaries.

        Args:
            file_path: Relative path from project root

        Returns:
            (can_access, reason) - True if allowed, False with reason if not
        """
        # Check forbidden paths first
        for forbidden in self.forbidden_paths:
            if file_path.startswith(forbidden):
                return False, f"forbidden_path:{forbidden}"

        # Check allowed paths
        for allowed in self.allowed_paths:
            if file_path.startswith(allowed):
                return True, None

        # Not in any allowed path
        return False, "not_in_allowed_paths"

    def check_file_size(self, file_path: Path) -> tuple[bool, Optional[str]]:
        """Check if file size is within limits.

        Args:
            file_path: Absolute path to file

        Returns:
            (within_limits, reason) - True if OK, False with reason if too large
        """
        if not file_path.exists():
            return False, "file_not_found"

        size = file_path.stat().st_size
        if size > self.max_file_size_bytes:
            return False, f"file_too_large:{size}_bytes"

        return True, None

    async def read_file(self, file_path: str) -> tuple[Optional[str], Optional[str]]:
        """Read source file with permission check.

        Args:
            file_path: Relative path from project root

        Returns:
            (content, error) - File content if successful, error message if not
        """
        # Check access
        can_access, reason = self.can_access(file_path)
        if not can_access:
            logger.warning(f"Access denied to {file_path}: {reason}")
            return None, f"access_denied:{reason}"

        # Get absolute path
        abs_path = self.project_root / file_path

        # Check file size
        size_ok, size_reason = self.check_file_size(abs_path)
        if not size_ok:
            logger.warning(f"File size check failed for {file_path}: {size_reason}")
            return None, size_reason

        # Read file
        try:
            content = abs_path.read_text()
            logger.info(f"Read file: {file_path} ({len(content)} chars)")
            return content, None
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return None, f"read_error:{str(e)}"

    async def create_modification_branch(self, goal_id: str) -> tuple[Optional[str], Optional[str]]:
        """Create isolated branch for modifications.

        Args:
            goal_id: ID of the goal triggering this modification

        Returns:
            (branch_name, error) - Branch name if successful, error if not
        """
        # Generate branch name
        branch_name = f"astra/goal-{goal_id[:8]}-{uuid4().hex[:6]}"

        try:
            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
            current_branch = result.stdout.strip()

            # Create new branch
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )

            logger.info(f"Created modification branch: {branch_name} (from {current_branch})")
            return branch_name, None

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create branch: {e.stderr}")
            return None, f"git_error:{e.stderr}"

    async def modify_file(
        self,
        file_path: str,
        new_content: str,
        reason: str,
        goal_id: str,
    ) -> tuple[Optional[CodeModification], Optional[str]]:
        """Modify file in isolated git branch.

        Args:
            file_path: Relative path from project root
            new_content: New file content
            reason: Why this change is being made
            goal_id: ID of goal triggering this

        Returns:
            (modification, error) - CodeModification if successful, error if not
        """
        # Check access
        can_access, access_reason = self.can_access(file_path)
        if not can_access:
            logger.warning(f"Access denied to modify {file_path}: {access_reason}")
            return None, f"access_denied:{access_reason}"

        # Create branch if auto_branch enabled
        branch_name = None
        if self.auto_branch:
            branch_name, branch_error = await self.create_modification_branch(goal_id)
            if branch_error:
                return None, branch_error

        # Get absolute path
        abs_path = self.project_root / file_path

        # Read original content (or empty string if file doesn't exist yet)
        try:
            original_content = abs_path.read_text()
        except FileNotFoundError:
            # File doesn't exist yet - this is a creation, not modification
            original_content = ""
            logger.info(f"Creating new file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to read original content from {file_path}: {e}")
            return None, f"read_error:{str(e)}"

        # Write new content (create parent directories if needed)
        try:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(new_content)
            logger.info(f"Modified file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to write to {file_path}: {e}")
            return None, f"write_error:{str(e)}"

        # Get git diff
        try:
            diff_result = subprocess.run(
                ["git", "diff", file_path],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
            diff = diff_result.stdout
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to get git diff: {e.stderr}")
            diff = "(diff unavailable)"

        # Create modification record
        modification = CodeModification(
            id=f"mod_{uuid4().hex[:8]}",
            goal_id=goal_id,
            branch_name=branch_name or "(no branch)",
            files_modified=[file_path],
            reason=reason,
            diff=diff,
            created_at=datetime.now(timezone.utc),
            status=ModificationStatus.PENDING,
        )

        # Store modification
        self.modifications[modification.id] = modification

        logger.info(f"Created modification {modification.id} for goal {goal_id}")
        return modification, None

    async def commit_modification(
        self,
        modification: CodeModification,
        commit_message: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """Commit modification to git.

        Args:
            modification: The modification to commit
            commit_message: Custom commit message (generated if not provided)

        Returns:
            (success, error) - True if committed, False with error if not
        """
        if not commit_message:
            commit_message = f"""{modification.reason}

Goal: {modification.goal_id}
Modification: {modification.id}
Files: {', '.join(modification.files_modified)}

Generated by Astra's autonomous goal system.
"""

        try:
            # Stage files
            subprocess.run(
                ["git", "add"] + modification.files_modified,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )

            # Commit
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )

            logger.info(f"Committed modification {modification.id}")
            return True, None

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to commit modification: {e.stderr}")
            return False, f"git_error:{e.stderr}"

    async def run_tests(
        self,
        test_pattern: Optional[str] = None,
        timeout_seconds: int = 300,
    ) -> TestResult:
        """Run test suite.

        Args:
            test_pattern: Test pattern to run (default: all tests)
            timeout_seconds: Timeout for test execution

        Returns:
            TestResult with test outcomes
        """
        # Build pytest command
        cmd = ["python3", "-m", "pytest", "-v"]
        if test_pattern:
            cmd.append(test_pattern)

        logger.info(f"Running tests: {' '.join(cmd)}")

        try:
            start_time = datetime.now(timezone.utc)

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Parse pytest output
            output = result.stdout + result.stderr
            passed = result.returncode == 0

            # Extract test counts (simplified - could be more sophisticated)
            # pytest output format: "127 passed in 2.45s"
            total_tests = 0
            passed_tests = 0
            failed_tests = 0
            failures = []

            # TODO: Better parsing of pytest output
            if "passed" in output:
                # Simple extraction
                import re
                match = re.search(r"(\d+) passed", output)
                if match:
                    passed_tests = int(match.group(1))
                    total_tests = passed_tests

            if "failed" in output:
                match = re.search(r"(\d+) failed", output)
                if match:
                    failed_tests = int(match.group(1))
                    total_tests += failed_tests

            return TestResult(
                passed=passed,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                failures=failures,
                output=output,
                duration_seconds=duration,
            )

        except subprocess.TimeoutExpired:
            logger.error(f"Tests timed out after {timeout_seconds}s")
            return TestResult(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                failures=["Test execution timed out"],
                output=f"Tests timed out after {timeout_seconds}s",
                duration_seconds=timeout_seconds,
            )

        except Exception as e:
            logger.error(f"Failed to run tests: {e}")
            return TestResult(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                failures=[str(e)],
                output=f"Test execution failed: {e}",
                duration_seconds=0,
            )

    def get_modification(self, modification_id: str) -> Optional[CodeModification]:
        """Get modification by ID."""
        return self.modifications.get(modification_id)

    def list_modifications(
        self,
        status: Optional[ModificationStatus] = None,
    ) -> List[CodeModification]:
        """List all modifications, optionally filtered by status."""
        mods = list(self.modifications.values())
        if status:
            mods = [m for m in mods if m.status == status]
        return sorted(mods, key=lambda m: m.created_at, reverse=True)


def create_code_access_service(
    project_root: Path,
    **kwargs,
) -> CodeAccessService:
    """Factory function to create CodeAccessService."""
    return CodeAccessService(project_root=project_root, **kwargs)
