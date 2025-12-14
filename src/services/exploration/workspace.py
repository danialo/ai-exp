"""Workspace manager for sandboxed exploration jobs."""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional, List

from .models import WorkspacePolicy

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Manages isolated workspaces for exploration jobs."""

    def __init__(self, base_dir: str = "persona_space/workspaces"):
        """Initialize workspace manager.

        Args:
            base_dir: Base directory for all workspaces
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_workspace(self, job_id: str, policy: WorkspacePolicy) -> Path:
        """Create a new workspace for a job.

        Args:
            job_id: Unique job identifier
            policy: Workspace access policy

        Returns:
            Path to the workspace directory
        """
        workspace_dir = self.base_dir / job_id
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (workspace_dir / ".tmp").mkdir(exist_ok=True)
        (workspace_dir / "artifacts").mkdir(exist_ok=True)

        logger.info(f"Created workspace: {workspace_dir}")
        return workspace_dir

    def cleanup_workspace(self, job_id: str, keep_artifacts: bool = True):
        """Clean up a workspace after job completion.

        Args:
            job_id: Job identifier
            keep_artifacts: If True, only remove .tmp, keep artifacts
        """
        workspace_dir = self.base_dir / job_id

        if not workspace_dir.exists():
            return

        if keep_artifacts:
            # Only remove temp directory
            tmp_dir = workspace_dir / ".tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            logger.info(f"Cleaned temp files from workspace: {job_id}")
        else:
            # Remove entire workspace
            shutil.rmtree(workspace_dir)
            logger.info(f"Removed workspace: {job_id}")

    def is_path_allowed(
        self,
        path: Path,
        workspace_dir: Path,
        policy: WorkspacePolicy,
        write: bool = False
    ) -> bool:
        """Check if a path operation is allowed by the workspace policy.

        Args:
            path: Path to check
            workspace_dir: Root of the workspace
            policy: Workspace policy
            write: If True, check write permissions; if False, check read

        Returns:
            True if the operation is allowed
        """
        try:
            # Resolve to absolute path
            abs_path = path.resolve()
            abs_workspace = workspace_dir.resolve()

            # Check if path is within workspace
            try:
                abs_path.relative_to(abs_workspace)
                # Inside workspace - always allowed
                return True
            except ValueError:
                # Outside workspace
                pass

            # If write operation and not allowed outside
            if write and not policy.allow_write_outside:
                # Check allowlist
                for allowed_prefix in policy.extra_write_allowlist:
                    allowed_path = Path(allowed_prefix).resolve()
                    try:
                        abs_path.relative_to(allowed_path)
                        return True
                    except ValueError:
                        continue
                return False

            # Read operations outside workspace are allowed to persona_space
            if not write:
                persona_space = Path("persona_space").resolve()
                try:
                    abs_path.relative_to(persona_space)
                    return True
                except ValueError:
                    return False

            return False

        except Exception as e:
            logger.error(f"Error checking path permissions: {e}")
            return False

    def safe_write_file(
        self,
        path: str,
        content: str,
        workspace_dir: Path,
        policy: WorkspacePolicy
    ) -> bool:
        """Safely write a file within workspace constraints.

        Args:
            path: Relative or absolute path to write
            content: File content
            workspace_dir: Workspace root
            policy: Workspace policy

        Returns:
            True if write succeeded, False if blocked
        """
        try:
            # Convert to Path relative to workspace
            if os.path.isabs(path):
                file_path = Path(path)
            else:
                file_path = workspace_dir / path

            # Check permissions
            if not self.is_path_allowed(file_path, workspace_dir, policy, write=True):
                logger.warning(f"Write blocked by policy: {file_path}")
                return False

            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            file_path.write_text(content, encoding='utf-8')
            logger.info(f"Wrote file: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error writing file {path}: {e}")
            return False

    def safe_read_file(
        self,
        path: str,
        workspace_dir: Path,
        policy: WorkspacePolicy
    ) -> Optional[str]:
        """Safely read a file within workspace constraints.

        Args:
            path: Relative or absolute path to read
            workspace_dir: Workspace root
            policy: Workspace policy

        Returns:
            File content or None if blocked/failed
        """
        try:
            # Convert to Path
            if os.path.isabs(path):
                file_path = Path(path)
            else:
                file_path = workspace_dir / path

            # Check permissions
            if not self.is_path_allowed(file_path, workspace_dir, policy, write=False):
                logger.warning(f"Read blocked by policy: {file_path}")
                return None

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return None

            # Read file
            content = file_path.read_text(encoding='utf-8')
            logger.info(f"Read file: {file_path}")
            return content

        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            return None

    def list_artifacts(self, job_id: str) -> List[str]:
        """List all artifacts in the workspace artifacts directory.

        Args:
            job_id: Job identifier

        Returns:
            List of artifact paths relative to artifacts directory
        """
        artifacts_dir = self.base_dir / job_id / "artifacts"

        if not artifacts_dir.exists():
            return []

        artifacts = []
        for file_path in artifacts_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(artifacts_dir)
                artifacts.append(str(rel_path))

        return artifacts


def create_workspace_manager(base_dir: str = "persona_space/workspaces") -> WorkspaceManager:
    """Factory function to create a WorkspaceManager instance.

    Args:
        base_dir: Base directory for workspaces

    Returns:
        Configured WorkspaceManager instance
    """
    return WorkspaceManager(base_dir=base_dir)
