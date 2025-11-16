"""Project management for Astra's autonomous workspace.

This service creates and manages self-contained projects in Astra's workspace
(/home/d/astra-workspace/) for autonomous coding tasks.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
import os
import json
import shutil
import hashlib
import logging

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = "/home/d/astra-workspace"
PROJECTS_DIR = f"{WORKSPACE_ROOT}/projects"
TEMPLATES_DIR = f"{WORKSPACE_ROOT}/templates"
LOGS_DIR = f"{WORKSPACE_ROOT}/logs"


@dataclass
class ProjectMetadata:
    """Metadata for a workspace project."""
    project_id: str
    project_dir: str
    goal_text: str
    created_at: str
    status: str  # "created", "in_progress", "completed", "failed"
    template: str
    main_file: Optional[str] = None
    test_file: Optional[str] = None
    test_results: Optional[Dict[str, Any]] = None
    implementation_notes: str = ""
    next_steps: str = ""
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class ProjectManager:
    """Manages autonomous coding projects in Astra's workspace."""

    def __init__(self, workspace_root: str = WORKSPACE_ROOT):
        """Initialize project manager.

        Args:
            workspace_root: Root directory for Astra's workspace
        """
        self.workspace_root = workspace_root
        self.projects_dir = f"{workspace_root}/projects"
        self.templates_dir = f"{workspace_root}/templates"

        # Ensure workspace exists
        os.makedirs(self.projects_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)

        logger.info(f"ProjectManager initialized with workspace at {workspace_root}")

    def create_project(
        self,
        goal_text: str,
        template: str = "python-module",
        description: str = ""
    ) -> ProjectMetadata:
        """Create a new project in the workspace.

        Args:
            goal_text: The goal this project aims to achieve
            template: Project template to use (default: python-module)
            description: Optional description of what to build

        Returns:
            ProjectMetadata for the created project

        Raises:
            FileExistsError: If project already exists
            ValueError: If template doesn't exist
        """
        # Generate project ID from goal text and timestamp
        timestamp = datetime.utcnow()
        timestamp_str = timestamp.strftime("%Y%m%d-%H%M%S")
        goal_hash = hashlib.sha256(goal_text.encode()).hexdigest()[:8]
        project_id = f"proj-{timestamp_str}-{goal_hash}"

        project_dir = os.path.join(self.projects_dir, project_id)

        # Check if project already exists
        if os.path.exists(project_dir):
            logger.warning(f"Project {project_id} already exists")
            return self.load_project(project_id)

        # Verify template exists
        template_dir = os.path.join(self.templates_dir, template)
        if not os.path.exists(template_dir):
            raise ValueError(f"Template '{template}' not found at {template_dir}")

        # Create project directory structure
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(f"{project_dir}/src", exist_ok=True)
        os.makedirs(f"{project_dir}/tests", exist_ok=True)
        os.makedirs(f"{project_dir}/docs", exist_ok=True)

        # Create metadata
        metadata = ProjectMetadata(
            project_id=project_id,
            project_dir=project_dir,
            goal_text=goal_text,
            created_at=timestamp.isoformat(),
            status="created",
            template=template,
            implementation_notes=description
        )

        # Save metadata
        self._save_metadata(metadata)

        # Create README from template
        self._create_readme(metadata)

        # Create run script
        self._create_run_script(metadata)

        logger.info(f"Created project {project_id} at {project_dir}")
        return metadata

    def update_project(
        self,
        project_id: str,
        status: Optional[str] = None,
        main_file: Optional[str] = None,
        test_file: Optional[str] = None,
        test_results: Optional[Dict[str, Any]] = None,
        implementation_notes: Optional[str] = None,
        next_steps: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> ProjectMetadata:
        """Update project metadata.

        Args:
            project_id: Project identifier
            status: New status (created, in_progress, completed, failed)
            main_file: Main implementation file path
            test_file: Test file path
            test_results: Test execution results
            implementation_notes: Notes about implementation
            next_steps: Next steps or follow-up work
            error_message: Error message if failed

        Returns:
            Updated ProjectMetadata

        Raises:
            FileNotFoundError: If project doesn't exist
        """
        metadata = self.load_project(project_id)

        # Update fields if provided
        if status is not None:
            metadata.status = status
            if status in ("completed", "failed"):
                metadata.completed_at = datetime.utcnow().isoformat()
        if main_file is not None:
            metadata.main_file = main_file
        if test_file is not None:
            metadata.test_file = test_file
        if test_results is not None:
            metadata.test_results = test_results
        if implementation_notes is not None:
            metadata.implementation_notes = implementation_notes
        if next_steps is not None:
            metadata.next_steps = next_steps
        if error_message is not None:
            metadata.error_message = error_message

        # Save updated metadata
        self._save_metadata(metadata)

        # Update README
        self._create_readme(metadata)

        logger.info(f"Updated project {project_id}: status={metadata.status}")
        return metadata

    def load_project(self, project_id: str) -> ProjectMetadata:
        """Load project metadata.

        Args:
            project_id: Project identifier

        Returns:
            ProjectMetadata

        Raises:
            FileNotFoundError: If project doesn't exist
        """
        project_dir = os.path.join(self.projects_dir, project_id)
        metadata_path = os.path.join(project_dir, ".project.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Project {project_id} not found at {project_dir}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return ProjectMetadata(**data)

    def list_projects(
        self,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[ProjectMetadata]:
        """List projects in workspace.

        Args:
            status: Filter by status (optional)
            limit: Maximum number of projects to return

        Returns:
            List of ProjectMetadata, sorted by created_at descending
        """
        projects = []

        if not os.path.exists(self.projects_dir):
            return projects

        # List all project directories
        for project_id in os.listdir(self.projects_dir):
            try:
                metadata = self.load_project(project_id)
                if status is None or metadata.status == status:
                    projects.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load project {project_id}: {e}")

        # Sort by created_at descending
        projects.sort(key=lambda p: p.created_at, reverse=True)

        return projects[:limit]

    def get_project_path(self, project_id: str, subdir: str = "") -> str:
        """Get absolute path to project directory or subdirectory.

        Args:
            project_id: Project identifier
            subdir: Optional subdirectory (e.g., "src", "tests")

        Returns:
            Absolute path to project (sub)directory
        """
        project_dir = os.path.join(self.projects_dir, project_id)
        if subdir:
            return os.path.join(project_dir, subdir)
        return project_dir

    def delete_project(self, project_id: str) -> None:
        """Delete a project from workspace.

        Args:
            project_id: Project identifier

        Raises:
            FileNotFoundError: If project doesn't exist
        """
        project_dir = os.path.join(self.projects_dir, project_id)

        if not os.path.exists(project_dir):
            raise FileNotFoundError(f"Project {project_id} not found")

        shutil.rmtree(project_dir)
        logger.info(f"Deleted project {project_id}")

    def _save_metadata(self, metadata: ProjectMetadata) -> None:
        """Save project metadata to .project.json."""
        metadata_path = os.path.join(metadata.project_dir, ".project.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=2)

    def _create_readme(self, metadata: ProjectMetadata) -> None:
        """Create or update project README from template."""
        template_path = os.path.join(self.templates_dir, metadata.template, "README.template.md")
        readme_path = os.path.join(metadata.project_dir, "README.md")

        # Try to load template, fallback to basic README
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()
        else:
            template = self._default_readme_template()

        # Replace placeholders
        content = template.replace("{{PROJECT_NAME}}", metadata.project_id)
        content = content.replace("{{CREATED_DATE}}", metadata.created_at)
        content = content.replace("{{GOAL_TEXT}}", metadata.goal_text)
        content = content.replace("{{STATUS}}", metadata.status)
        content = content.replace("{{DESCRIPTION}}", metadata.implementation_notes or "TBD")
        content = content.replace("{{PROJECT_DIR}}", metadata.project_dir)
        content = content.replace("{{MAIN_FILE}}", metadata.main_file or "main.py")
        content = content.replace("{{IMPLEMENTATION_NOTES}}", metadata.implementation_notes or "TBD")
        content = content.replace("{{NEXT_STEPS}}", metadata.next_steps or "TBD")

        # Format test results
        if metadata.test_results:
            test_results_str = f"- Status: {metadata.test_results.get('status', 'unknown')}\n"
            test_results_str += f"- Passed: {metadata.test_results.get('passed', 0)}\n"
            test_results_str += f"- Failed: {metadata.test_results.get('failed', 0)}\n"
        else:
            test_results_str = "No tests run yet"
        content = content.replace("{{TEST_RESULTS}}", test_results_str)

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _create_run_script(self, metadata: ProjectMetadata) -> None:
        """Create a basic run script for the project."""
        script_path = os.path.join(metadata.project_dir, "run.sh")

        script = f"""#!/bin/bash
# Run script for {metadata.project_id}
set -e

echo "Running {metadata.project_id}..."
echo "Goal: {metadata.goal_text}"
echo ""

# Run implementation
if [ -f "src/{metadata.main_file or 'main.py'}" ]; then
    echo "Running implementation..."
    python3 "src/{metadata.main_file or 'main.py'}"
fi

# Run tests
if [ -d "tests" ] && [ "$(ls -A tests)" ]; then
    echo ""
    echo "Running tests..."
    python3 -m pytest tests/ -v
fi

echo ""
echo "Done!"
"""

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        # Make executable
        os.chmod(script_path, 0o755)

    def _default_readme_template(self) -> str:
        """Default README template if template file not found."""
        return """# {{PROJECT_NAME}}

**Created**: {{CREATED_DATE}}
**Goal**: {{GOAL_TEXT}}
**Status**: {{STATUS}}

## Overview

{{DESCRIPTION}}

## Usage

```bash
cd {{PROJECT_DIR}}
python src/{{MAIN_FILE}}
```

## Tests

```bash
pytest tests/ -v
```

## Results

{{TEST_RESULTS}}

---

*Autonomously created by Astra*
"""
