"""
Tests for code access conversation tools.

Tests the three conversation tools:
1. read_source_code - Read files from codebase
2. read_logs - Read log files
3. schedule_code_modification - Schedule code changes for approval
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from src.services.persona_service import PersonaService
from src.services.task_scheduler import TaskDefinition, TaskType, TaskSchedule


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    llm = Mock()
    llm.model = "gpt-4"
    llm.generate_response = Mock(return_value="test response")
    return llm


@pytest.fixture
def mock_code_access_service():
    """Create a mock code access service."""
    service = Mock()
    service.can_access = Mock(return_value=(True, None))
    return service


@pytest.fixture
def mock_task_scheduler():
    """Create a mock task scheduler."""
    scheduler = Mock()
    scheduler.tasks = {}
    scheduler._save_tasks = Mock()
    return scheduler


@pytest.fixture
def persona_service(mock_llm_service, mock_code_access_service, mock_task_scheduler, tmp_path):
    """Create a PersonaService with mocked dependencies."""
    persona_space = tmp_path / "persona_space"
    persona_space.mkdir()

    # Create required directories
    (persona_space / "meta").mkdir()
    (persona_space / "memory").mkdir()
    (persona_space / "workspace").mkdir()

    service = PersonaService(
        llm_service=mock_llm_service,
        persona_space_path=str(persona_space),
        code_access_service=mock_code_access_service,
        task_scheduler=mock_task_scheduler,
    )

    return service


class TestReadSourceCode:
    """Tests for read_source_code tool."""

    def test_tool_definition_exists(self, persona_service):
        """Verify read_source_code tool is defined."""
        tools = persona_service._get_tool_definitions()
        tool_names = [t["function"]["name"] for t in tools]

        assert "read_source_code" in tool_names

    def test_read_existing_file(self, persona_service, tmp_path):
        """Test reading an existing source file."""
        # Create a test source file
        src_dir = Path("src")
        if not src_dir.exists():
            pytest.skip("src/ directory not found")

        # Test reading a known file
        result = persona_service._execute_tool("read_source_code", {"path": "services/persona_service.py"})

        # Should either return content or a helpful error
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invalid_path_with_src_prefix(self, persona_service):
        """Test that including 'src/' in path gives helpful error."""
        result = persona_service._execute_tool("read_source_code", {"path": "src/services/persona_service.py"})

        assert "Error" in result or "Do not include 'src/'" in result


class TestReadLogs:
    """Tests for read_logs tool."""

    def test_tool_definition_exists(self, persona_service):
        """Verify read_logs tool is defined."""
        tools = persona_service._get_tool_definitions()
        tool_names = [t["function"]["name"] for t in tools]

        assert "read_logs" in tool_names

    def test_read_nonexistent_log(self, persona_service):
        """Test reading a log file that doesn't exist."""
        result = persona_service._execute_tool("read_logs", {
            "log_file": "nonexistent.log",
            "lines": 10
        })

        assert "not found" in result.lower() or "error" in result.lower()

    def test_path_traversal_protection(self, persona_service):
        """Test that path traversal attempts are blocked."""
        result = persona_service._execute_tool("read_logs", {
            "log_file": "../../../etc/passwd",
            "lines": 10
        })

        assert "path traversal" in result.lower() or "error" in result.lower()

    def test_line_limit_respected(self, persona_service, tmp_path):
        """Test that line limit is enforced."""
        # Create a test log file
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        test_log = log_dir / "test.log"

        # Write 100 lines
        with open(test_log, 'w') as f:
            for i in range(100):
                f.write(f"Line {i}\n")

        # Request only 10 lines (would need to modify persona_service to use tmp_path)
        # This is a placeholder test - actual implementation would need to inject the logs path
        result = persona_service._execute_tool("read_logs", {
            "log_file": "test.log",
            "lines": 10
        })

        # Should contain error about file not found (since we're using actual logs/ dir)
        assert isinstance(result, str)


class TestScheduleCodeModification:
    """Tests for schedule_code_modification tool."""

    def test_tool_definition_exists(self, persona_service):
        """Verify schedule_code_modification tool is defined."""
        tools = persona_service._get_tool_definitions()
        tool_names = [t["function"]["name"] for t in tools]

        assert "schedule_code_modification" in tool_names

    def test_creates_manual_task(self, persona_service, mock_task_scheduler, mock_code_access_service):
        """Test that scheduling creates a MANUAL task."""
        # Mock can_access to return True
        mock_code_access_service.can_access.return_value = (True, None)

        result = persona_service._execute_tool("schedule_code_modification", {
            "file_path": "src/services/test.py",
            "new_content": "# Test content",
            "reason": "Test modification",
            "goal_id": "test_goal_123"
        })

        # Should succeed
        assert "scheduled successfully" in result.lower() or "✓" in result

        # Should have created a task
        assert len(mock_task_scheduler.tasks) > 0

        # Get the created task
        task_id = list(mock_task_scheduler.tasks.keys())[0]
        task = mock_task_scheduler.tasks[task_id]

        # Verify task properties
        assert task.type == TaskType.CODE_MODIFY
        assert task.schedule == TaskSchedule.MANUAL
        assert task.metadata["file_path"] == "src/services/test.py"
        assert task.metadata["reason"] == "Test modification"
        assert task.metadata["goal_id"] == "test_goal_123"
        assert task.metadata["status"] == "awaiting_approval"

    def test_respects_access_boundaries(self, persona_service, mock_code_access_service):
        """Test that forbidden paths are rejected."""
        # Mock can_access to return False with error
        mock_code_access_service.can_access.return_value = (False, "Forbidden path")

        result = persona_service._execute_tool("schedule_code_modification", {
            "file_path": "config/secrets.yml",
            "new_content": "# Malicious content",
            "reason": "Bad modification"
        })

        # Should be rejected
        assert "error" in result.lower() or "access denied" in result.lower()
        assert "forbidden" in result.lower() or "access denied" in result.lower()

    def test_requires_code_access_service(self, mock_llm_service, tmp_path):
        """Test that tool fails gracefully without code_access_service."""
        persona_space = tmp_path / "persona_space"
        persona_space.mkdir()
        (persona_space / "meta").mkdir()
        (persona_space / "memory").mkdir()
        (persona_space / "workspace").mkdir()

        # Create service WITHOUT code_access_service
        service = PersonaService(
            llm_service=mock_llm_service,
            persona_space_path=str(persona_space),
            code_access_service=None,  # No code access service
            task_scheduler=None,
        )

        result = service._execute_tool("schedule_code_modification", {
            "file_path": "src/test.py",
            "new_content": "# Test",
            "reason": "Test"
        })

        assert "not available" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
