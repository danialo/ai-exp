"""Tests for CodeAccessService."""

import pytest
from pathlib import Path
from src.services.code_access import (
    CodeAccessService,
    ModificationStatus,
)


@pytest.fixture
def code_access():
    """Create CodeAccessService for testing."""
    return CodeAccessService(
        project_root=Path("/home/d/git/ai-exp"),
        max_file_size_kb=100,
        auto_branch=False,  # Don't auto-create branches in tests
    )


def test_can_access_allowed_path(code_access):
    """Test that allowed paths are accessible."""
    can_access, reason = code_access.can_access("src/services/goal_store.py")
    assert can_access is True
    assert reason is None


def test_can_access_forbidden_path(code_access):
    """Test that forbidden paths are blocked."""
    can_access, reason = code_access.can_access("config/settings.py")
    assert can_access is False
    assert "forbidden_path" in reason


def test_can_access_env_file(code_access):
    """Test that .env files are blocked."""
    can_access, reason = code_access.can_access(".env")
    assert can_access is False


def test_can_access_app_py(code_access):
    """Test that app.py is blocked."""
    can_access, reason = code_access.can_access("app.py")
    assert can_access is False


def test_can_access_persona_space(code_access):
    """Test that persona_space is blocked."""
    can_access, reason = code_access.can_access("persona_space/beliefs/test.md")
    assert can_access is False


def test_can_access_tests(code_access):
    """Test that test files are accessible."""
    can_access, reason = code_access.can_access("tests/test_goal_store.py")
    assert can_access is True


def test_can_access_not_in_allowed(code_access):
    """Test that paths not in allowed list are blocked."""
    can_access, reason = code_access.can_access("some/random/file.py")
    assert can_access is False
    assert "not_in_allowed_paths" in reason


@pytest.mark.asyncio
async def test_read_file_success(code_access):
    """Test reading an allowed file."""
    content, error = await code_access.read_file("src/services/code_access.py")
    assert error is None
    assert content is not None
    assert "CodeAccessService" in content


@pytest.mark.asyncio
async def test_read_file_forbidden(code_access):
    """Test reading a forbidden file fails."""
    content, error = await code_access.read_file("config/settings.py")
    assert content is None
    assert "access_denied" in error


@pytest.mark.asyncio
async def test_read_file_not_exists(code_access):
    """Test reading non-existent file."""
    content, error = await code_access.read_file("src/services/nonexistent.py")
    assert content is None
    assert "file_not_found" in error or "read_error" in error


def test_modification_tracking(code_access):
    """Test that modifications are tracked."""
    assert len(code_access.list_modifications()) == 0


def test_get_modification_not_found(code_access):
    """Test getting non-existent modification."""
    mod = code_access.get_modification("nonexistent")
    assert mod is None
