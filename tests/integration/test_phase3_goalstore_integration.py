"""Phase 3 Integration Tests: GoalStore + Full Pipeline

Tests the complete autonomous coding pipeline from goal creation to execution.
"""
import pytest
import asyncio
import tempfile
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone

from src.services.goal_store import (
    create_goal_store,
    GoalDefinition,
    GoalCategory,
    GoalState,
    GoalSource
)
from src.services.code_access import create_code_access_service


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def goal_store(temp_db):
    """Create GoalStore for testing."""
    return create_goal_store(temp_db)


@pytest.fixture
def project_root():
    """Project root for testing."""
    return Path("/home/d/git/ai-exp")


@pytest.fixture
def code_access(project_root):
    """CodeAccessService for testing."""
    return create_code_access_service(
        project_root=project_root,
        max_file_size_kb=100,
        auto_branch=False
    )


class TestPhase3GoalStoreCRUD:
    """Phase 3A: GoalStore CRUD operations."""

    def test_create_goal(self, goal_store):
        """Test creating a goal."""
        goal = GoalDefinition(
            id=str(uuid4()),
            text="implement_feature",
            category=GoalCategory.USER_REQUESTED,
            value=0.8,
            effort=0.3,
            risk=0.2,
            horizon_min_min=0,
            horizon_max_min=60
        )

        created = goal_store.create_goal(goal)

        assert created.id == goal.id
        assert created.text == goal.text
        assert created.state == GoalState.PROPOSED

    def test_get_goal(self, goal_store):
        """Test retrieving a goal."""
        goal = GoalDefinition(
            id=str(uuid4()),
            text="test_goal",
            category=GoalCategory.USER_REQUESTED,
            value=0.5,
            effort=0.5,
            risk=0.5,
            horizon_min_min=0
        )

        goal_store.create_goal(goal)
        retrieved = goal_store.get_goal(goal.id)

        assert retrieved is not None
        assert retrieved.id == goal.id

    def test_list_goals(self, goal_store):
        """Test listing goals."""
        # Create multiple goals
        for i in range(3):
            goal = GoalDefinition(
                id=str(uuid4()),
                text=f"goal_{i}",
                category=GoalCategory.USER_REQUESTED,
                value=0.5,
                effort=0.5,
                risk=0.5,
                horizon_min_min=0
            )
            goal_store.create_goal(goal)

        goals = goal_store.list_goals()
        assert len(goals) >= 3

    def test_update_goal(self, goal_store):
        """Test updating a goal."""
        goal = GoalDefinition(
            id=str(uuid4()),
            text="original_text",
            category=GoalCategory.USER_REQUESTED,
            value=0.5,
            effort=0.5,
            risk=0.5,
            horizon_min_min=0
        )

        created = goal_store.create_goal(goal)

        # Update
        updated = goal_store.update_goal(
            goal.id,
            {"text": "updated_text"},
            expected_version=created.version
        )

        assert updated is not None
        assert updated.text == "updated_text"
        assert updated.version == created.version + 1


class TestPhase3GoalAdoption:
    """Phase 3B: Goal adoption and safety checks."""

    def test_adopt_goal_success(self, goal_store):
        """Test adopting a goal successfully."""
        goal = GoalDefinition(
            id=str(uuid4()),
            text="implement_feature",
            category=GoalCategory.USER_REQUESTED,
            value=0.8,
            effort=0.3,
            risk=0.2,
            horizon_min_min=0
        )

        goal_store.create_goal(goal)

        # Adopt
        adopted, adopted_goal, details = goal_store.adopt_goal(goal.id)

        assert adopted is True
        assert adopted_goal.state == GoalState.ADOPTED

    def test_adopt_nonexistent_goal(self, goal_store):
        """Test adopting a goal that doesn't exist."""
        adopted, goal, details = goal_store.adopt_goal("nonexistent_id")

        assert adopted is False
        assert goal is None
        assert details["reason"] == "not_found"

    def test_adopt_with_contradicting_belief(self, goal_store):
        """Test that contradicting beliefs block adoption."""
        goal = GoalDefinition(
            id=str(uuid4()),
            text="test_goal",
            category=GoalCategory.USER_REQUESTED,
            value=0.5,
            effort=0.5,
            risk=0.5,
            horizon_min_min=0,
            contradicts=["belief_123"]  # Contradicts a belief
        )

        goal_store.create_goal(goal)

        # Try to adopt with active belief
        adopted, adopted_goal, details = goal_store.adopt_goal(
            goal.id,
            active_belief_ids=["belief_123"]
        )

        assert adopted is False
        assert details.get("blocked_by_belief") is True

    def test_idempotent_adoption(self, goal_store):
        """Test that adoption is idempotent."""
        goal = GoalDefinition(
            id=str(uuid4()),
            text="test_goal",
            category=GoalCategory.USER_REQUESTED,
            value=0.5,
            effort=0.5,
            risk=0.5,
            horizon_min_min=0
        )

        goal_store.create_goal(goal)

        # Adopt twice with same idempotency key
        key = "test_key_123"
        adopted1, _, _ = goal_store.adopt_goal(goal.id, idempotency_key=key)
        adopted2, _, details2 = goal_store.adopt_goal(goal.id, idempotency_key=key)

        assert adopted1 is True
        assert adopted2 is True
        assert details2.get("reason") == "idempotent"


class TestPhase3GoalExecution:
    """Phase 3C: Goal execution via GoalStore."""

    @pytest.mark.asyncio
    async def test_execute_adopted_goal(self, goal_store, code_access):
        """Test executing an adopted goal."""
        goal = GoalDefinition(
            id=str(uuid4()),
            text="implement_feature",
            category=GoalCategory.USER_REQUESTED,
            value=0.8,
            effort=0.3,
            risk=0.2,
            horizon_min_min=0,
            metadata={"test": "phase3"}
        )

        goal_store.create_goal(goal)
        goal_store.adopt_goal(goal.id)

        # Execute
        result = await goal_store.execute_goal(
            goal_id=goal.id,
            code_access_service=code_access,
            timeout_ms=60000
        )

        # Verify result
        assert result is not None
        assert result.goal_id == goal.id
        assert result.total_tasks >= 3

    @pytest.mark.asyncio
    async def test_execute_updates_goal_state(self, goal_store, code_access):
        """Test that execution updates goal state."""
        goal = GoalDefinition(
            id=str(uuid4()),
            text="add_tests",
            category=GoalCategory.USER_REQUESTED,
            value=0.5,
            effort=0.5,
            risk=0.5,
            horizon_min_min=0
        )

        goal_store.create_goal(goal)
        goal_store.adopt_goal(goal.id)

        initial_state = goal_store.get_goal(goal.id).state
        assert initial_state == GoalState.ADOPTED

        # Execute
        await goal_store.execute_goal(
            goal_id=goal.id,
            code_access_service=code_access,
            timeout_ms=60000
        )

        # Check final state
        final_goal = goal_store.get_goal(goal.id)

        # Should be SATISFIED (if successful) or ADOPTED (if failed)
        assert final_goal.state in [GoalState.SATISFIED, GoalState.ADOPTED]

    @pytest.mark.asyncio
    async def test_execute_stores_result_metadata(self, goal_store, code_access):
        """Test that execution results are stored in goal metadata."""
        goal = GoalDefinition(
            id=str(uuid4()),
            text="implement_feature",
            category=GoalCategory.USER_REQUESTED,
            value=0.8,
            effort=0.3,
            risk=0.2,
            horizon_min_min=0
        )

        goal_store.create_goal(goal)
        goal_store.adopt_goal(goal.id)

        # Execute
        await goal_store.execute_goal(
            goal_id=goal.id,
            code_access_service=code_access,
            timeout_ms=60000
        )

        # Check metadata
        final_goal = goal_store.get_goal(goal.id)
        assert "execution_result" in final_goal.metadata

        exec_result = final_goal.metadata["execution_result"]
        assert "success" in exec_result
        assert "completed_tasks" in exec_result
        assert "failed_tasks" in exec_result
        assert "execution_time_ms" in exec_result

    @pytest.mark.asyncio
    async def test_execute_unadopted_goal_fails(self, goal_store, code_access):
        """Test that executing unadopted goal raises error."""
        goal = GoalDefinition(
            id=str(uuid4()),
            text="implement_feature",
            category=GoalCategory.USER_REQUESTED,
            value=0.5,
            effort=0.5,
            risk=0.5,
            horizon_min_min=0
        )

        goal_store.create_goal(goal)
        # Don't adopt

        # Try to execute
        with pytest.raises(ValueError, match="must be ADOPTED"):
            await goal_store.execute_goal(
                goal_id=goal.id,
                code_access_service=code_access,
                timeout_ms=60000
            )

    @pytest.mark.asyncio
    async def test_execute_nonexistent_goal_fails(self, goal_store, code_access):
        """Test that executing nonexistent goal raises error."""
        with pytest.raises(ValueError, match="not found"):
            await goal_store.execute_goal(
                goal_id="nonexistent_id",
                code_access_service=code_access,
                timeout_ms=60000
            )


class TestPhase3EndToEnd:
    """Phase 3D: End-to-end pipeline tests."""

    @pytest.mark.asyncio
    async def test_full_pipeline_implement_feature(self, goal_store, code_access):
        """Test complete pipeline: create → adopt → execute → verify."""
        # Step 1: Create goal
        goal = GoalDefinition(
            id=str(uuid4()),
            text="implement_feature",
            category=GoalCategory.USER_REQUESTED,
            value=0.8,
            effort=0.3,
            risk=0.2,
            horizon_min_min=0,
            metadata={"e2e_test": True}
        )

        created = goal_store.create_goal(goal)
        assert created.state == GoalState.PROPOSED

        # Step 2: Adopt goal
        adopted, adopted_goal, _ = goal_store.adopt_goal(goal.id)
        assert adopted is True
        assert adopted_goal.state == GoalState.ADOPTED

        # Step 3: Execute goal
        result = await goal_store.execute_goal(
            goal_id=goal.id,
            code_access_service=code_access,
            timeout_ms=90000
        )

        # Step 4: Verify results
        assert result.goal_id == goal.id
        assert result.total_tasks >= 3

        # Should have created files
        assert len(result.completed_tasks) >= 2

        # Step 5: Verify final goal state
        final_goal = goal_store.get_goal(goal.id)
        assert final_goal.state in [GoalState.SATISFIED, GoalState.ADOPTED]
        assert "execution_result" in final_goal.metadata

        # Step 6: Verify files exist
        generated_files = list(Path("tests/generated").glob(f"feature_{result.goal_id[:8]}*.py"))
        assert len(generated_files) > 0

    @pytest.mark.asyncio
    async def test_multiple_goals_sequential(self, goal_store, code_access):
        """Test executing multiple goals sequentially."""
        results = []

        for goal_type in ["implement_feature", "add_tests"]:
            goal = GoalDefinition(
                id=str(uuid4()),
                text=goal_type,
                category=GoalCategory.USER_REQUESTED,
                value=0.5,
                effort=0.5,
                risk=0.5,
                horizon_min_min=0
            )

            goal_store.create_goal(goal)
            goal_store.adopt_goal(goal.id)

            result = await goal_store.execute_goal(
                goal_id=goal.id,
                code_access_service=code_access,
                timeout_ms=60000
            )

            results.append(result)

        # All should complete
        assert len(results) == 2
        assert all(r.total_tasks > 0 for r in results)

    @pytest.mark.asyncio
    async def test_goal_state_transitions(self, goal_store, code_access):
        """Test that goal states transition correctly."""
        goal = GoalDefinition(
            id=str(uuid4()),
            text="implement_feature",
            category=GoalCategory.USER_REQUESTED,
            value=0.8,
            effort=0.3,
            risk=0.2,
            horizon_min_min=0
        )

        # PROPOSED
        created = goal_store.create_goal(goal)
        assert created.state == GoalState.PROPOSED

        # ADOPTED
        adopted, adopted_goal, _ = goal_store.adopt_goal(goal.id)
        assert adopted_goal.state == GoalState.ADOPTED

        # EXECUTING → SATISFIED/ADOPTED
        result = await goal_store.execute_goal(
            goal_id=goal.id,
            code_access_service=code_access,
            timeout_ms=60000
        )

        final_goal = goal_store.get_goal(goal.id)
        # Should transition based on success
        if result.success:
            assert final_goal.state == GoalState.SATISFIED
        else:
            # Failed execution reverts to ADOPTED
            assert final_goal.state == GoalState.ADOPTED

    @pytest.mark.asyncio
    async def test_goal_source_tracking(self, goal_store, code_access):
        """Test that goal source is tracked correctly."""
        goal = GoalDefinition(
            id=str(uuid4()),
            text="implement_feature",
            category=GoalCategory.USER_REQUESTED,
            value=0.5,
            effort=0.5,
            risk=0.5,
            horizon_min_min=0,
            source=GoalSource.USER,
            created_by="test_user"
        )

        created = goal_store.create_goal(goal)
        assert created.source == GoalSource.USER
        assert created.created_by == "test_user"

        # Execute and verify source persists
        goal_store.adopt_goal(goal.id)
        await goal_store.execute_goal(
            goal_id=goal.id,
            code_access_service=code_access,
            timeout_ms=60000
        )

        final_goal = goal_store.get_goal(goal.id)
        assert final_goal.source == GoalSource.USER


class TestPhase3Performance:
    """Phase 3E: Performance and scaling tests."""

    @pytest.mark.asyncio
    async def test_execution_performance(self, goal_store, code_access):
        """Test that execution completes in reasonable time."""
        import time

        goal = GoalDefinition(
            id=str(uuid4()),
            text="implement_feature",
            category=GoalCategory.USER_REQUESTED,
            value=0.8,
            effort=0.3,
            risk=0.2,
            horizon_min_min=0
        )

        goal_store.create_goal(goal)
        goal_store.adopt_goal(goal.id)

        start = time.monotonic()

        result = await goal_store.execute_goal(
            goal_id=goal.id,
            code_access_service=code_access,
            timeout_ms=60000
        )

        elapsed = time.monotonic() - start

        # Should complete reasonably fast (< 5 seconds)
        assert elapsed < 5.0

        # Result should have timing info
        assert result.execution_time_ms > 0


def test_suite_info():
    """Print test suite information."""
    print("\n" + "="*70)
    print("PHASE 3 TEST SUITE: GoalStore + Full Pipeline Integration")
    print("="*70)
    print("\nTest Categories:")
    print("  A. GoalStore CRUD Operations")
    print("     - Create goal")
    print("     - Get goal")
    print("     - List goals")
    print("     - Update goal")
    print("  B. Goal Adoption")
    print("     - Adopt goal successfully")
    print("     - Adopt nonexistent goal")
    print("     - Belief contradiction blocking")
    print("     - Idempotent adoption")
    print("  C. Goal Execution")
    print("     - Execute adopted goal")
    print("     - State updates during execution")
    print("     - Result metadata storage")
    print("     - Error handling (unadopted, nonexistent)")
    print("  D. End-to-End Pipeline")
    print("     - Full pipeline: create → adopt → execute → verify")
    print("     - Multiple goals sequential")
    print("     - State transitions")
    print("     - Source tracking")
    print("  E. Performance")
    print("     - Execution performance")
    print("\nRun with: pytest tests/integration/test_phase3_goalstore_integration.py -v")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_suite_info()
