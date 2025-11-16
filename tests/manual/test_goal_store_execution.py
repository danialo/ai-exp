"""End-to-end test for goal creation → adoption → execution.

Tests the full pipeline:
1. Create a goal in GoalStore
2. Adopt the goal
3. Execute using GoalStore.execute_goal()
4. Verify state transitions and results
"""
import asyncio
from pathlib import Path
from uuid import uuid4

from src.services.goal_store import create_goal_store, GoalDefinition, GoalCategory, GoalState
from src.services.code_access import create_code_access_service


async def test_goal_execution_pipeline():
    """Test end-to-end goal execution."""

    # Initialize services
    project_root = Path("/home/d/git/ai-exp")
    code_access = create_code_access_service(
        project_root=project_root,
        max_file_size_kb=100,
        auto_branch=True
    )

    goal_store = create_goal_store("temp_goals.db")

    print("\n🎯 End-to-End Goal Execution Test")
    print("=" * 70)

    # Step 1: Create a goal
    goal_id = str(uuid4())
    goal = GoalDefinition(
        id=goal_id,
        text="implement_feature",
        category=GoalCategory.USER_REQUESTED,
        value=0.8,
        effort=0.3,
        risk=0.2,
        horizon_min_min=0,
        horizon_max_min=60,
        metadata={"test": "end_to_end"}
    )

    created_goal = goal_store.create_goal(goal)
    print(f"\n✓ Step 1: Created goal {created_goal.id[:8]}")
    print(f"  State: {created_goal.state.value}")
    print(f"  Text: {created_goal.text}")

    # Step 2: Adopt the goal
    adopted, adopted_goal, details = goal_store.adopt_goal(goal_id)
    print(f"\n✓ Step 2: Adopted goal")
    print(f"  Success: {adopted}")
    print(f"  State: {adopted_goal.state.value if adopted_goal else 'N/A'}")

    if not adopted:
        print(f"  Adoption failed: {details}")
        return

    # Step 3: Execute the goal
    print(f"\n⚙️  Step 3: Executing goal...")
    try:
        result = await goal_store.execute_goal(
            goal_id=goal_id,
            code_access_service=code_access,
            timeout_ms=60000
        )

        print(f"\n✓ Step 3: Goal execution complete")
        print(f"  Success: {result.success}")
        print(f"  Execution time: {result.execution_time_ms:.1f}ms")
        print(f"  Total tasks: {result.total_tasks}")
        print(f"  Completed: {len(result.completed_tasks)}")
        print(f"  Failed: {len(result.failed_tasks)}")

        # Show completed tasks
        if result.completed_tasks:
            print(f"\n  ✅ Completed Tasks:")
            for task in result.completed_tasks:
                file_path = task.artifacts.get("file_path", "N/A")
                print(f"    - {task.task_name}: {file_path}")

        # Show failed tasks
        if result.failed_tasks:
            print(f"\n  ❌ Failed Tasks:")
            for task in result.failed_tasks:
                print(f"    - {task.task_name}: {task.error}")

    except Exception as e:
        print(f"\n✗ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Verify goal state
    final_goal = goal_store.get_goal(goal_id)
    print(f"\n✓ Step 4: Verified final state")
    print(f"  State: {final_goal.state.value}")
    print(f"  Has execution result: {'execution_result' in final_goal.metadata}")

    if 'execution_result' in final_goal.metadata:
        exec_meta = final_goal.metadata['execution_result']
        print(f"  Result metadata:")
        print(f"    - Success: {exec_meta.get('success')}")
        print(f"    - Completed tasks: {exec_meta.get('completed_tasks')}")
        print(f"    - Failed tasks: {exec_meta.get('failed_tasks')}")

    # Cleanup
    goal_store.close()
    Path("temp_goals.db").unlink(missing_ok=True)

    print("\n" + "=" * 70)
    print("✅ END-TO-END TEST COMPLETE")


if __name__ == "__main__":
    asyncio.run(test_goal_execution_pipeline())
