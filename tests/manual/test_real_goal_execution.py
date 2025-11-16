"""Manual test: Real goal execution with actual CodeAccessService.

This script demonstrates autonomous coding by giving Astra a real goal
and watching it execute tasks using the full pipeline:
  Goal → HTN Plan → TaskGraph → Execution → Files Created

Usage:
    python tests/manual/test_real_goal_execution.py
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.services.goal_execution_service import GoalExecutionService
from src.services.code_access import CodeAccessService


async def test_implement_simple_calculator():
    """Test goal: Implement a simple calculator module.

    This will:
    1. Decompose goal using HTN planner
    2. Create calculator.py with basic functions
    3. Create test_calculator.py with tests
    4. Run the tests (which will fail since it's just placeholders)
    """
    print("=" * 70)
    print("REAL-WORLD TEST: Autonomous Code Generation")
    print("=" * 70)
    print()

    # Initialize CodeAccessService
    print("📦 Initializing CodeAccessService...")
    project_root = "/home/d/git/ai-exp"
    code_access = CodeAccessService(project_root=project_root)
    print("✅ CodeAccessService ready")
    print()

    # Initialize GoalExecutionService
    print("🤖 Initializing GoalExecutionService...")
    service = GoalExecutionService(
        code_access=code_access,
        workdir="/home/d/git/ai-exp",
        max_concurrent=2
    )
    print(f"✅ Service ready with {len(service.planner.methods)} HTN methods")
    print()

    # Execute goal
    goal = "implement_feature"
    print(f"🎯 Goal: {goal}")
    print("   Expected: HTN will decompose to [create_file, create_file, run_tests]")
    print()

    print("⚙️  Executing goal...")
    result = await service.execute_goal(
        goal_text=goal,
        context={},
        timeout_ms=60000  # 1 minute timeout
    )

    # Display results
    print()
    print("=" * 70)
    print("EXECUTION RESULTS")
    print("=" * 70)
    print()

    print(f"✅ Success: {result.success}")
    print(f"⏱️  Execution Time: {result.execution_time_ms:.1f}ms")
    print(f"📋 Total Tasks: {result.total_tasks}")
    print(f"✅ Completed: {len(result.completed_tasks)}")
    print(f"❌ Failed: {len(result.failed_tasks)}")
    print(f"🔄 Retries: {result.retry_count}")
    print()

    # Show HTN planning
    print("🧠 HTN Planning:")
    print(f"   Plan ID: {result.plan_id}")
    print(f"   Methods Used: {result.methods_used}")
    print(f"   Planning Cost: {result.planning_cost}")
    print()

    # Show completed tasks
    if result.completed_tasks:
        print("✅ Completed Tasks:")
        for task in result.completed_tasks:
            print(f"   - {task.task_name} (ID: {task.task_id})")
            if 'file_path' in task.artifacts:
                print(f"     File: {task.artifacts['file_path']}")
        print()

    # Show failed tasks
    if result.failed_tasks:
        print("❌ Failed Tasks:")
        for task in result.failed_tasks:
            print(f"   - {task.task_name} (ID: {task.task_id})")
            print(f"     Error: {task.error}")
            print(f"     Class: {task.error_class}")
        print()

    # Show errors
    if result.errors:
        print("⚠️  Errors:")
        for error in result.errors:
            print(f"   - {error}")
        print()

    # Show artifacts
    if result.artifacts:
        print("📦 Artifacts:")
        for task_id, artifacts in result.artifacts.items():
            print(f"   Task {task_id}:")
            for key, value in artifacts.items():
                if isinstance(value, str) and len(value) > 50:
                    print(f"     {key}: {value[:50]}...")
                else:
                    print(f"     {key}: {value}")
        print()

    print("=" * 70)
    if result.success:
        print("🎉 SUCCESS: Goal executed successfully!")
    else:
        print("⚠️  INCOMPLETE: Some tasks failed")
    print("=" * 70)

    return result


async def test_fix_simple_bug():
    """Test goal: Fix a bug in existing code."""
    print("\n" + "=" * 70)
    print("TEST 2: Bug Fix Goal")
    print("=" * 70)
    print()

    project_root = "/home/d/git/ai-exp"
    code_access = CodeAccessService(project_root=project_root)
    service = GoalExecutionService(
        code_access=code_access,
        workdir=project_root,
        max_concurrent=2
    )

    print("🎯 Goal: fix_bug")
    print("   Expected: HTN will decompose to [modify_code, run_tests]")
    print()

    result = await service.execute_goal(
        goal_text="fix_bug",
        timeout_ms=60000
    )

    print(f"\n✅ Result: {result.success}")
    print(f"📋 Tasks: {len(result.completed_tasks)} completed, {len(result.failed_tasks)} failed")
    print(f"⏱️  Time: {result.execution_time_ms:.1f}ms")

    return result


async def main():
    """Run all manual tests."""
    print("\n" + "#" * 70)
    print("# MANUAL TEST SUITE: Real Goal Execution")
    print("#" * 70)
    print()

    try:
        # Test 1: Implement feature
        result1 = await test_implement_simple_calculator()

        # Test 2: Fix bug
        result2 = await test_fix_simple_bug()

        print("\n" + "#" * 70)
        print("# ALL TESTS COMPLETE")
        print("#" * 70)
        print()
        print(f"Test 1 (implement_feature): {'✅ PASS' if result1.success else '❌ FAIL'}")
        print(f"Test 2 (fix_bug): {'✅ PASS' if result2.success else '❌ FAIL'}")
        print()

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
