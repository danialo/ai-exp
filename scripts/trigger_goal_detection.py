#!/usr/bin/env python3
"""Manually trigger goal generator pattern detection for testing.

This script bypasses the hourly background task and runs pattern detection
immediately, allowing you to test the system without waiting.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.services.goal_store import create_goal_store
from src.services.belief_store import create_belief_store
from src.services.task_scheduler import create_task_scheduler
from src.services.goal_generator import GoalGenerator
from src.services.detectors.task_failure_detector import TaskFailureDetector
from src.memory.raw_store import create_raw_store


async def main():
    """Run pattern detection manually."""
    print("=" * 60)
    print("Manual Goal Generator Trigger")
    print("=" * 60)
    print()

    # Initialize stores
    print("📦 Initializing stores...")
    raw_store = create_raw_store(settings.RAW_STORE_DB_PATH)
    goal_store = create_goal_store(settings.RAW_STORE_DB_PATH)

    belief_store = None
    if settings.PERSONA_MODE_ENABLED:
        belief_store = create_belief_store(
            persona_space_path=settings.PERSONA_SPACE_PATH
        )

    task_scheduler = None
    if settings.PERSONA_MODE_ENABLED:
        task_scheduler = create_task_scheduler(
            persona_space_path=settings.PERSONA_SPACE_PATH,
            raw_store=raw_store,
        )

    # Create detectors
    print("🔍 Initializing detectors...")
    detectors = []
    if task_scheduler:
        task_failure_detector = TaskFailureDetector(
            task_scheduler=task_scheduler,
            failure_threshold=3,
            lookback_hours=24,
            min_confidence=0.75,
            scan_interval=60,
            detector_enabled=True,
        )
        detectors.append(task_failure_detector)
        print(f"  ✓ TaskFailureDetector")

    # Create goal generator
    print("🎯 Initializing goal generator...")
    generator = GoalGenerator(
        goal_store=goal_store,
        belief_store=belief_store,
        detectors=detectors,
        min_confidence=0.7,
        max_system_goals_per_day=10,
        max_goals_per_detector_per_day=3,
        belief_alignment_threshold=0.5,
        auto_approve_threshold=0.9,
    )

    print()
    print("=" * 60)
    print("Running Pattern Detection...")
    print("=" * 60)
    print()

    # Run detection
    created, rejected = await generator.generate_and_create_goals()

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"✅ Goals Created:  {created}")
    print(f"❌ Goals Rejected: {rejected}")
    print()

    # Show telemetry
    telemetry = generator.get_telemetry()
    print("📊 Telemetry:")
    print(f"  Total proposals:    {telemetry['total_proposals_evaluated']}")
    print(f"  Total created:      {telemetry['total_goals_created']}")
    print(f"  Total rejected:     {telemetry['total_goals_rejected']}")
    print(f"  Auto-approved:      {telemetry['auto_approved_count']}")
    print()
    print(f"  Rejection reasons:")
    for reason, count in telemetry['rejection_reasons'].items():
        print(f"    - {reason}: {count}")
    print()

    # Show created goals
    if created > 0:
        print("=" * 60)
        print("Newly Created Goals")
        print("=" * 60)
        all_goals = goal_store.list_goals(state_filter="all")
        system_goals = [g for g in all_goals if g.source.value == "system"]
        # Show last N created
        for goal in system_goals[-created:]:
            print(f"\n🎯 {goal.text}")
            print(f"   ID: {goal.id}")
            print(f"   State: {goal.state.value}")
            print(f"   Source: {goal.source.value}")
            print(f"   Created by: {goal.created_by}")
            print(f"   Auto-approved: {goal.auto_approved}")
            print(f"   Confidence: {goal.metadata.get('confidence', 'N/A') if goal.metadata else 'N/A'}")


if __name__ == "__main__":
    asyncio.run(main())
