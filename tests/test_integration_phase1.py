#!/usr/bin/env python3
"""
Test script for Integration Layer Phase 1

Verifies signal flow from subsystems to IntegrationLayer:
1. Create IntegrationEventHub
2. Create IntegrationLayer and start it
3. Publish test signals directly to hub
4. Verify signals accumulate in AstraState
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from integration import (
    IntegrationEventHub,
    IntegrationLayer,
    IdentityService,
    PerceptSignal,
    DissonanceSignal,
    Priority,
    ExecutionMode,
)


async def test_signal_flow():
    """Test that signals flow from event hub to Integration Layer."""

    print("=" * 60)
    print("Integration Layer Phase 1 Signal Flow Test")
    print("=" * 60)

    # 1. Create event hub
    print("\n1. Creating IntegrationEventHub...")
    event_hub = IntegrationEventHub()
    print(f"   ✓ Event hub created")

    # 2. Create identity service (minimal, no real dependencies)
    print("\n2. Creating IdentityService (stub)...")
    identity_service = IdentityService()  # No args = all None
    print(f"   ✓ Identity service created")

    # 3. Create and start Integration Layer
    print("\n3. Creating IntegrationLayer...")
    il = IntegrationLayer(
        event_hub=event_hub,
        identity_service=identity_service,
        mode=ExecutionMode.INTERACTIVE
    )
    print(f"   ✓ Integration Layer created")

    print("\n4. Starting IntegrationLayer...")
    await il.start()
    print(f"   ✓ Integration Layer started")

    # Give it a moment to subscribe
    await asyncio.sleep(0.1)

    # 5. Publish test percept signals
    print("\n5. Publishing test PerceptSignals...")
    for i in range(3):
        signal = PerceptSignal(
            signal_id=f"test_percept_{i}",
            source="test_script",
            timestamp=datetime.now(),
            priority=Priority.HIGH if i == 0 else Priority.NORMAL,
            percept_type="user" if i == 0 else "token",
            content={"text": f"Test message {i}"},
            novelty=0.8,
            entropy=0.6
        )
        event_hub.publish("percepts", signal)
        print(f"   ✓ Published percept {i}: {signal.percept_type}")

    # 6. Publish test dissonance signals
    print("\n6. Publishing test DissonanceSignals...")
    for i in range(2):
        signal = DissonanceSignal(
            signal_id=f"test_dissonance_{i}",
            source="test_script",
            timestamp=datetime.now(),
            priority=Priority.HIGH if i == 0 else Priority.NORMAL,
            pattern="contradiction" if i == 0 else "hedging",
            belief_id=f"belief_{i}",
            conflicting_memory=f"Memory contradicts belief {i}",
            severity=0.8 if i == 0 else 0.5
        )
        event_hub.publish("dissonance", signal)
        print(f"   ✓ Published dissonance {i}: {signal.pattern} (severity={signal.severity})")

    # Give IL time to process
    await asyncio.sleep(0.5)

    # 7. Verify signals accumulated in AstraState
    print("\n7. Verifying signal accumulation in AstraState...")
    stats = il.get_stats()
    state = il.get_state()

    print(f"\n   IL Stats:")
    print(f"   - Mode: {stats['mode']}")
    print(f"   - Percept buffer size: {stats['percept_buffer_size']}")
    print(f"   - Dissonance alert count: {stats['dissonance_alert_count']}")
    print(f"   - Focus stack size: {stats['focus_stack_size']}")
    print(f"   - Self-model loaded: {stats['self_model_loaded']}")

    # Assertions
    assert stats['percept_buffer_size'] == 3, f"Expected 3 percepts, got {stats['percept_buffer_size']}"
    assert stats['dissonance_alert_count'] == 2, f"Expected 2 dissonance alerts, got {stats['dissonance_alert_count']}"

    print("\n   ✓ All signals accumulated correctly!")

    # 8. Verify percept details
    print("\n8. Verifying percept details...")
    for i, percept in enumerate(state.percept_buffer):
        print(f"   Percept {i}: type={percept.percept_type}, priority={percept.priority.name}, novelty={percept.novelty}")

    # 9. Verify dissonance details
    print("\n9. Verifying dissonance details...")
    for i, dissonance in enumerate(state.dissonance_alerts):
        print(f"   Dissonance {i}: pattern={dissonance.pattern}, severity={dissonance.severity}, belief={dissonance.belief_id}")

    # 10. Stop Integration Layer
    print("\n10. Stopping IntegrationLayer...")
    await il.stop()
    print(f"    ✓ Integration Layer stopped")

    print("\n" + "=" * 60)
    print("✓ Phase 1 Signal Flow Test PASSED")
    print("=" * 60)
    print("\nNext steps:")
    print("- Wire IL into app.py startup")
    print("- Pass event_hub to awareness_loop and belief_checker")
    print("- Add GET /api/integration/state endpoint")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(test_signal_flow())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
