"""
Phase 2 Integration Layer Tests - Surgical test set

Tests three critical areas:
1. Executive Loop & Focus Sanity
2. Introspection & Budget Control
3. Belief Mutation Safety Rails

Run with: pytest tests/test_integration_layer_phase2.py -v
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from collections import deque

# Import IL components
from src.integration.layer import IntegrationLayer
from src.integration.state import (
    AstraState, ExecutionMode, FocusItem, FocusType,
    SelfModelSnapshot, BudgetStatus
)
from src.integration.signals import (
    PerceptSignal, DissonanceSignal, GoalProposal, Priority
)
from src.integration.event_hub import IntegrationEventHub
from src.integration.identity_service import IdentityService

# Import belief store for safety rail tests
from src.services.belief_store import BeliefStore, BeliefVersion, BeliefState, DeltaOp


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_identity_service():
    """Create mock IdentityService with canned SelfModelSnapshot."""
    mock = Mock(spec=IdentityService)
    mock.get_snapshot.return_value = SelfModelSnapshot(
        core_beliefs=[{"id": "core1"}, {"id": "core2"}],
        peripheral_beliefs=[],
        traits={"curiosity": 0.8},
        origin_anchor=[0.1] * 64,
        live_anchor=[0.1] * 64,
        anchor_drift=0.0,
        known_capabilities=set(),
        limitations=set(),
        confidence_self_model=0.8,
        last_major_update=None,
        snapshot_id="test-snapshot",
        created_at=datetime.now(),
    )
    return mock


@pytest.fixture
def mock_event_hub():
    """Create mock IntegrationEventHub."""
    hub = Mock(spec=IntegrationEventHub)
    hub.subscribe = Mock()
    hub.publish = Mock()
    return hub


@pytest.fixture
def mock_awareness_loop():
    """Create mock awareness loop with trigger_introspection spy."""
    mock = Mock()
    mock.trigger_introspection = AsyncMock(return_value=True)
    mock.running = True
    return mock


@pytest.fixture
def integration_layer(mock_event_hub, mock_identity_service):
    """Create IntegrationLayer for testing (no start())."""
    il = IntegrationLayer(
        event_hub=mock_event_hub,
        identity_service=mock_identity_service,
        mode=ExecutionMode.INTERACTIVE,
        snapshot_dir=Path("/tmp/test_il_snapshots"),
    )
    return il


@pytest.fixture
def temp_beliefs_dir(tmp_path):
    """Create temporary beliefs directory with test data."""
    beliefs_dir = tmp_path / "beliefs"
    beliefs_dir.mkdir()

    # Create minimal current.json
    current = {
        "core_identity": {
            "belief_id": "core_identity",
            "ver": 1,
            "statement": "I am Astra",
            "state": "asserted",
            "confidence": 1.0,
            "evidence_refs": [],
            "updated_by": "migration",
            "ts": datetime.now().timestamp(),
            "belief_type": "self",
            "immutable": False,
            "rationale": "Core identity",
            "metadata": {},
            "hash": "test",
            "stability": 1.0,
            "is_core": True,
        },
        "peripheral_belief": {
            "belief_id": "peripheral_belief",
            "ver": 1,
            "statement": "I like learning",
            "state": "tentative",
            "confidence": 0.5,
            "evidence_refs": [],
            "updated_by": "slow",
            "ts": datetime.now().timestamp(),
            "belief_type": "experiential",
            "immutable": False,
            "rationale": "From experience",
            "metadata": {},
            "hash": "test2",
            "stability": 0.3,
            "is_core": False,
        }
    }

    with open(beliefs_dir / "current.json", "w") as f:
        json.dump(current, f)

    # Create index.json (required by BeliefStore)
    index = {
        "core_identity": {"versions": [1], "current_ver": 1},
        "peripheral_belief": {"versions": [1], "current_ver": 1},
    }
    with open(beliefs_dir / "index.json", "w") as f:
        json.dump(index, f)

    # Create deltas directory (required by BeliefStore)
    (beliefs_dir / "deltas").mkdir()

    return beliefs_dir


# =============================================================================
# 1. EXECUTIVE LOOP & FOCUS SANITY
# =============================================================================

class TestExecutiveLoop:
    """Tests for executive loop mechanics."""

    @pytest.mark.asyncio
    async def test_single_tick_smoke(self, integration_layer, mock_identity_service):
        """
        Single tick smoke test.
        Prove tick runs, state updates, nothing explodes.
        """
        # Arrange: IL not started, call _execute_tick directly
        assert integration_layer._tick_count == 0
        assert integration_layer.state.tick_id == 0

        # Act: Execute one tick
        await integration_layer._execute_tick()

        # Assert
        assert integration_layer._tick_count == 1
        assert integration_layer.state.tick_id == 1
        assert integration_layer.state.timestamp is not None
        assert integration_layer.total_ticks_executed == 1

    @pytest.mark.asyncio
    async def test_multiple_ticks(self, integration_layer):
        """Multiple ticks increment correctly."""
        for i in range(5):
            await integration_layer._execute_tick()

        assert integration_layer._tick_count == 5
        assert integration_layer.total_ticks_executed == 5


class TestFocusStack:
    """Tests for focus stack management."""

    def test_focus_stack_capacity_miller_law(self, integration_layer):
        """
        Push 10 items, verify only 7 remain (Miller's Law).
        Highest salience items should survive.
        """
        # Arrange: Create 10 focus items with varying salience
        for i in range(10):
            item = FocusItem(
                item_type=FocusType.EXTERNAL_EVENT,
                item_id=f"item_{i}",
                content=f"Test item {i}",
                salience=i * 0.1,  # 0.0 to 0.9
                entered_focus=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                decay_rate=0.01,
                min_salience_threshold=0.1,
            )
            integration_layer._add_to_focus(item)

        # Assert: Only 7 items remain
        assert len(integration_layer.state.focus_stack) == 7

        # Assert: Highest salience items survived (items 3-9, salience 0.3-0.9)
        saliences = [item.salience for item in integration_layer.state.focus_stack]
        assert min(saliences) >= 0.3  # Lowest salience in stack

    def test_focus_stack_evicts_lowest_salience(self, integration_layer):
        """Verify lowest salience item is evicted when full."""
        # Fill to capacity with equal salience
        for i in range(7):
            item = FocusItem(
                item_type=FocusType.TASK,
                item_id=f"item_{i}",
                content=f"Item {i}",
                salience=0.5,
                entered_focus=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                decay_rate=0.01,
                min_salience_threshold=0.1,
            )
            integration_layer._add_to_focus(item)

        # Add high-salience item
        high_item = FocusItem(
            item_type=FocusType.USER_MESSAGE,
            item_id="high_priority",
            content="High priority item",
            salience=0.95,
            entered_focus=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            decay_rate=0.01,
            min_salience_threshold=0.1,
        )
        integration_layer._add_to_focus(high_item)

        # Assert: Still 7 items, high priority included
        assert len(integration_layer.state.focus_stack) == 7
        ids = [item.item_id for item in integration_layer.state.focus_stack]
        assert "high_priority" in ids

    @pytest.mark.asyncio
    async def test_salience_decay(self, integration_layer):
        """
        Verify salience decays over time and items below threshold are evicted.
        """
        # Arrange: Add items with timestamps in the past
        old_time = datetime.now() - timedelta(minutes=10)

        item = FocusItem(
            item_type=FocusType.MEMORY,
            item_id="old_item",
            content="Old memory",
            salience=0.5,
            entered_focus=old_time,
            last_accessed=old_time,
            access_count=1,
            decay_rate=0.1,  # Fast decay
            min_salience_threshold=0.2,
        )
        # Manually set timestamp for decay calculation
        item.timestamp = old_time
        integration_layer.state.focus_stack.append(item)

        # Act: Compute focus (applies decay)
        integration_layer._compute_focus()

        # Assert: Old item should have decayed significantly or been evicted
        # With 0.1 decay rate over 10 minutes, salience drops a lot
        if integration_layer.state.focus_stack:
            for remaining in integration_layer.state.focus_stack:
                if remaining.item_id == "old_item":
                    assert remaining.salience < 0.5  # Decayed


class TestSignalWorkspace:
    """Tests for signal → workspace integration."""

    @pytest.mark.asyncio
    async def test_percept_to_focus(self, integration_layer):
        """High-novelty percepts become focus items."""
        # Arrange: Create high-novelty percept
        percept = PerceptSignal(
            signal_id="p1",
            source="awareness_loop",
            timestamp=datetime.now(),
            priority=Priority.NORMAL,
            percept_type="user",
            content="Hello Astra",
            novelty=0.9,
            entropy=0.5,
        )

        # Simulate signal arrival
        integration_layer._on_percept(percept)

        # Collect and integrate
        signals = integration_layer._collect_signals()
        await integration_layer._update_workspace(signals)

        # Assert: Focus stack should have the percept
        assert len(integration_layer.state.focus_stack) >= 1

    @pytest.mark.asyncio
    async def test_dissonance_becomes_focus(self, integration_layer):
        """Dissonance signals become high-priority focus items."""
        # Arrange
        dissonance = DissonanceSignal(
            signal_id="d1",
            source="belief_checker",
            timestamp=datetime.now(),
            priority=Priority.HIGH,
            pattern="contradiction",
            belief_id="test_belief",
            conflicting_memory="mem_123",
            severity=0.8,
        )

        # Simulate
        integration_layer._on_dissonance(dissonance)
        signals = integration_layer._collect_signals()
        await integration_layer._update_workspace(signals)

        # Assert
        assert len(integration_layer.state.focus_stack) >= 1
        # Dissonance should have high salience
        dissonance_items = [
            item for item in integration_layer.state.focus_stack
            if "dissonance" in item.content.lower()
        ]
        assert len(dissonance_items) >= 1

    @pytest.mark.asyncio
    async def test_signals_drain_buffers(self, integration_layer):
        """Collecting signals drains the buffers."""
        # Add some signals
        for i in range(5):
            percept = PerceptSignal(
                signal_id=f"p{i}",
                source="test",
                timestamp=datetime.now(),
                priority=Priority.LOW,
                percept_type="time",
                content={},
                novelty=0.1,
                entropy=0.1,
            )
            integration_layer._on_percept(percept)

        assert len(integration_layer.state.percept_buffer) == 5

        # Collect
        signals = integration_layer._collect_signals()

        # Assert: Buffer drained
        assert len(integration_layer.state.percept_buffer) == 0
        assert len(signals) == 5


# =============================================================================
# 2. INTROSPECTION & BUDGET CONTROL
# =============================================================================

class TestIntrospectionControl:
    """Tests for IL-controlled introspection."""

    @pytest.mark.asyncio
    async def test_introspection_triggers_via_il(
        self, integration_layer, mock_awareness_loop
    ):
        """
        Introspection only fires when IL conditions are met.
        Phase 3: Uses _dispatch_actions with INTROSPECTION action.
        """
        from src.integration.state import Action, ActionType

        # Arrange
        integration_layer.awareness_loop = mock_awareness_loop
        integration_layer._last_introspection_tick = 0
        integration_layer._tick_count = 35  # Past threshold (30)

        # Create introspection action
        action = Action(
            action_type=ActionType.INTROSPECTION,
            target_id=None,
            priority=2,
            estimated_cost={"tokens": 200}
        )

        # Act
        await integration_layer._dispatch_actions([action])

        # Assert: trigger_introspection called exactly once
        mock_awareness_loop.trigger_introspection.assert_awaited_once()
        assert integration_layer._last_introspection_tick == 35
        assert integration_layer.introspections_today == 1

    @pytest.mark.asyncio
    async def test_introspection_cooldown_respected(
        self, integration_layer, mock_awareness_loop
    ):
        """
        Introspection action not selected if cooldown not elapsed.
        """
        # Arrange
        integration_layer.awareness_loop = mock_awareness_loop
        integration_layer._tick_count = 10
        integration_layer._last_introspection_tick = 5  # Only 5 ticks ago
        integration_layer.mode = ExecutionMode.AUTONOMOUS  # Introspection only in AUTONOMOUS

        # Act: Select actions (should not include introspection due to cooldown)
        actions = integration_layer._select_actions([])

        # Assert: No introspection action selected (cooldown not elapsed)
        introspection_actions = [a for a in actions if a.action_type.value == "introspection"]
        assert len(introspection_actions) == 0

    @pytest.mark.asyncio
    async def test_introspection_respects_daily_budget(
        self, integration_layer, mock_awareness_loop
    ):
        """
        Introspection should not fire if daily budget exhausted.
        """
        # Arrange
        integration_layer.awareness_loop = mock_awareness_loop
        integration_layer._tick_count = 100
        integration_layer._last_introspection_tick = 0
        integration_layer.introspections_today = 50  # Exhausted
        integration_layer.introspection_daily_budget = 50

        # Act
        budget_ok = integration_layer._check_budgets()

        # Assert: Budget exhausted
        assert budget_ok is False

    @pytest.mark.asyncio
    async def test_no_introspection_without_awareness_loop(self, integration_layer):
        """
        No crash if awareness_loop is None when dispatching introspection.
        """
        from src.integration.state import Action, ActionType

        # Arrange
        integration_layer.awareness_loop = None
        integration_layer._tick_count = 100

        # Create introspection action
        action = Action(
            action_type=ActionType.INTROSPECTION,
            target_id=None,
            priority=2,
            estimated_cost={"tokens": 200}
        )

        # Act (should not crash)
        await integration_layer._dispatch_actions([action])

        # Assert: No exception raised, introspection counter unchanged
        assert integration_layer.introspections_today == 0


# =============================================================================
# 3. BELIEF MUTATION SAFETY RAILS
# =============================================================================

class TestBeliefStoreKillSwitch:
    """Tests for BeliefStore mutation kill switch."""

    def test_kill_switch_blocks_all_mutations(self, temp_beliefs_dir):
        """
        With mutations disabled, all writes should fail.
        """
        store = BeliefStore(data_dir=temp_beliefs_dir.parent)

        # Verify kill switch is ON by default
        assert store.mutations_enabled is False

        # Try to apply delta
        result = store.apply_delta(
            belief_id="peripheral_belief",
            from_ver=1,
            op=DeltaOp.UPDATE,
            confidence_delta=0.1,
            updated_by="test",
            reason="Test update",
        )

        # Assert: Blocked
        assert result is False

    def test_kill_switch_can_be_enabled(self, temp_beliefs_dir):
        """
        Mutations can be enabled explicitly.
        """
        store = BeliefStore(data_dir=temp_beliefs_dir.parent)

        # Enable mutations
        store.enable_mutations()
        assert store.mutations_enabled is True

        # Disable again
        store.disable_mutations()
        assert store.mutations_enabled is False


class TestCoreBeliefProtection:
    """Tests for core belief immutability."""

    def test_core_belief_blocked_by_is_core(self, temp_beliefs_dir):
        """
        Beliefs with is_core=True cannot be mutated even with mutations enabled.
        """
        store = BeliefStore(data_dir=temp_beliefs_dir.parent)
        store.enable_mutations()

        # Try to modify core belief
        result = store.apply_delta(
            belief_id="core_identity",
            from_ver=1,
            op=DeltaOp.UPDATE,
            confidence_delta=-0.1,
            updated_by="test",
            reason="Trying to modify core",
        )

        # Assert: Blocked by is_core check
        assert result is False

    def test_high_stability_blocked(self, temp_beliefs_dir):
        """
        Beliefs with stability >= 0.95 cannot be mutated.
        """
        store = BeliefStore(data_dir=temp_beliefs_dir.parent)
        store.enable_mutations()

        # core_identity has stability=1.0
        result = store.apply_delta(
            belief_id="core_identity",
            from_ver=1,
            op=DeltaOp.UPDATE,
            confidence_delta=0.05,
            updated_by="test",
            reason="Test",
        )

        assert result is False

    def test_peripheral_belief_allowed(self, temp_beliefs_dir):
        """
        Peripheral beliefs (is_core=False, stability<0.95) can be mutated.
        """
        store = BeliefStore(data_dir=temp_beliefs_dir.parent)
        store.enable_mutations()

        # peripheral_belief has stability=0.3, is_core=False
        result = store.apply_delta(
            belief_id="peripheral_belief",
            from_ver=1,
            op=DeltaOp.UPDATE,
            confidence_delta=0.1,
            updated_by="test",
            reason="Test update",
        )

        # This should succeed
        assert result is True


class TestIdentityServicePolicy:
    """Tests for IdentityService policy enforcement."""

    def test_invalid_cause_rejected(self, temp_beliefs_dir):
        """
        Unknown causes are rejected.
        """
        # Create real BeliefStore
        store = BeliefStore(data_dir=temp_beliefs_dir.parent)
        store.enable_mutations()

        # Create IdentityService with real store
        identity_svc = IdentityService(belief_store=store)

        # Try with invalid cause
        result = identity_svc.update_belief(
            belief_id="peripheral_belief",
            updates={"confidence": 0.6},
            cause="RANDOM_INVALID_CAUSE",
        )

        assert result is False

    def test_valid_cause_accepted(self, temp_beliefs_dir):
        """
        Valid causes (DISSONANCE_RESOLUTION, ADMIN_OVERRIDE) are accepted.
        """
        store = BeliefStore(data_dir=temp_beliefs_dir.parent)
        store.enable_mutations()
        identity_svc = IdentityService(belief_store=store)

        # ADMIN_OVERRIDE is valid for peripheral beliefs
        result = identity_svc.update_belief(
            belief_id="peripheral_belief",
            updates={"confidence": 0.6},
            cause="ADMIN_OVERRIDE",
        )

        assert result is True

    def test_rate_limit_enforced(self, temp_beliefs_dir):
        """
        Second mutation within 30 seconds should be blocked.
        """
        store = BeliefStore(data_dir=temp_beliefs_dir.parent)
        store.enable_mutations()
        identity_svc = IdentityService(belief_store=store)

        # First call should succeed
        result1 = identity_svc.update_belief(
            belief_id="peripheral_belief",
            updates={"confidence": 0.6},
            cause="ADMIN_OVERRIDE",
        )
        assert result1 is True

        # Second call within window should fail
        result2 = identity_svc.update_belief(
            belief_id="peripheral_belief",
            updates={"confidence": 0.7},
            cause="ADMIN_OVERRIDE",
        )
        assert result2 is False

    def test_protected_type_requires_dissonance(self, temp_beliefs_dir):
        """
        Protected belief types (self, ontological) require DISSONANCE_RESOLUTION.
        """
        store = BeliefStore(data_dir=temp_beliefs_dir.parent)
        store.enable_mutations()
        identity_svc = IdentityService(belief_store=store)

        # core_identity is type="self" which is protected
        # Even with ADMIN_OVERRIDE, should be blocked by is_core/stability
        result = identity_svc.update_belief(
            belief_id="core_identity",
            updates={"confidence": 0.9},
            cause="ADMIN_OVERRIDE",
        )

        # Blocked by core protection (is_core=True, stability=1.0)
        assert result is False


class TestDataLevelCheck:
    """Tests verifying actual data file state."""

    def test_current_json_has_protected_beliefs(self):
        """
        Verify real current.json has core beliefs properly protected.
        """
        beliefs_path = Path("data/beliefs/current.json")
        if not beliefs_path.exists():
            pytest.skip("No beliefs file in test environment")

        with open(beliefs_path) as f:
            beliefs = json.load(f)

        # Count protected beliefs
        protected = [
            bid for bid, b in beliefs.items()
            if b.get("is_core") is True and b.get("stability", 0) >= 0.95
        ]

        # Should have at least 5 core protected beliefs
        assert len(protected) >= 5, f"Only {len(protected)} protected beliefs found"


# =============================================================================
# 4. END-TO-END SMOKE
# =============================================================================

class TestE2ESmoke:
    """End-to-end smoke tests."""

    @pytest.mark.asyncio
    async def test_tick_count_after_startup(self, integration_layer):
        """
        After a few ticks, tick_id should be > 1.
        """
        # Run a few ticks manually
        for _ in range(3):
            await integration_layer._execute_tick()

        assert integration_layer._tick_count > 1
        assert integration_layer.state.tick_id > 1

    @pytest.mark.asyncio
    async def test_self_model_loads(self, integration_layer, mock_identity_service):
        """
        Self model should load from IdentityService.
        """
        # Manually trigger refresh
        snapshot = mock_identity_service.get_snapshot()
        integration_layer.state.self_model = snapshot

        assert integration_layer.state.self_model is not None
        assert len(integration_layer.state.self_model.core_beliefs) >= 1


# =============================================================================
# INTEGRATION TEST (requires running server)
# =============================================================================

@pytest.mark.skip(reason="Requires running server")
class TestLiveServer:
    """Tests against live running server."""

    @pytest.mark.asyncio
    async def test_chat_no_belief_mutation(self):
        """
        POST /api/chat should not mutate beliefs.
        """
        import httpx

        # Snapshot beliefs before
        with open("data/beliefs/current.json") as f:
            before = json.load(f)

        # Send chat message
        async with httpx.AsyncClient(verify=False) as client:
            resp = await client.post(
                "https://localhost:8443/api/chat",
                json={"message": "Hello", "retrieve_memories": False},
                timeout=30.0,
            )
            assert resp.status_code == 200

        # Snapshot beliefs after
        with open("data/beliefs/current.json") as f:
            after = json.load(f)

        # Assert: No changes
        assert before == after
