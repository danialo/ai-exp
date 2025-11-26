"""
Phase 3 Integration Layer Tests - Goal Arbitration & Mode Transitions

Tests four critical Phase 3 areas:
1. Conflict Detection
2. Action Selection & Arbitration
3. Action Dispatch
4. Mode Transitions

Run with: pytest tests/test_integration_layer_phase3.py -v
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.integration.layer import IntegrationLayer
from src.integration.state import (
    AstraState, ExecutionMode, FocusItem, FocusType,
    Action, ActionType, Conflict, ConflictType, GoalHandle,
    SelfModelSnapshot
)
from src.integration.event_hub import IntegrationEventHub


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_event_hub():
    """Create mock IntegrationEventHub."""
    hub = Mock(spec=IntegrationEventHub)
    hub.subscribe = Mock()
    hub.publish = Mock()
    return hub


@pytest.fixture
def mock_identity_service():
    """Create mock IdentityService."""
    mock = Mock()
    mock.get_snapshot.return_value = SelfModelSnapshot(
        core_beliefs=[{"id": "core1"}],
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
def mock_goal_store():
    """Create mock GoalStore that returns test goals."""
    mock = Mock()

    # Create test goals
    goal1 = Mock()
    goal1.id = "goal-1"
    goal1.text = "Learn about Python"
    goal1.value = 0.8
    goal1.effort = 0.3
    goal1.risk = 0.1
    goal1.category = Mock(value="exploration")
    goal1.state = Mock(value="adopted")
    goal1.contradicts = []
    goal1.aligns_with = []

    goal2 = Mock()
    goal2.id = "goal-2"
    goal2.text = "Conserve resources"
    goal2.value = 0.5
    goal2.effort = 0.1
    goal2.risk = 0.0
    goal2.category = Mock(value="maintenance")
    goal2.state = Mock(value="adopted")
    goal2.contradicts = ["goal-1"]  # Conflicts with goal-1
    goal2.aligns_with = []

    mock.list_goals.return_value = [goal1, goal2]
    mock.safe_adopt.return_value = (True, goal1, {})

    return mock


@pytest.fixture
def mock_awareness_loop():
    """Create mock awareness loop."""
    mock = Mock()
    mock.trigger_introspection = AsyncMock(return_value=True)
    mock.running = True
    return mock


@pytest.fixture
def mock_belief_gardener():
    """Create mock belief gardener."""
    mock = Mock()
    mock.run_pattern_scan = Mock()
    return mock


@pytest.fixture
def integration_layer(mock_event_hub, mock_identity_service, mock_goal_store):
    """Create IntegrationLayer with Phase 3 features."""
    il = IntegrationLayer(
        event_hub=mock_event_hub,
        identity_service=mock_identity_service,
        goal_store=mock_goal_store,
        mode=ExecutionMode.INTERACTIVE,
    )
    return il


# =============================================================================
# 1. CONFLICT DETECTION
# =============================================================================

class TestConflictDetection:
    """Tests for Phase 3 conflict detection."""

    def test_goal_goal_conflict_detected(self, integration_layer, mock_goal_store):
        """Goals with explicit contradicts are detected as conflicts."""
        # Act: Detect conflicts
        conflicts = integration_layer._detect_conflicts()

        # Assert: Goal-goal conflict should be detected
        goal_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.GOAL_GOAL]
        assert len(goal_conflicts) == 1
        assert "goal-1" in goal_conflicts[0].involved
        assert "goal-2" in goal_conflicts[0].involved

    def test_no_conflict_when_no_contradicts(self, integration_layer, mock_goal_store):
        """No conflict when goals don't contradict."""
        # Arrange: Remove contradicts
        mock_goal_store.list_goals.return_value[1].contradicts = []

        # Act
        conflicts = integration_layer._detect_conflicts()

        # Assert: No goal-goal conflicts
        goal_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.GOAL_GOAL]
        assert len(goal_conflicts) == 0

    def test_dissonance_goal_conflict(self, integration_layer, mock_goal_store):
        """High-salience dissonance creates conflict with goals."""
        # Arrange: Add high-salience dissonance to focus
        dissonance_item = FocusItem(
            item_type=FocusType.DISSONANCE,
            item_id="dissonance-1",
            content="Belief contradiction",
            salience=0.9,  # High salience
            entered_focus=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            decay_rate=0.05,
            min_salience_threshold=0.1,
        )
        integration_layer.state.focus_stack.append(dissonance_item)

        # Act
        conflicts = integration_layer._detect_conflicts()

        # Assert: Dissonance-goal conflicts detected
        dissonance_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.DISSONANCE_GOAL]
        assert len(dissonance_conflicts) >= 1


# =============================================================================
# 2. ACTION SELECTION & ARBITRATION
# =============================================================================

class TestActionSelection:
    """Tests for Phase 3 action selection."""

    def test_user_response_highest_priority_interactive(self, integration_layer):
        """USER_RESPONSE is highest priority in INTERACTIVE mode."""
        # Arrange: Add user message to focus
        user_item = FocusItem(
            item_type=FocusType.USER_MESSAGE,
            item_id="msg-1",
            content="Hello Astra",
            salience=0.9,
            entered_focus=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            decay_rate=0.01,
            min_salience_threshold=0.1,
        )
        integration_layer.state.focus_stack.append(user_item)
        integration_layer.mode = ExecutionMode.INTERACTIVE

        # Act
        actions = integration_layer._select_actions([])

        # Assert: USER_RESPONSE is selected with highest priority
        assert len(actions) >= 1
        user_actions = [a for a in actions if a.action_type == ActionType.USER_RESPONSE]
        assert len(user_actions) == 1
        assert user_actions[0].priority == 4  # CRITICAL

    def test_goal_pursuit_selected_with_goals(self, integration_layer, mock_goal_store):
        """GOAL_PURSUIT is selected when active goals exist."""
        # Act
        actions = integration_layer._select_actions([])

        # Assert: GOAL_PURSUIT action selected
        goal_actions = [a for a in actions if a.action_type == ActionType.GOAL_PURSUIT]
        assert len(goal_actions) == 1
        assert goal_actions[0].target_id == "goal-1"  # Highest value goal

    def test_dissonance_resolution_high_priority(self, integration_layer):
        """DISSONANCE_RESOLUTION is high priority when dissonance in focus."""
        # Arrange
        dissonance_item = FocusItem(
            item_type=FocusType.DISSONANCE,
            item_id="dis-1",
            content="Contradiction detected",
            salience=0.8,
            entered_focus=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            decay_rate=0.05,
            min_salience_threshold=0.1,
        )
        integration_layer.state.focus_stack.append(dissonance_item)

        # Act
        actions = integration_layer._select_actions([])

        # Assert
        dis_actions = [a for a in actions if a.action_type == ActionType.DISSONANCE_RESOLUTION]
        assert len(dis_actions) == 1
        assert dis_actions[0].priority == 3  # HIGH

    def test_introspection_in_autonomous_mode(self, integration_layer):
        """INTROSPECTION is selected in AUTONOMOUS mode after cooldown."""
        # Arrange
        integration_layer.mode = ExecutionMode.AUTONOMOUS
        integration_layer._tick_count = 100
        integration_layer._last_introspection_tick = 0  # Long cooldown elapsed

        # Act
        actions = integration_layer._select_actions([])

        # Assert
        intro_actions = [a for a in actions if a.action_type == ActionType.INTROSPECTION]
        assert len(intro_actions) == 1

    def test_belief_gardening_in_maintenance_mode(self, integration_layer):
        """BELIEF_GARDENING is selected in MAINTENANCE mode."""
        # Arrange
        integration_layer.mode = ExecutionMode.MAINTENANCE
        integration_layer.goal_store = None  # No goals to pursue

        # Act
        actions = integration_layer._select_actions([])

        # Assert
        garden_actions = [a for a in actions if a.action_type == ActionType.BELIEF_GARDENING]
        assert len(garden_actions) == 1
        assert garden_actions[0].priority == 1  # LOW

    def test_max_three_actions_selected(self, integration_layer):
        """At most 3 actions are selected per tick."""
        # Arrange: Create conditions for many actions
        integration_layer.mode = ExecutionMode.AUTONOMOUS
        integration_layer._tick_count = 100
        integration_layer._last_introspection_tick = 0

        # Add dissonance
        dissonance_item = FocusItem(
            item_type=FocusType.DISSONANCE,
            item_id="dis-1",
            content="Contradiction",
            salience=0.8,
            entered_focus=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            decay_rate=0.05,
            min_salience_threshold=0.1,
        )
        integration_layer.state.focus_stack.append(dissonance_item)

        # Act
        actions = integration_layer._select_actions([])

        # Assert: Max 3 actions
        assert len(actions) <= 3


# =============================================================================
# 3. ACTION DISPATCH
# =============================================================================

class TestActionDispatch:
    """Tests for Phase 3 action dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_introspection(self, integration_layer, mock_awareness_loop):
        """INTROSPECTION action triggers awareness loop."""
        # Arrange
        integration_layer.awareness_loop = mock_awareness_loop
        integration_layer._tick_count = 50

        action = Action(
            action_type=ActionType.INTROSPECTION,
            target_id=None,
            priority=2,
            estimated_cost={"tokens": 200}
        )

        # Act
        await integration_layer._dispatch_actions([action])

        # Assert
        mock_awareness_loop.trigger_introspection.assert_awaited_once()
        assert integration_layer._last_introspection_tick == 50
        assert integration_layer.introspections_today == 1

    @pytest.mark.asyncio
    async def test_dispatch_goal_pursuit(self, integration_layer, mock_event_hub):
        """GOAL_PURSUIT action publishes event."""
        action = Action(
            action_type=ActionType.GOAL_PURSUIT,
            target_id="goal-1",
            priority=2,
            estimated_cost={"tokens": 400}
        )

        # Act
        await integration_layer._dispatch_actions([action])

        # Assert: Event published
        mock_event_hub.publish.assert_called()
        call_args = mock_event_hub.publish.call_args
        assert call_args[0][0] == "goal_pursue"
        assert call_args[0][1]["goal_id"] == "goal-1"

    @pytest.mark.asyncio
    async def test_dispatch_belief_gardening(
        self, integration_layer, mock_belief_gardener
    ):
        """BELIEF_GARDENING action triggers gardener."""
        integration_layer.belief_gardener = mock_belief_gardener

        action = Action(
            action_type=ActionType.BELIEF_GARDENING,
            target_id=None,
            priority=1,
            estimated_cost={}
        )

        # Act
        await integration_layer._dispatch_actions([action])

        # Assert
        mock_belief_gardener.run_pattern_scan.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_user_response(self, integration_layer, mock_event_hub):
        """USER_RESPONSE action publishes event."""
        action = Action(
            action_type=ActionType.USER_RESPONSE,
            target_id="msg-1",
            priority=4,
            estimated_cost={"tokens": 500}
        )

        # Act
        await integration_layer._dispatch_actions([action])

        # Assert
        mock_event_hub.publish.assert_called()
        call_args = mock_event_hub.publish.call_args
        assert call_args[0][0] == "action_selected"
        assert call_args[0][1]["action_type"] == "user_response"

    @pytest.mark.asyncio
    async def test_dispatch_handles_errors(self, integration_layer, mock_awareness_loop):
        """Dispatch handles errors gracefully."""
        # Arrange: Make trigger_introspection raise
        mock_awareness_loop.trigger_introspection = AsyncMock(
            side_effect=Exception("Test error")
        )
        integration_layer.awareness_loop = mock_awareness_loop

        action = Action(
            action_type=ActionType.INTROSPECTION,
            target_id=None,
            priority=2,
            estimated_cost={"tokens": 200}
        )

        # Act: Should not raise
        await integration_layer._dispatch_actions([action])

        # Assert: Introspection counter NOT incremented (failed)
        assert integration_layer.introspections_today == 0


# =============================================================================
# 4. MODE TRANSITIONS
# =============================================================================

class TestModeTransitions:
    """Tests for Phase 3 mode transitions."""

    def test_interactive_to_autonomous_on_inactivity(self, integration_layer):
        """Mode transitions to AUTONOMOUS after inactivity timeout."""
        # Arrange
        integration_layer.mode = ExecutionMode.INTERACTIVE
        integration_layer.state.mode = ExecutionMode.INTERACTIVE
        integration_layer._last_user_interaction = (
            datetime.now() - timedelta(seconds=2000)  # Past timeout
        )

        # Act
        integration_layer._check_mode_transition()

        # Assert
        assert integration_layer.mode == ExecutionMode.AUTONOMOUS

    def test_stays_interactive_with_recent_activity(self, integration_layer):
        """Mode stays INTERACTIVE with recent user activity."""
        # Arrange
        integration_layer.mode = ExecutionMode.INTERACTIVE
        integration_layer._last_user_interaction = datetime.now()

        # Add recent user message to focus
        user_item = FocusItem(
            item_type=FocusType.USER_MESSAGE,
            item_id="msg-1",
            content="Hello",
            salience=0.9,
            entered_focus=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            decay_rate=0.01,
            min_salience_threshold=0.1,
        )
        integration_layer.state.focus_stack.append(user_item)

        # Act
        integration_layer._check_mode_transition()

        # Assert
        assert integration_layer.mode == ExecutionMode.INTERACTIVE

    def test_record_user_interaction_resets_mode(self, integration_layer):
        """Recording user interaction resets mode to INTERACTIVE."""
        # Arrange
        integration_layer.mode = ExecutionMode.AUTONOMOUS

        # Act
        integration_layer.record_user_interaction()

        # Assert
        assert integration_layer.mode == ExecutionMode.INTERACTIVE

    def test_maintenance_mode_in_window(self, integration_layer):
        """Mode transitions to MAINTENANCE during maintenance window."""
        # Arrange
        integration_layer.mode = ExecutionMode.AUTONOMOUS
        integration_layer._last_user_interaction = (
            datetime.now() - timedelta(hours=1)
        )

        # Mock datetime to return 3am (in maintenance window 2-5am)
        with patch('src.integration.layer.datetime') as mock_dt:
            mock_now = Mock()
            mock_now.hour = 3
            mock_now.now.return_value = mock_now
            mock_dt.now.return_value = mock_now

            # Act
            integration_layer._check_mode_transition()

        # Assert
        assert integration_layer.mode == ExecutionMode.MAINTENANCE

    def test_mode_change_publishes_event(self, integration_layer, mock_event_hub):
        """Mode changes publish events."""
        # Arrange
        integration_layer.mode = ExecutionMode.INTERACTIVE
        integration_layer._last_user_interaction = (
            datetime.now() - timedelta(seconds=2000)
        )

        # Act
        integration_layer._check_mode_transition()

        # Assert: mode_changed event published
        mock_event_hub.publish.assert_called()
        call_args = mock_event_hub.publish.call_args
        assert call_args[0][0] == "mode_changed"


# =============================================================================
# 5. GOAL ADOPTION (IL-controlled)
# =============================================================================

class TestGoalAdoption:
    """Tests for IL-controlled goal adoption."""

    @pytest.mark.asyncio
    async def test_adopt_goal_success(self, integration_layer, mock_goal_store, mock_event_hub):
        """Goal adoption through IL works."""
        # Act
        await integration_layer._adopt_goal("goal-1")

        # Assert
        mock_goal_store.safe_adopt.assert_called_once_with("goal-1")
        mock_event_hub.publish.assert_called()

    @pytest.mark.asyncio
    async def test_adopt_goal_no_store(self, integration_layer):
        """Goal adoption handles missing goal_store."""
        # Arrange
        integration_layer.goal_store = None

        # Act (should not crash)
        await integration_layer._adopt_goal("goal-1")

        # Assert: No exception

    @pytest.mark.asyncio
    async def test_adopt_goal_blocked(self, integration_layer, mock_goal_store, mock_event_hub):
        """Blocked goal adoption doesn't publish event."""
        # Arrange: Make safe_adopt return blocked
        mock_goal_store.safe_adopt.return_value = (False, None, {"reason": "contradiction"})

        # Act
        await integration_layer._adopt_goal("goal-1")

        # Assert: goal_adopted event NOT published
        for call in mock_event_hub.publish.call_args_list:
            assert call[0][0] != "goal_adopted"


# =============================================================================
# 6. FULL TICK E2E
# =============================================================================

class TestFullTick:
    """End-to-end tick execution tests."""

    @pytest.mark.asyncio
    async def test_full_tick_executes(self, integration_layer):
        """Full tick executes all phases without error."""
        # Act
        await integration_layer._execute_tick()

        # Assert
        assert integration_layer._tick_count == 1
        assert integration_layer.total_ticks_executed == 1

    @pytest.mark.asyncio
    async def test_tick_with_actions(self, integration_layer, mock_awareness_loop):
        """Tick with actions dispatches them."""
        # Arrange
        integration_layer.mode = ExecutionMode.AUTONOMOUS
        integration_layer.awareness_loop = mock_awareness_loop
        integration_layer._tick_count = 30
        integration_layer._last_introspection_tick = 0
        initial_tick_count = integration_layer._tick_count

        # Act: Run multiple ticks to trigger introspection
        for _ in range(5):
            await integration_layer._execute_tick()

        # Assert: 5 ticks executed (relative to starting point)
        assert integration_layer._tick_count == initial_tick_count + 5
