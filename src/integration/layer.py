"""
Integration Layer - Phase 3 Implementation

Phase 3 (Full Arbitration): Integration Layer with complete goal arbitration,
conflict detection, action selection, and mode transitions.

Based on INTEGRATION_LAYER_SPEC.md Section 4 and 7.

Executive Loop Phases:
1. COLLECT SIGNALS - Gather all signals from subsystems
2. UPDATE WORKSPACE - Integrate signals into AstraState
3. COMPUTE FOCUS - Update focus stack based on salience
4. DETECT CONFLICTS - Identify conflicting goals/actions
5. APPLY BUDGETS - Check resource availability
6. SELECT ACTIONS - Choose actions via arbitration
7. DISPATCH ACTIONS - Execute selected actions
8. PERSIST SNAPSHOT - Save state

Phase 3 implements: All 8 phases + mode transitions + goal arbitration
"""

import asyncio
import logging
import json
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.awareness_loop import AwarenessLoop
    from src.services.goal_store import GoalStore
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

from .state import (
    AstraState, ExecutionMode, FocusItem, FocusType,
    Action, ActionType, Conflict, ConflictType, GoalHandle
)
from .signals import Signal, PerceptSignal, DissonanceSignal, Priority
from .event_hub import IntegrationEventHub
from .identity_service import IdentityService

logger = logging.getLogger(__name__)


class IntegrationLayer:
    """
    Integration Layer - Phase 2 Implementation.

    Executive loop that runs at configurable tick rate to:
    - Collect and integrate signals from subsystems
    - Compute focus stack based on salience
    - Enforce budgets on actions
    - Dispatch actions (introspection, belief gardening)

    Phase 2 capabilities:
    - Executive loop with phases 1-3, 5, 7, 8
    - Focus stack management (Miller's Law: max 7 items)
    - Budget tracking and enforcement
    - Introspection triggering (replaces awareness_loop's internal scheduler)
    - Snapshot persistence

    Phase 3+ (future):
    - Conflict detection (phase 4)
    - Full action selection (phase 6)
    - Goal arbitration
    - Mode transitions
    """

    # Executive loop configuration
    TICK_INTERVAL_SECONDS = 1.0  # Run at 1 Hz
    FOCUS_STACK_MAX_SIZE = 7    # Miller's Law
    SNAPSHOT_INTERVAL_TICKS = 10  # Persist every 10 ticks
    INTROSPECTION_INTERVAL_TICKS = 30  # Trigger introspection every ~30 seconds

    # Phase 3: Mode transition configuration
    INACTIVITY_TIMEOUT_SECONDS = 1800  # 30 min inactivity → AUTONOMOUS
    MAINTENANCE_WINDOW_HOURS = (2, 5)  # 2am-5am → MAINTENANCE mode

    def __init__(
        self,
        event_hub: IntegrationEventHub,
        identity_service: Optional[IdentityService] = None,
        mode: ExecutionMode = ExecutionMode.INTERACTIVE,
        awareness_loop: Optional[Any] = None,
        belief_gardener: Optional[Any] = None,
        belief_consolidator: Optional[Any] = None,
        goal_store: Optional[Any] = None,
        snapshot_dir: Optional[Path] = None,
        # Phase 4: Memory consolidation components
        session_consolidator: Optional[Any] = None,
        insight_extractor: Optional[Any] = None,
        memory_pruner: Optional[Any] = None,
        temporal_anchor: Optional[Any] = None,
    ):
        """
        Initialize Integration Layer.

        Args:
            event_hub: Event bus for subscribing to signals
            identity_service: PIM facade for self-model
            mode: Execution mode (INTERACTIVE/AUTONOMOUS/MAINTENANCE)
            awareness_loop: Optional reference for triggering introspection
            belief_gardener: Optional reference for triggering belief gardening
            belief_consolidator: Optional reference for LLM-based belief consolidation
            goal_store: Optional reference for goal approval
            snapshot_dir: Directory for persisting snapshots
            session_consolidator: Phase 4 - session to narrative compression
            insight_extractor: Phase 4 - pattern extraction from narratives
            memory_pruner: Phase 4 - decay-based memory management
            temporal_anchor: Phase 4 - identity continuity tracking
        """
        self.event_hub = event_hub
        self.identity_service = identity_service
        self.mode = mode
        self.awareness_loop = awareness_loop
        self.belief_gardener = belief_gardener
        self.belief_consolidator = belief_consolidator
        self.goal_store = goal_store

        # Phase 4: Memory consolidation
        self.session_consolidator = session_consolidator
        self.insight_extractor = insight_extractor
        self.memory_pruner = memory_pruner
        self.temporal_anchor = temporal_anchor
        self.snapshot_dir = snapshot_dir or Path("data/integration_snapshots")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Global Workspace
        self.state = AstraState(mode=mode)

        # Executive loop state
        self._tick_count = 0
        self._last_introspection_tick = 0
        self._last_snapshot_tick = 0
        self._running = False
        self._executive_task: Optional[asyncio.Task] = None
        self._self_model_task: Optional[asyncio.Task] = None

        # Metrics
        self.total_percepts_seen = 0
        self.total_dissonance_seen = 0
        self.total_ticks_executed = 0
        self.last_signal_timestamp: Optional[datetime] = None

        # Budget tracking (simple version - Phase 2)
        self.introspections_today = 0
        self.introspection_daily_budget = 50  # Max introspections per day

        # Consolidation tracking (runs less frequently than gardening)
        self._last_consolidation_tick = 0
        self.consolidation_interval_ticks = 720  # Every ~6 hours at 30s ticks

        # Phase 4: Memory consolidation tracking
        self._last_memory_consolidation_tick = 0
        self.memory_consolidation_interval_ticks = 7200  # Every ~2 hours (7200 * 1s)
        self._consolidation_running = False

        # Phase 3: Mode transition tracking
        self._last_user_interaction: datetime = datetime.now()
        self._mode_lock = False  # Prevent rapid mode flapping

        # Phase 3: Goal tracking
        self._goal_proposal_queue: List[str] = []  # Goal IDs awaiting adoption
        self._pending_conflicts: List[Conflict] = []

        # Phase 5: Tick history for observability
        self.tick_history: deque = deque(maxlen=1000)
        self.action_log: deque = deque(maxlen=1000)

        logger.info(
            f"IntegrationLayer initialized (Phase 3: full arbitration, mode={mode.value}, "
            f"tick_interval={self.TICK_INTERVAL_SECONDS}s)"
        )

    async def start(self):
        """
        Start the Integration Layer.

        Phase 2: Start executive loop and subscribe to signals.
        """
        logger.info("Starting IntegrationLayer (Phase 2: Executive Loop)")

        # Subscribe to signal topics
        self.event_hub.subscribe("percepts", self._on_percept)
        self.event_hub.subscribe("dissonance", self._on_dissonance)
        logger.info("Subscribed to topics: percepts, dissonance")

        # Start executive loop
        self._running = True
        self._executive_task = asyncio.create_task(self._executive_loop())
        logger.info(f"Executive loop started (tick_interval={self.TICK_INTERVAL_SECONDS}s)")

        # Start self-model refresh (separate from executive loop for reliability)
        if self.identity_service:
            self._self_model_task = asyncio.create_task(self._self_model_refresh_loop())
            logger.info("Self-model refresh loop started")

        logger.info("IntegrationLayer started successfully")

    async def stop(self):
        """Stop the Integration Layer and cleanup."""
        logger.info("Stopping IntegrationLayer")

        self._running = False

        # Cancel executive loop
        if self._executive_task:
            self._executive_task.cancel()
            try:
                await self._executive_task
            except asyncio.CancelledError:
                pass

        # Cancel self-model refresh
        if self._self_model_task:
            self._self_model_task.cancel()
            try:
                await self._self_model_task
            except asyncio.CancelledError:
                pass

        # Final snapshot
        await self._persist_snapshot()

        logger.info(f"IntegrationLayer stopped after {self._tick_count} ticks")

    def _on_percept(self, signal: PerceptSignal):
        """
        Handle incoming percept signal.

        Phase 1: Just accumulate in buffer.
        Phase 2+: Will compute salience and update focus stack.
        """
        self.state.percept_buffer.append(signal)
        self.total_percepts_seen += 1
        self.last_signal_timestamp = signal.timestamp

        # Log high-priority percepts
        if signal.priority.value >= 3:  # HIGH or CRITICAL
            logger.debug(f"[IL] Percept received: {signal.percept_type} (priority={signal.priority.value}, novelty={signal.novelty:.2f})")

    def _on_dissonance(self, signal: DissonanceSignal):
        """
        Handle incoming dissonance signal.

        Phase 1: Just accumulate in alerts list.
        Phase 2+: Will add to focus stack and trigger resolution.
        """
        self.state.dissonance_alerts.append(signal)
        self.total_dissonance_seen += 1
        self.last_signal_timestamp = signal.timestamp

        # Phase 1.5: Log high-severity dissonance for early visibility
        if signal.severity >= 0.6:
            logger.warning(f"[IL] HIGH DISSONANCE: {signal.pattern} (severity={signal.severity:.2f}, belief_id={signal.belief_id})")
        else:
            logger.info(f"[IL] Dissonance alert: {signal.pattern} (severity={signal.severity:.2f}, belief_id={signal.belief_id})")

    async def _self_model_refresh_loop(self):
        """
        Periodically refresh self-model snapshot from IdentityService.

        Runs every 5 seconds to keep self_model current.
        Phase 2+: Will be integrated into executive loop tick.
        """
        consecutive_failures = 0
        max_backoff = 60  # Max 60s backoff on repeated failures

        while True:
            try:
                # Base interval + backoff on failures
                sleep_time = 5 + min(consecutive_failures * 5, max_backoff)
                await asyncio.sleep(sleep_time)

                # Refresh self-model
                snapshot = self.identity_service.get_snapshot()
                self.state.self_model = snapshot
                self.state.timestamp = datetime.now()

                # Reset failure counter on success
                if consecutive_failures > 0:
                    logger.info(f"[IL] Self-model refresh recovered after {consecutive_failures} failures")
                    consecutive_failures = 0

                logger.debug(f"[IL] Self-model refreshed: {len(snapshot.core_beliefs)} core beliefs, drift={snapshot.anchor_drift:.3f}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures <= 3:
                    logger.error(f"[IL] Error refreshing self-model (attempt {consecutive_failures}): {e}", exc_info=True)
                else:
                    # After 3 failures, log without traceback to avoid spam
                    logger.warning(f"[IL] Self-model refresh still failing (attempt {consecutive_failures}): {e}")

    def get_state(self) -> AstraState:
        """
        Get current Global Workspace state (read-only).

        Returns:
            Current AstraState
        """
        return self.state

    def get_stats(self) -> dict:
        """
        Get current statistics about IL state.

        Useful for debugging and monitoring.
        """
        return {
            "mode": self.state.mode.value,
            "tick_count": self._tick_count,
            "percept_buffer_size": len(self.state.percept_buffer),
            "dissonance_alert_count": len(self.state.dissonance_alerts),
            "focus_stack_size": len(self.state.focus_stack),
            "active_goals": len(self.state.active_goals),
            "self_model_loaded": self.state.self_model is not None,
            "last_update": self.state.timestamp.isoformat() if self.state.timestamp else None,
            # Phase 2 metrics
            "total_percepts_seen": self.total_percepts_seen,
            "total_dissonance_seen": self.total_dissonance_seen,
            "total_ticks_executed": self.total_ticks_executed,
            "introspections_today": self.introspections_today,
            "last_signal_timestamp": self.last_signal_timestamp.isoformat() if self.last_signal_timestamp else None,
        }

    # =========================================================================
    # EXECUTIVE LOOP - Phase 2 Implementation
    # =========================================================================

    async def _executive_loop(self):
        """
        Main executive loop - runs continuously at TICK_INTERVAL_SECONDS.

        Executes phases 1-3, 5, 7 (introspection), 8 on each tick.
        Phase 4 (conflicts) and 6 (action selection) are Phase 3 features.
        """
        logger.info("Executive loop starting")

        while self._running:
            try:
                tick_start = datetime.now()
                await self._execute_tick()
                tick_duration = (datetime.now() - tick_start).total_seconds()

                # Log every 10 ticks at debug level
                if self._tick_count % 10 == 0:
                    logger.debug(
                        f"[IL] Tick {self._tick_count}: duration={tick_duration:.3f}s, "
                        f"focus={len(self.state.focus_stack)}, "
                        f"percepts={len(self.state.percept_buffer)}"
                    )

                # Sleep for remainder of tick interval
                sleep_time = max(0, self.TICK_INTERVAL_SECONDS - tick_duration)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                logger.info("Executive loop cancelled")
                break
            except Exception as e:
                logger.error(f"[IL] Error in executive loop tick {self._tick_count}: {e}", exc_info=True)
                await asyncio.sleep(self.TICK_INTERVAL_SECONDS)  # Sleep and retry

        logger.info("Executive loop stopped")

    async def _execute_tick(self):
        """Execute one tick of the Executive Loop (Phase 3: full arbitration)."""
        tick_start = datetime.now()
        self._tick_count += 1
        self.total_ticks_executed += 1
        self.state.tick_id = self._tick_count
        self.state.timestamp = tick_start

        # === PHASE 0: MODE TRANSITION CHECK ===
        self._check_mode_transition()

        # === PHASE 1: COLLECT SIGNALS ===
        signals = self._collect_signals()

        # === PHASE 2: UPDATE WORKSPACE ===
        await self._update_workspace(signals)

        # === PHASE 3: COMPUTE FOCUS ===
        self._compute_focus()

        # === PHASE 4: DETECT CONFLICTS ===
        conflicts = self._detect_conflicts()
        self._pending_conflicts = conflicts

        # === PHASE 5: APPLY BUDGETS ===
        budget_ok = self._check_budgets()

        # === PHASE 6: SELECT ACTIONS ===
        actions = self._select_actions(conflicts) if budget_ok else []

        # === PHASE 7: DISPATCH ACTIONS ===
        if actions:
            await self._dispatch_actions(actions)

        # === PHASE 8: PERSIST SNAPSHOT ===
        if self._tick_count - self._last_snapshot_tick >= self.SNAPSHOT_INTERVAL_TICKS:
            await self._persist_snapshot()
            self._last_snapshot_tick = self._tick_count

        # === PHASE 9: RECORD HISTORY ===
        tick_duration = (datetime.now() - tick_start).total_seconds()
        self.tick_history.append({
            "tick_id": self._tick_count,
            "timestamp": tick_start.isoformat(),
            "duration_ms": round(tick_duration * 1000, 2),
            "signals_processed": len(signals),
            "actions_dispatched": len(actions),
            "mode": self.mode.value,
            "focus_stack_size": len(self.state.focus_stack),
            "cognitive_load": self.state.cognitive_load,
        })

        for action in actions:
            self.action_log.append({
                "tick_id": self._tick_count,
                "timestamp": tick_start.isoformat(),
                "action_type": action.action_type.value,
                "target_id": action.target_id,
            })

    # =========================================================================
    # PHASE 1: COLLECT SIGNALS
    # =========================================================================

    def _collect_signals(self) -> List[Signal]:
        """
        Collect all pending signals from buffers.

        Drains percept_buffer and dissonance_alerts into a sorted list.
        """
        signals: List[Signal] = []

        # Drain percept buffer (it's a deque, convert to list)
        while self.state.percept_buffer:
            signals.append(self.state.percept_buffer.popleft())

        # Collect dissonance alerts
        signals.extend(self.state.dissonance_alerts)
        self.state.dissonance_alerts.clear()

        # Sort by priority (highest first)
        signals.sort(key=lambda s: s.priority.value, reverse=True)

        return signals

    # =========================================================================
    # PHASE 2: UPDATE WORKSPACE
    # =========================================================================

    async def _update_workspace(self, signals: List[Signal]):
        """
        Integrate signals into Global Workspace.

        Updates focus stack for attention-worthy signals.
        """
        for signal in signals:
            await self._integrate_signal(signal)

        # Update cognitive load estimate
        self.state.cognitive_load = self._estimate_cognitive_load()

    async def _integrate_signal(self, signal: Signal):
        """Integrate a single signal into workspace."""
        if isinstance(signal, PerceptSignal):
            # High-novelty percepts become focus items
            if signal.novelty > 0.7:
                focus_item = self._percept_to_focus_item(signal)
                if focus_item:
                    self._add_to_focus(focus_item)

        elif isinstance(signal, DissonanceSignal):
            # All dissonance signals become focus items (they're important)
            focus_item = self._dissonance_to_focus_item(signal)
            if focus_item:
                self._add_to_focus(focus_item)

    def _percept_to_focus_item(self, signal: PerceptSignal) -> Optional[FocusItem]:
        """Convert high-novelty percept to focus item."""
        from .state import FocusType
        now = datetime.now()
        return FocusItem(
            item_type=FocusType.USER_MESSAGE if signal.percept_type == "user" else FocusType.EXTERNAL_EVENT,
            item_id=signal.signal_id,
            content=f"percept:{signal.percept_type}",
            salience=signal.novelty,
            entered_focus=now,
            last_accessed=now,
            access_count=1,
            decay_rate=0.1,  # Percepts decay faster
            min_salience_threshold=0.1,
        )

    def _dissonance_to_focus_item(self, signal: DissonanceSignal) -> Optional[FocusItem]:
        """Convert dissonance signal to focus item."""
        from .state import FocusType
        now = datetime.now()
        return FocusItem(
            item_type=FocusType.DISSONANCE,
            item_id=signal.signal_id,
            content=f"dissonance:{signal.pattern}",
            salience=signal.severity,  # High severity = high salience
            entered_focus=now,
            last_accessed=now,
            access_count=1,
            decay_rate=0.05,  # Dissonance decays slower (needs resolution)
            min_salience_threshold=0.1,
        )

    def _add_to_focus(self, item: FocusItem):
        """Add item to focus stack, respecting Miller's Law (max 7 items)."""
        if len(self.state.focus_stack) >= self.FOCUS_STACK_MAX_SIZE:
            # Evict least salient item
            self.state.focus_stack.sort(key=lambda x: x.salience)
            evicted = self.state.focus_stack.pop(0)
            logger.debug(f"[IL] Focus evicted: {evicted.content} (salience={evicted.salience:.2f})")

        self.state.focus_stack.append(item)

    def _estimate_cognitive_load(self) -> float:
        """
        Estimate current cognitive load (0-1).

        Simple heuristic based on focus stack and pending signals.
        """
        focus_load = len(self.state.focus_stack) / self.FOCUS_STACK_MAX_SIZE
        goal_load = min(1.0, len(self.state.active_goals) / 5)  # Assume 5 goals is high
        return min(1.0, (focus_load + goal_load) / 2)

    # =========================================================================
    # PHASE 3: COMPUTE FOCUS
    # =========================================================================

    def _compute_focus(self):
        """
        Update salience scores with decay and re-sort focus stack.

        Salience decays over time; items that fall below threshold are evicted.
        """
        now = datetime.now()
        surviving_items = []

        for item in self.state.focus_stack:
            # Calculate time since item was added
            age_seconds = (now - item.entered_focus).total_seconds()

            # Apply decay: salience decreases over time
            decayed_salience = item.salience * (1 - item.decay_rate * (age_seconds / 60))

            # Keep items above minimum threshold
            if decayed_salience > 0.1:
                item.salience = decayed_salience
                surviving_items.append(item)
            else:
                logger.debug(f"[IL] Focus decayed out: {item.content}")

        # Re-sort by salience (highest first)
        surviving_items.sort(key=lambda x: x.salience, reverse=True)
        self.state.focus_stack = surviving_items

    def get_focus_top_k(self, k: int = 3) -> List[FocusItem]:
        """Get top K items from focus stack."""
        return self.state.focus_stack[:k]

    # =========================================================================
    # PHASE 5: APPLY BUDGETS
    # =========================================================================

    def _check_budgets(self) -> bool:
        """
        Check if budgets allow actions this tick.

        Returns True if actions are allowed.
        """
        # Check introspection budget
        if self.introspections_today >= self.introspection_daily_budget:
            return False

        return True

    # =========================================================================
    # PHASE 0: MODE TRANSITION (Phase 3)
    # =========================================================================

    def _check_mode_transition(self):
        """
        Check if mode should transition based on activity and time.

        Transitions:
        - INTERACTIVE → AUTONOMOUS: 30 min inactivity
        - AUTONOMOUS → MAINTENANCE: 2am-5am window
        - MAINTENANCE → INTERACTIVE: User activity
        - Any → INTERACTIVE: User message received
        """
        now = datetime.now()

        # Check for maintenance window (2am-5am)
        hour = now.hour
        in_maintenance_window = self.MAINTENANCE_WINDOW_HOURS[0] <= hour < self.MAINTENANCE_WINDOW_HOURS[1]

        # User activity resets to INTERACTIVE
        if self._has_recent_user_activity():
            if self.mode != ExecutionMode.INTERACTIVE:
                self._transition_mode(ExecutionMode.INTERACTIVE)
            return

        # Maintenance window takes priority
        if in_maintenance_window and self.mode != ExecutionMode.MAINTENANCE:
            self._transition_mode(ExecutionMode.MAINTENANCE)
            return

        # Inactivity timeout → AUTONOMOUS
        if self.mode == ExecutionMode.INTERACTIVE:
            seconds_since_user = (now - self._last_user_interaction).total_seconds()
            if seconds_since_user > self.INACTIVITY_TIMEOUT_SECONDS:
                self._transition_mode(ExecutionMode.AUTONOMOUS)

    def _has_recent_user_activity(self) -> bool:
        """Check if there's been recent user activity (within last 60s)."""
        # Check focus stack for recent user messages
        for item in self.state.focus_stack:
            if item.item_type == FocusType.USER_MESSAGE:
                age_seconds = (datetime.now() - item.entered_focus).total_seconds()
                if age_seconds < 60:
                    return True
        return False

    def _transition_mode(self, new_mode: ExecutionMode):
        """Transition to a new execution mode."""
        old_mode = self.mode
        self.mode = new_mode
        self.state.mode = new_mode
        logger.info(f"[IL] Mode transition: {old_mode.value} → {new_mode.value}")

        # Publish mode change event (non-critical, don't crash on failure)
        try:
            self.event_hub.publish("mode_changed", {
                "from": old_mode.value,
                "to": new_mode.value,
                "tick": self._tick_count,
            })
        except Exception as e:
            logger.error(f"[IL] Failed to publish mode_changed event: {e}")

    def record_user_interaction(self):
        """Record that user interaction occurred (call from chat handler)."""
        self._last_user_interaction = datetime.now()
        if self.mode != ExecutionMode.INTERACTIVE:
            self._transition_mode(ExecutionMode.INTERACTIVE)

    # =========================================================================
    # PHASE 4: DETECT CONFLICTS (Phase 3)
    # =========================================================================

    def _detect_conflicts(self) -> List[Conflict]:
        """
        Detect conflicts between goals, tasks, and constraints.

        Conflict types:
        - Goal-goal: Two goals contradict
        - Goal-constraint: Goal violates safety constraint
        - Dissonance-goal: Belief contradicts active goal
        """
        conflicts: List[Conflict] = []

        # Get active goals as GoalHandles
        active_goals = self._get_active_goals()

        # Goal-goal conflicts
        for i, goal_a in enumerate(active_goals):
            for goal_b in active_goals[i + 1:]:
                if self._goals_conflict(goal_a, goal_b):
                    conflicts.append(Conflict(
                        conflict_type=ConflictType.GOAL_GOAL,
                        involved=[goal_a.id, goal_b.id],
                        severity=0.6,
                        description=f"Goals '{goal_a.text[:30]}' and '{goal_b.text[:30]}' contradict"
                    ))

        # Dissonance-goal conflicts (check dissonance focus items)
        dissonance_items = [
            f for f in self.state.focus_stack
            if f.item_type == FocusType.DISSONANCE
        ]
        for dissonance in dissonance_items:
            for goal in active_goals:
                # If dissonance relates to goal (simple heuristic: shared terms)
                if self._dissonance_affects_goal(dissonance, goal):
                    conflicts.append(Conflict(
                        conflict_type=ConflictType.DISSONANCE_GOAL,
                        involved=[dissonance.item_id, goal.id],
                        severity=dissonance.salience,
                        description=f"Dissonance may affect goal '{goal.text[:30]}'"
                    ))

        if conflicts:
            logger.info(f"[IL] Detected {len(conflicts)} conflicts")

        return conflicts

    def _get_active_goals(self) -> List[GoalHandle]:
        """Get active goals from goal_store as GoalHandles."""
        if not self.goal_store:
            return []

        try:
            # Get adopted/executing goals from store
            goals = self.goal_store.list_goals(
                states=["adopted", "executing"],
                include_deleted=False
            )
            return [
                GoalHandle(
                    id=g.id,
                    text=g.text,
                    value=g.value,
                    effort=g.effort,
                    risk=g.risk,
                    category=g.category.value if hasattr(g.category, 'value') else str(g.category),
                    state=g.state.value if hasattr(g.state, 'value') else str(g.state),
                    contradicts=g.contradicts or [],
                    aligns_with=g.aligns_with or [],
                )
                for g in goals
            ]
        except Exception as e:
            logger.error(f"[IL] Failed to get active goals: {e}")
            return []

    def _goals_conflict(self, goal_a: GoalHandle, goal_b: GoalHandle) -> bool:
        """Check if two goals contradict each other."""
        # Check explicit contradicts list
        if goal_b.id in goal_a.contradicts:
            return True
        if goal_a.id in goal_b.contradicts:
            return True
        return False

    def _dissonance_affects_goal(self, dissonance: FocusItem, goal: GoalHandle) -> bool:
        """Check if dissonance might affect goal (simple heuristic)."""
        # For now, assume high-severity dissonance affects all goals
        if dissonance.salience > 0.8:
            return True
        return False

    # =========================================================================
    # PHASE 6: SELECT ACTIONS (Phase 3)
    # =========================================================================

    def _select_actions(self, conflicts: List[Conflict]) -> List[Action]:
        """
        Select actions to execute this tick.

        Arbitration strategy:
        1. USER_RESPONSE has highest priority in INTERACTIVE mode
        2. DISSONANCE_RESOLUTION is high priority
        3. GOAL_PURSUIT is normal priority
        4. INTROSPECTION in AUTONOMOUS mode
        5. BELIEF_GARDENING in MAINTENANCE mode
        """
        candidate_actions: List[Action] = []

        # === USER RESPONSE (highest priority in INTERACTIVE mode) ===
        if self.mode == ExecutionMode.INTERACTIVE:
            user_focus = [
                f for f in self.state.focus_stack
                if f.item_type == FocusType.USER_MESSAGE
            ]
            if user_focus:
                candidate_actions.append(Action(
                    action_type=ActionType.USER_RESPONSE,
                    target_id=user_focus[0].item_id,
                    priority=4,  # CRITICAL
                    estimated_cost={"tokens": 500}
                ))

        # === DISSONANCE RESOLUTION (high priority) ===
        dissonance_focus = [
            f for f in self.state.focus_stack
            if f.item_type == FocusType.DISSONANCE
        ]
        if dissonance_focus and self.introspections_today < self.introspection_daily_budget:
            candidate_actions.append(Action(
                action_type=ActionType.DISSONANCE_RESOLUTION,
                target_id=dissonance_focus[0].item_id,
                priority=3,  # HIGH
                estimated_cost={"tokens": 300}
            ))

        # === GOAL PURSUIT (normal priority) ===
        active_goals = self._get_active_goals()
        if active_goals:
            # Select highest-value goal
            top_goal = max(active_goals, key=lambda g: g.value)
            candidate_actions.append(Action(
                action_type=ActionType.GOAL_PURSUIT,
                target_id=top_goal.id,
                priority=2,  # NORMAL
                estimated_cost={"tokens": 400},
                metadata={"goal_text": top_goal.text}
            ))

        # === INTROSPECTION (in AUTONOMOUS mode) ===
        if self.mode == ExecutionMode.AUTONOMOUS:
            ticks_since_introspection = self._tick_count - self._last_introspection_tick
            if ticks_since_introspection >= self.INTROSPECTION_INTERVAL_TICKS:
                if self.introspections_today < self.introspection_daily_budget:
                    candidate_actions.append(Action(
                        action_type=ActionType.INTROSPECTION,
                        target_id=None,
                        priority=2,  # NORMAL
                        estimated_cost={"tokens": 200}
                    ))

        # === BELIEF GARDENING (in MAINTENANCE mode) ===
        if self.mode == ExecutionMode.MAINTENANCE:
            candidate_actions.append(Action(
                action_type=ActionType.BELIEF_GARDENING,
                target_id=None,
                priority=1,  # LOW
                estimated_cost={}
            ))

            # === MEMORY CONSOLIDATION (Phase 4, in MAINTENANCE mode) ===
            ticks_since_mem_consolidation = self._tick_count - self._last_memory_consolidation_tick
            if ticks_since_mem_consolidation >= self.memory_consolidation_interval_ticks:
                if not self._consolidation_running:
                    candidate_actions.append(Action(
                        action_type=ActionType.MEMORY_CONSOLIDATION,
                        target_id=None,
                        priority=1,  # LOW - background task
                        estimated_cost={"tokens": 500}
                    ))

        # === CONFLICT RESOLUTION ===
        if conflicts:
            candidate_actions = self._filter_conflicting_actions(candidate_actions, conflicts)

        # Sort by priority (highest first) and take top 3
        candidate_actions.sort(key=lambda a: a.priority, reverse=True)
        selected = candidate_actions[:3]

        if selected:
            logger.debug(f"[IL] Selected actions: {[a.action_type.value for a in selected]}")

        return selected

    def _filter_conflicting_actions(
        self, actions: List[Action], conflicts: List[Conflict]
    ) -> List[Action]:
        """Remove actions that conflict with higher-priority actions."""
        # For now, simple: if goal-goal conflict exists, only pursue higher-value goal
        # More sophisticated logic would go here
        return actions

    # =========================================================================
    # PHASE 7: DISPATCH ACTIONS (Phase 3)
    # =========================================================================

    async def _dispatch_actions(self, actions: List[Action]):
        """
        Execute selected actions by invoking subsystems.

        Dispatch targets:
        - USER_RESPONSE → Event (PersonaService listens)
        - GOAL_PURSUIT → Event (TaskExecutor listens)
        - INTROSPECTION → awareness_loop.trigger_introspection()
        - DISSONANCE_RESOLUTION → Event (BeliefChecker listens)
        - BELIEF_GARDENING → belief_gardener.run_pattern_scan()
        """
        for action in actions:
            try:
                await self._dispatch_single_action(action)
            except Exception as e:
                logger.error(f"[IL] Action dispatch failed: {action.action_type.value}, error: {e}")

    async def _dispatch_single_action(self, action: Action):
        """Dispatch a single action to appropriate subsystem."""

        if action.action_type == ActionType.USER_RESPONSE:
            # Publish event for PersonaService (it handles the actual response)
            self.event_hub.publish("action_selected", {
                "action_type": "user_response",
                "target_id": action.target_id,
                "tick": self._tick_count,
            })
            logger.debug(f"[IL] Dispatched USER_RESPONSE for {action.target_id}")

        elif action.action_type == ActionType.INTROSPECTION:
            # Trigger awareness loop introspection
            if self.awareness_loop and hasattr(self.awareness_loop, 'trigger_introspection'):
                try:
                    await self.awareness_loop.trigger_introspection()
                    self._last_introspection_tick = self._tick_count
                    self.introspections_today += 1
                    now = datetime.now()
                    self.state.last_introspection = now
                    self.state.budget_status.last_introspection = now  # For cooldown tracking
                    logger.info(f"[IL] Dispatched INTROSPECTION (tick {self._tick_count})")
                except Exception as e:
                    logger.error(f"[IL] Introspection failed: {e}")

        elif action.action_type == ActionType.GOAL_PURSUIT:
            # Publish event for HTN planner / task executor
            self.event_hub.publish("goal_pursue", {
                "goal_id": action.target_id,
                "tick": self._tick_count,
            })
            logger.info(f"[IL] Dispatched GOAL_PURSUIT for {action.target_id}")

        elif action.action_type == ActionType.DISSONANCE_RESOLUTION:
            # Publish event for belief consistency checker
            self.event_hub.publish("dissonance_resolve", {
                "dissonance_id": action.target_id,
                "tick": self._tick_count,
            })
            logger.info(f"[IL] Dispatched DISSONANCE_RESOLUTION for {action.target_id}")

        elif action.action_type == ActionType.BELIEF_GARDENING:
            # Trigger belief gardener (pattern-based)
            if self.belief_gardener and hasattr(self.belief_gardener, 'run_pattern_scan'):
                try:
                    self.belief_gardener.run_pattern_scan()
                    logger.info(f"[IL] Dispatched BELIEF_GARDENING (pattern scan)")
                except Exception as e:
                    logger.error(f"[IL] Belief gardening failed: {e}")

            # Run consolidation if interval elapsed (LLM-based, less frequent)
            ticks_since_consolidation = self._tick_count - self._last_consolidation_tick
            if (
                self.belief_consolidator
                and hasattr(self.belief_consolidator, 'consolidate_beliefs')
                and ticks_since_consolidation >= self.consolidation_interval_ticks
            ):
                try:
                    result = self.belief_consolidator.consolidate_beliefs()
                    self._last_consolidation_tick = self._tick_count
                    logger.info(f"[IL] Dispatched BELIEF_CONSOLIDATION: {result.get('candidates_found', 0)} candidates, "
                               f"{result.get('conflicts_detected', 0)} conflicts")
                except Exception as e:
                    logger.error(f"[IL] Belief consolidation failed: {e}")

        elif action.action_type == ActionType.GOAL_ADOPTION:
            # Adopt a proposed goal
            await self._adopt_goal(action.target_id)

        elif action.action_type == ActionType.MEMORY_CONSOLIDATION:
            # Phase 4: Run full memory consolidation pipeline
            await self._run_memory_consolidation()

    async def _run_memory_consolidation(self):
        """
        Phase 4: Memory consolidation pipeline.

        Runs during MAINTENANCE mode:
        1. Consolidate ended sessions into narratives
        2. Extract insights from recent narratives
        3. Prune decayed memories
        4. Update temporal anchor for identity continuity
        """
        if self._consolidation_running:
            logger.debug("[IL] Memory consolidation already running, skipping")
            return

        self._consolidation_running = True

        try:
            logger.info("[IL] Starting memory consolidation pipeline...")
            results = {
                "sessions_consolidated": 0,
                "insights_extracted": 0,
                "memories_pruned": 0,
                "anchor_updated": False,
            }

            # Step 1: Consolidate ended sessions
            if self.session_consolidator:
                try:
                    session_result = await self.session_consolidator.consolidate_all_pending()
                    results["sessions_consolidated"] = session_result.get("consolidated", 0)
                    logger.info(f"[IL] Session consolidation: {results['sessions_consolidated']} sessions")
                except Exception as e:
                    logger.error(f"[IL] Session consolidation failed: {e}")

            # Step 2: Extract insights from narratives
            new_insights = []
            if self.insight_extractor:
                try:
                    insight_result = await self.insight_extractor.extract_and_store()
                    results["insights_extracted"] = insight_result.get("patterns_stored", 0)
                    new_insights = insight_result.get("pattern_ids", [])
                    logger.info(f"[IL] Insight extraction: {results['insights_extracted']} patterns")
                except Exception as e:
                    logger.error(f"[IL] Insight extraction failed: {e}")

            # Step 3: Prune decayed memories
            if self.memory_pruner:
                try:
                    prune_result = await self.memory_pruner.prune()
                    results["memories_pruned"] = prune_result.get("deleted", 0) + prune_result.get("archived", 0)
                    logger.info(f"[IL] Memory pruning: {results['memories_pruned']} memories")
                except Exception as e:
                    logger.error(f"[IL] Memory pruning failed: {e}")

            # Step 4: Update temporal anchor
            if self.temporal_anchor and new_insights:
                try:
                    # Get narrative texts for anchor update
                    narrative_texts = []
                    if self.session_consolidator and hasattr(self.session_consolidator, 'raw_store'):
                        for insight_id in new_insights[:5]:
                            exp = self.session_consolidator.raw_store.get_experience(insight_id)
                            if exp and exp.content:
                                narrative_texts.append(exp.content.text)

                    if narrative_texts:
                        anchor_result = await self.temporal_anchor.update_after_consolidation(narrative_texts)
                        results["anchor_updated"] = True
                        results["anchor_drift"] = anchor_result.get("drift_from_origin", 0)
                        logger.info(f"[IL] Anchor update: drift={results.get('anchor_drift', 0):.3f}")
                except Exception as e:
                    logger.error(f"[IL] Temporal anchor update failed: {e}")

            # Update tracking
            self._last_memory_consolidation_tick = self._tick_count

            # Publish event
            self.event_hub.publish("memory_consolidation_complete", {
                "tick": self._tick_count,
                **results,
            })

            logger.info(f"[IL] Memory consolidation complete: {results}")

        finally:
            self._consolidation_running = False

    async def _adopt_goal(self, goal_id: str):
        """Adopt a proposed goal through goal_store."""
        if not self.goal_store:
            logger.warning("[IL] Cannot adopt goal: no goal_store")
            return

        try:
            adopted, goal, details = self.goal_store.safe_adopt(goal_id)
            if adopted:
                logger.info(f"[IL] Adopted goal: {goal_id}")
                self.event_hub.publish("goal_adopted", {
                    "goal_id": goal_id,
                    "tick": self._tick_count,
                })
            else:
                logger.warning(f"[IL] Goal adoption blocked: {details}")
        except Exception as e:
            logger.error(f"[IL] Goal adoption failed: {e}")

    # =========================================================================
    # PHASE 8: PERSIST SNAPSHOT
    # =========================================================================

    async def _persist_snapshot(self):
        """
        Save current state to disk for debugging/recovery.

        Writes JSON snapshot to snapshot_dir.
        """
        try:
            # Get active goals for snapshot
            active_goals = self._get_active_goals()

            snapshot = {
                "tick_id": self._tick_count,
                "timestamp": datetime.now().isoformat(),
                "mode": self.state.mode.value,
                "focus_stack": [
                    {
                        "type": f.item_type.value,
                        "content": f.content,
                        "salience": f.salience,
                        "item_id": f.item_id,
                    }
                    for f in self.state.focus_stack
                ],
                "cognitive_load": self.state.cognitive_load,
                "active_goals": [
                    {"id": g.id, "text": g.text[:50], "value": g.value}
                    for g in active_goals
                ],
                "pending_conflicts": len(self._pending_conflicts),
                "self_model_loaded": self.state.self_model is not None,
                "metrics": {
                    "total_percepts_seen": self.total_percepts_seen,
                    "total_dissonance_seen": self.total_dissonance_seen,
                    "introspections_today": self.introspections_today,
                    "total_ticks_executed": self.total_ticks_executed,
                },
            }

            # Write to latest.json (overwrite)
            snapshot_file = self.snapshot_dir / "latest.json"
            snapshot_file.write_text(json.dumps(snapshot, indent=2))

            logger.debug(f"[IL] Snapshot persisted at tick {self._tick_count}")

        except Exception as e:
            logger.error(f"[IL] Failed to persist snapshot: {e}")
