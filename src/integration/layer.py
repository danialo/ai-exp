"""
Integration Layer - Phase 2 Implementation

Phase 2 (Executive Loop): Integration Layer that runs a tick-based executive loop
to integrate signals, compute focus, enforce budgets, and dispatch actions.

Based on INTEGRATION_LAYER_SPEC.md Section 4.

Executive Loop Phases:
1. COLLECT SIGNALS - Gather all signals from subsystems
2. UPDATE WORKSPACE - Integrate signals into AstraState
3. COMPUTE FOCUS - Update focus stack based on salience
4. DETECT CONFLICTS - Identify conflicting goals/actions (Phase 3)
5. APPLY BUDGETS - Check resource availability
6. SELECT ACTIONS - Choose actions via arbitration (Phase 3)
7. DISPATCH ACTIONS - Execute selected actions
8. PERSIST SNAPSHOT - Save state

Phase 2 implements: Phases 1-3, 5, 7 (introspection only), 8
"""

import asyncio
import logging
import json
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.awareness_loop import AwarenessLoop
from datetime import datetime
from pathlib import Path
from collections import deque

from .state import AstraState, ExecutionMode, FocusItem
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

    def __init__(
        self,
        event_hub: IntegrationEventHub,
        identity_service: Optional[IdentityService] = None,
        mode: ExecutionMode = ExecutionMode.INTERACTIVE,
        awareness_loop: Optional[Any] = None,
        belief_gardener: Optional[Any] = None,
        snapshot_dir: Optional[Path] = None,
    ):
        """
        Initialize Integration Layer.

        Args:
            event_hub: Event bus for subscribing to signals
            identity_service: PIM facade for self-model
            mode: Execution mode (INTERACTIVE/AUTONOMOUS/MAINTENANCE)
            awareness_loop: Optional reference for triggering introspection
            belief_gardener: Optional reference for triggering belief gardening
            snapshot_dir: Directory for persisting snapshots
        """
        self.event_hub = event_hub
        self.identity_service = identity_service
        self.mode = mode
        self.awareness_loop = awareness_loop
        self.belief_gardener = belief_gardener
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

        logger.info(
            f"IntegrationLayer initialized (Phase 2: executive loop, mode={mode.value}, "
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
        """Execute one tick of the Executive Loop."""
        self._tick_count += 1
        self.total_ticks_executed += 1
        self.state.tick_id = self._tick_count
        self.state.timestamp = datetime.now()

        # === PHASE 1: COLLECT SIGNALS ===
        signals = self._collect_signals()

        # === PHASE 2: UPDATE WORKSPACE ===
        await self._update_workspace(signals)

        # === PHASE 3: COMPUTE FOCUS ===
        self._compute_focus()

        # === PHASE 4: DETECT CONFLICTS (Phase 3 feature - skip) ===
        # conflicts = await self._detect_conflicts()

        # === PHASE 5: APPLY BUDGETS ===
        budget_ok = self._check_budgets()

        # === PHASE 6: SELECT ACTIONS (Phase 3 feature - skip) ===
        # actions = await self._select_actions(conflicts)

        # === PHASE 7: DISPATCH ACTIONS (limited in Phase 2) ===
        if budget_ok:
            await self._dispatch_introspection_if_needed()

        # === PHASE 8: PERSIST SNAPSHOT ===
        if self._tick_count - self._last_snapshot_tick >= self.SNAPSHOT_INTERVAL_TICKS:
            await self._persist_snapshot()
            self._last_snapshot_tick = self._tick_count

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
        return FocusItem(
            content=f"percept:{signal.percept_type}",
            source=signal.source,
            salience=signal.novelty,
            timestamp=signal.timestamp,
            decay_rate=0.1,  # Percepts decay faster
        )

    def _dissonance_to_focus_item(self, signal: DissonanceSignal) -> Optional[FocusItem]:
        """Convert dissonance signal to focus item."""
        return FocusItem(
            content=f"dissonance:{signal.pattern}",
            source=signal.source,
            salience=signal.severity,  # High severity = high salience
            timestamp=signal.timestamp,
            decay_rate=0.05,  # Dissonance decays slower (needs resolution)
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
            age_seconds = (now - item.timestamp).total_seconds()

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
    # PHASE 7: DISPATCH ACTIONS (Phase 2: Introspection only)
    # =========================================================================

    async def _dispatch_introspection_if_needed(self):
        """
        Trigger introspection if conditions are met.

        Phase 2: IL controls when introspection happens, replacing
        awareness_loop's internal scheduler.
        """
        # Check if enough ticks have passed
        ticks_since_introspection = self._tick_count - self._last_introspection_tick
        if ticks_since_introspection < self.INTROSPECTION_INTERVAL_TICKS:
            return

        # Check if awareness_loop is available
        if not self.awareness_loop:
            return

        # Check if awareness_loop has trigger_introspection method
        if not hasattr(self.awareness_loop, 'trigger_introspection'):
            logger.debug("[IL] awareness_loop missing trigger_introspection - skipping")
            return

        # Trigger introspection
        try:
            logger.info(f"[IL] Triggering introspection (tick {self._tick_count})")
            await self.awareness_loop.trigger_introspection()
            self._last_introspection_tick = self._tick_count
            self.introspections_today += 1
        except Exception as e:
            logger.error(f"[IL] Failed to trigger introspection: {e}")

    # =========================================================================
    # PHASE 8: PERSIST SNAPSHOT
    # =========================================================================

    async def _persist_snapshot(self):
        """
        Save current state to disk for debugging/recovery.

        Writes JSON snapshot to snapshot_dir.
        """
        try:
            snapshot = {
                "tick_id": self._tick_count,
                "timestamp": datetime.now().isoformat(),
                "mode": self.state.mode.value,
                "focus_stack": [
                    {
                        "content": f.content,
                        "salience": f.salience,
                        "source": f.source,
                    }
                    for f in self.state.focus_stack
                ],
                "cognitive_load": self.state.cognitive_load,
                "active_goals": len(self.state.active_goals),
                "self_model_loaded": self.state.self_model is not None,
                "metrics": {
                    "total_percepts_seen": self.total_percepts_seen,
                    "total_dissonance_seen": self.total_dissonance_seen,
                    "introspections_today": self.introspections_today,
                },
            }

            # Write to latest.json (overwrite)
            snapshot_file = self.snapshot_dir / "latest.json"
            snapshot_file.write_text(json.dumps(snapshot, indent=2))

            logger.debug(f"[IL] Snapshot persisted at tick {self._tick_count}")

        except Exception as e:
            logger.error(f"[IL] Failed to persist snapshot: {e}")
