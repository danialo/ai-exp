"""
Integration Layer - Placeholder implementation for Phase 1

Phase 1 (Read-only observer): Simple integration layer that subscribes to
signals and accumulates them in AstraState. No executive loop or action
dispatch yet - just passive observation and state accumulation.

Based on INTEGRATION_LAYER_SPEC.md Section 3.1.

Full executive loop (8 phases) will be added in Phase 2.
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime

from .state import AstraState, ExecutionMode
from .signals import Signal, PerceptSignal, DissonanceSignal
from .event_hub import IntegrationEventHub
from .identity_service import IdentityService

logger = logging.getLogger(__name__)


class IntegrationLayer:
    """
    Placeholder Integration Layer for Phase 1.

    Current capabilities:
    - Subscribe to percepts and dissonance topics
    - Accumulate signals in AstraState buffers
    - Provide read-only access to current state
    - Update self_model snapshot periodically

    Future (Phase 2+):
    - Executive loop with 8 phases
    - Focus management and salience computation
    - Action selection and dispatch
    - Budget enforcement
    """

    def __init__(
        self,
        event_hub: IntegrationEventHub,
        identity_service: Optional[IdentityService] = None,
        mode: ExecutionMode = ExecutionMode.INTERACTIVE
    ):
        """
        Initialize Integration Layer.

        Args:
            event_hub: Event bus for subscribing to signals
            identity_service: Optional PIM facade for self-model (Phase 1: can be None)
            mode: Execution mode (INTERACTIVE/AUTONOMOUS/MAINTENANCE)
        """
        self.event_hub = event_hub
        self.identity_service = identity_service
        self.mode = mode

        # Global Workspace
        self.state = AstraState(mode=mode)

        # Background task handle
        self._task: Optional[asyncio.Task] = None

        # Phase 1.5: Simple metrics
        self.total_percepts_seen = 0
        self.total_dissonance_seen = 0
        self.last_signal_timestamp: Optional[datetime] = None

        if identity_service:
            logger.info(f"IntegrationLayer initialized (Phase 1: read-only observer, mode={mode.value}, identity_service=enabled)")
        else:
            logger.info(f"IntegrationLayer initialized (Phase 1: read-only observer, mode={mode.value}, identity_service=disabled)")

    async def start(self):
        """
        Start the Integration Layer.

        Phase 1: Subscribe to signals and start periodic self-model refresh.
        Phase 2+: Will add executive loop here.
        """
        logger.info("Starting IntegrationLayer (Phase 1)")

        # Subscribe to signal topics
        self.event_hub.subscribe("percepts", self._on_percept)
        self.event_hub.subscribe("dissonance", self._on_dissonance)

        logger.info("Subscribed to topics: percepts, dissonance")

        # Start background task for periodic self-model update (only if identity_service available)
        if self.identity_service:
            self._task = asyncio.create_task(self._self_model_refresh_loop())
            logger.info("Self-model refresh loop started")
        else:
            logger.info("Self-model refresh loop disabled (no identity_service)")

        logger.info("IntegrationLayer started successfully")

    async def stop(self):
        """Stop the Integration Layer and cleanup."""
        logger.info("Stopping IntegrationLayer")

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("IntegrationLayer stopped")

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
            "percept_buffer_size": len(self.state.percept_buffer),
            "dissonance_alert_count": len(self.state.dissonance_alerts),
            "focus_stack_size": len(self.state.focus_stack),
            "active_goals": len(self.state.active_goals),
            "self_model_loaded": self.state.self_model is not None,
            "last_update": self.state.timestamp.isoformat() if self.state.timestamp else None,
            # Phase 1.5 metrics
            "total_percepts_seen": self.total_percepts_seen,
            "total_dissonance_seen": self.total_dissonance_seen,
            "last_signal_timestamp": self.last_signal_timestamp.isoformat() if self.last_signal_timestamp else None,
        }
