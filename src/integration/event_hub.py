"""
Integration Event Hub - Pub/Sub message bus for cross-subsystem communication

Simple in-process event bus. Subsystems publish signals; Integration Layer
and other subscribers receive them.

Based on INTEGRATION_LAYER_SPEC.md Section 3.4.

Usage Pattern (from spec):
    Subsystems SHOULD publish to IntegrationEventHub, not call
    IntegrationLayer.submit_signal() directly. The hub is the standard
    integration path; submit_signal() is reserved for tightly coupled
    components (e.g., awareness loop percept buffer).
"""

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Callable

from .signals import Signal

logger = logging.getLogger(__name__)


class IntegrationEventHub:
    """
    Event bus for cross-subsystem signals.

    Subsystems publish signals; IntegrationLayer and other subscribers receive.
    Supports both sync and async callbacks.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)

    def subscribe(self, topic: str, callback: Callable[[Signal], None]):
        """
        Subscribe to a signal topic.

        Args:
            topic: Topic name (e.g., "percepts", "dissonance", "goals")
            callback: Callable that takes a Signal. Can be sync or async.
        """
        self._subscribers[topic].append(callback)
        logger.debug(f"Subscribed to topic '{topic}': {callback.__name__}")

    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe from a topic."""
        if callback in self._subscribers[topic]:
            self._subscribers[topic].remove(callback)
            logger.debug(f"Unsubscribed from topic '{topic}': {callback.__name__}")

    def publish(self, topic: str, signal):
        """
        Publish a signal to all subscribers of topic (synchronous).

        Errors in individual handlers are caught and logged but don't
        prevent other handlers from running.

        Args:
            topic: Topic name
            signal: Signal instance or dict to publish
        """
        # Handle both Signal objects and plain dicts
        if hasattr(signal, 'signal_id'):
            signal_id = signal.signal_id[:8]
        elif isinstance(signal, dict) and 'signal_id' in signal:
            signal_id = signal['signal_id'][:8]
        else:
            signal_id = "n/a"
        logger.debug(f"Publishing to topic '{topic}': {signal.__class__.__name__} (id={signal_id}...)")

        for callback in self._subscribers[topic]:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Event handler error for topic '{topic}' (handler={callback.__name__}): {e}", exc_info=True)

    def publish_async(self, topic: str, signal):
        """
        Publish signal asynchronously (non-blocking).

        Creates a background task to publish. Useful when publisher
        doesn't want to block waiting for handlers.

        Args:
            topic: Topic name
            signal: Signal instance or dict to publish
        """
        asyncio.create_task(self._async_publish(topic, signal))

    async def _async_publish(self, topic: str, signal):
        """
        Async publish helper.

        Handles both sync and async callbacks. Sync callbacks are run
        in executor to avoid blocking the event loop.
        """
        logger.debug(f"Async publishing to topic '{topic}': {signal.__class__.__name__}")

        for callback in self._subscribers[topic]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    # Run sync callback in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, callback, signal)
            except Exception as e:
                logger.error(f"Async event handler error for topic '{topic}' (handler={callback.__name__}): {e}", exc_info=True)

    def get_topics(self) -> List[str]:
        """Get list of all topics with subscribers."""
        return list(self._subscribers.keys())

    def get_subscriber_count(self, topic: str) -> int:
        """Get number of subscribers for a topic."""
        return len(self._subscribers[topic])

    def clear_topic(self, topic: str):
        """Remove all subscribers from a topic."""
        if topic in self._subscribers:
            count = len(self._subscribers[topic])
            del self._subscribers[topic]
            logger.info(f"Cleared {count} subscribers from topic '{topic}'")

    def clear_all(self):
        """Remove all subscribers from all topics."""
        topic_count = len(self._subscribers)
        total_subs = sum(len(subs) for subs in self._subscribers.values())
        self._subscribers.clear()
        logger.info(f"Cleared all subscriptions: {total_subs} subscribers from {topic_count} topics")
