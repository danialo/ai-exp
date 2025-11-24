"""
Integration Layer Signals - Normalized messages from subsystems

Signal taxonomy for cross-subsystem communication via IntegrationEventHub.
All signals inherit from base Signal class and can convert to FocusItem.

Based on INTEGRATION_LAYER_SPEC.md Section 2.3 and Appendix A.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from .state import FocusItem, FocusType


class Priority(Enum):
    """Signal priority levels."""
    CRITICAL = 4
    HIGH = 3
    NORMAL = 2
    LOW = 1


@dataclass
class Signal(ABC):
    """
    Base class for all signals from subsystems to Integration Layer.

    Signals are normalized messages that flow through IntegrationEventHub.
    Each signal can optionally convert to a FocusItem if attention-worthy.
    """

    signal_id: str
    source: str  # Subsystem that generated signal
    timestamp: datetime
    priority: Priority

    @abstractmethod
    def to_focus_item(self) -> Optional[FocusItem]:
        """Convert signal to focus item, if attention-worthy."""
        pass


@dataclass
class PerceptSignal(Signal):
    """
    Signal from awareness loop (CIL layer in Iida's model).

    Represents observations from fast/slow ticks: user messages, token
    consumption, tool usage, time passage, system events.
    """

    percept_type: str  # "user", "token", "tool", "time", "sys"
    content: Any
    novelty: float  # 0.0-1.0 from awareness loop computation
    entropy: float  # 0.0-1.0 information density

    def to_focus_item(self) -> Optional[FocusItem]:
        """
        High-novelty percepts (>0.7) become focus items.
        User messages always get high salience.
        """
        if self.novelty < 0.7 and self.percept_type != "user":
            return None

        # User messages get very high base salience
        salience = 0.9 if self.percept_type == "user" else self.novelty

        return FocusItem(
            item_type=FocusType.USER_MESSAGE if self.percept_type == "user" else FocusType.EXTERNAL_EVENT,
            item_id=self.signal_id,
            content=str(self.content)[:200],  # Truncate for display
            salience=salience,
            entered_focus=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            decay_rate=0.01,
            min_salience_threshold=0.3
        )


@dataclass
class DissonanceSignal(Signal):
    """
    Signal from belief consistency checker (PPL layer in Iida's model).

    Represents identity conflicts detected between beliefs and memories,
    or between different beliefs.
    """

    pattern: str  # "alignment", "contradiction", "hedging", "external_imposition"
    belief_id: str
    conflicting_memory: str
    severity: float  # 0.0-1.0

    def to_focus_item(self) -> Optional[FocusItem]:
        """
        High-severity dissonance (>0.5) always enters focus.
        Identity threats are urgent and demand attention.
        """
        if self.severity < 0.5:
            return None

        return FocusItem(
            item_type=FocusType.DISSONANCE,
            item_id=self.signal_id,
            content=f"Dissonance: {self.pattern} (severity={self.severity:.2f})",
            salience=0.8 + (self.severity * 0.2),  # 0.8-1.0 range
            entered_focus=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            decay_rate=0.005,  # Slower decay - dissonance lingers
            min_salience_threshold=0.4
        )


@dataclass
class GoalProposal(Signal):
    """
    Signal proposing a new goal for adoption.

    Can come from user, system, belief gardener, or other subsystems.
    IL arbitrates which goals to adopt.
    """

    goal: Any  # GoalDefinition (placeholder for now)
    rationale: str
    proposer: str  # "USER", "SYSTEM", "BELIEF_GARDENER", etc.

    def to_focus_item(self) -> Optional[FocusItem]:
        """
        Goal proposals from users get high salience.
        System proposals get moderate salience.
        """
        salience = 0.8 if self.proposer == "USER" else 0.5

        return FocusItem(
            item_type=FocusType.GOAL,
            item_id=self.signal_id,
            content=f"Goal proposal: {self.rationale[:100]}",
            salience=salience,
            entered_focus=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            decay_rate=0.02,
            min_salience_threshold=0.3
        )


@dataclass
class IntegrationEvent(Signal):
    """
    Cross-subsystem event requiring coordination.

    Generic event type for system lifecycle events: belief formed,
    goal satisfied, task completed, identity updated, etc.
    """

    event_type: str  # "belief_formed", "goal_satisfied", "task_completed", etc.
    payload: dict

    def to_focus_item(self) -> Optional[FocusItem]:
        """
        Most integration events don't enter focus directly.
        They update workspace state but don't demand attention.
        """
        # Only major events enter focus
        if self.event_type in ["identity_updated", "major_dissonance_resolved"]:
            return FocusItem(
                item_type=FocusType.INTROSPECTION,
                item_id=self.signal_id,
                content=f"Event: {self.event_type}",
                salience=0.6,
                entered_focus=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                decay_rate=0.03,
                min_salience_threshold=0.3
            )

        return None
