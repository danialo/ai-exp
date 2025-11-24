"""
Integration Layer State - Global Workspace and Snapshots

This module defines AstraState (the rich in-memory Global Workspace) and
AstraStateSnapshot (the lightweight JSON-serializable representation).

Based on INTEGRATION_LAYER_SPEC.md Section 1.2 and 2.1.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Deque
from collections import deque
from enum import Enum


class ExecutionMode(Enum):
    """Execution mode for Integration Layer tick scheduling."""
    INTERACTIVE = "interactive"     # User interaction active
    AUTONOMOUS = "autonomous"       # No user, self-directed
    MAINTENANCE = "maintenance"     # Background consolidation/cleanup


class FocusType(Enum):
    """Type of item in focus stack."""
    USER_MESSAGE = "user_message"
    GOAL = "goal"
    DISSONANCE = "dissonance"
    INTROSPECTION = "introspection"
    TASK = "task"
    MEMORY = "memory"
    EXTERNAL_EVENT = "external_event"


@dataclass
class FocusItem:
    """Single item of attention in the focus stack."""

    item_type: FocusType
    item_id: str  # Reference to underlying object
    content: str  # Human-readable description

    salience: float  # 0.0-1.0, computed by focus algorithm
    entered_focus: datetime
    last_accessed: datetime
    access_count: int

    # Decay parameters
    decay_rate: float  # How quickly salience decreases
    min_salience_threshold: float  # When to evict from stack


@dataclass
class BudgetStatus:
    """Current budget allocation and usage."""

    # Token budgets (LLM usage)
    tokens_per_minute_limit: int = 2000
    tokens_used_this_minute: int = 0
    tokens_available: int = 2000

    # Belief gardener budgets (daily)
    beliefs_form_limit: int = 3
    beliefs_formed_today: int = 0
    beliefs_promote_limit: int = 5
    beliefs_promoted_today: int = 0
    beliefs_deprecate_limit: int = 3
    beliefs_deprecated_today: int = 0

    # URL fetch budget (per conversation)
    url_fetch_limit: int = 3
    url_fetches_this_session: int = 0

    # Introspection frequency (cooldown)
    min_introspection_interval_seconds: float = 30.0
    last_introspection: Optional[datetime] = None

    # Dissonance event cooldown (per belief)
    dissonance_cooldown_minutes: int = 120
    dissonance_cooldown_map: Dict[str, datetime] = field(default_factory=dict)

    # Computational resources (0.0-1.0)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0


@dataclass
class SelfModelSnapshot:
    """
    Immutable snapshot of Astra's identity state.

    Computed by IdentityService from fragmented stores. Provides unified
    read-only view of beliefs, traits, anchors, and capabilities.

    Note: Anchors stored as list[float] for serialization compatibility.
    Convert to numpy only where needed for computation.
    """

    # Core Identity
    core_beliefs: List[Any]  # List[Belief] - immutable ontological beliefs
    peripheral_beliefs: List[Any]  # List[Belief] - mutable experiential beliefs
    traits: Dict[str, float]  # Personality traits

    # Identity Anchors (from awareness loop)
    origin_anchor: List[float]  # Baseline identity vector
    live_anchor: List[float]  # Current evolved identity
    anchor_drift: float  # Distance between origin and live

    # Capabilities
    known_capabilities: set  # What Astra knows it can do
    limitations: set  # What Astra knows it cannot do

    # Self-Assessment
    confidence_self_model: float  # 0.0-1.0, how well Astra knows itself
    last_major_update: Optional[datetime]  # When self-model significantly changed

    # Metadata
    snapshot_id: str
    created_at: datetime


@dataclass
class AstraState:
    """
    The Global Workspace - rich in-memory representation of Astra's
    current conscious state.

    This is what the Integration Layer and subsystems work with during
    execution. For persistence, convert to AstraStateSnapshot.
    """

    # Identity
    self_model: Optional[SelfModelSnapshot] = None

    # Attention
    focus_stack: List[FocusItem] = field(default_factory=list)
    attention_capacity: float = 1.0  # 0.0-1.0

    # Goals & Tasks (placeholder types for now)
    active_goals: List[Any] = field(default_factory=list)  # List[GoalHandle]
    goal_queue: Any = None  # PriorityQueue[GoalProposal]
    task_context: Dict[str, Any] = field(default_factory=dict)  # Dict[str, TaskNode]

    # Signals & Events (placeholder types for now)
    percept_buffer: Deque[Any] = field(default_factory=deque)  # Deque[PerceptSignal]
    dissonance_alerts: List[Any] = field(default_factory=list)  # List[DissonanceSignal]
    integration_events: List[Any] = field(default_factory=list)  # List[IntegrationEvent]

    # Modulation & Affect
    emotional_state: Optional[Any] = None  # EmotionalStateVector (VAD model)
    arousal_level: float = 0.0  # 0.0-1.0
    cognitive_load: float = 0.0  # 0.0-1.0

    # Budgets & Resources
    budget_status: BudgetStatus = field(default_factory=BudgetStatus)

    # Temporal
    tick_id: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    last_introspection: Optional[datetime] = None

    # Metadata
    mode: ExecutionMode = ExecutionMode.INTERACTIVE
    session_id: Optional[str] = None

    def to_snapshot(self) -> 'AstraStateSnapshot':
        """
        Convert rich AstraState to lightweight AstraStateSnapshot for persistence.

        Converts:
        - Full objects → references/IDs
        - Buffers → counts
        - datetime → ISO strings
        """
        return AstraStateSnapshot(
            # Identity (simplified)
            self_model_id=self.self_model.snapshot_id if self.self_model else None,
            core_belief_count=len(self.self_model.core_beliefs) if self.self_model else 0,
            peripheral_belief_count=len(self.self_model.peripheral_beliefs) if self.self_model else 0,
            anchor_drift=self.self_model.anchor_drift if self.self_model else 0.0,

            # Attention (serializable)
            focus_stack=[
                {
                    'type': f.item_type.value,
                    'content': f.content,
                    'salience': f.salience
                }
                for f in self.focus_stack
            ],
            attention_capacity=self.attention_capacity,

            # Goals & Tasks (references only)
            active_goal_ids=[getattr(g, 'id', str(g)) for g in self.active_goals],
            pending_goal_count=self.goal_queue.qsize() if self.goal_queue else 0,
            active_task_ids=list(self.task_context.keys()),

            # Signals (counts, not full buffers)
            percept_buffer_size=len(self.percept_buffer),
            dissonance_alert_count=len(self.dissonance_alerts),
            integration_event_count=len(self.integration_events),

            # Modulation & Affect (serializable)
            emotional_state={
                'valence': getattr(self.emotional_state, 'valence', 0.0),
                'arousal': getattr(self.emotional_state, 'arousal', 0.0),
                'dominance': getattr(self.emotional_state, 'dominance', 0.0)
            } if self.emotional_state else None,
            arousal_level=self.arousal_level,
            cognitive_load=self.cognitive_load,

            # Budgets (serializable)
            tokens_available=self.budget_status.tokens_available,
            tokens_used_this_minute=self.budget_status.tokens_used_this_minute,
            beliefs_formed_today=self.budget_status.beliefs_formed_today,

            # Temporal (ISO strings)
            tick_id=self.tick_id,
            timestamp=self.timestamp.isoformat(),
            last_introspection=self.last_introspection.isoformat() if self.last_introspection else None,

            # Metadata
            mode=self.mode.value,
            session_id=self.session_id
        )


@dataclass
class AstraStateSnapshot:
    """
    Serializable snapshot of AstraState for persistence.

    Lightweight, JSON-safe representation that can be persisted to Redis/JSON.
    Use AstraState.to_snapshot() to create.
    """

    # Identity (simplified)
    self_model_id: Optional[str]
    core_belief_count: int
    peripheral_belief_count: int
    anchor_drift: float

    # Attention (serializable)
    focus_stack: List[Dict[str, Any]]
    attention_capacity: float

    # Goals & Tasks (references only)
    active_goal_ids: List[str]
    pending_goal_count: int
    active_task_ids: List[str]

    # Signals (counts, not full buffers)
    percept_buffer_size: int
    dissonance_alert_count: int
    integration_event_count: int

    # Modulation & Affect (serializable)
    emotional_state: Optional[Dict[str, float]]
    arousal_level: float
    cognitive_load: float

    # Budgets (serializable)
    tokens_available: int
    tokens_used_this_minute: int
    beliefs_formed_today: int

    # Temporal (ISO strings)
    tick_id: int
    timestamp: str
    last_introspection: Optional[str]

    # Metadata
    mode: str
    session_id: Optional[str]
