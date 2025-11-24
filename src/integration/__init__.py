"""
Integration Layer Package

Astra's cognitive integration system - the executive function that orchestrates
subsystems, maintains Global Workspace, and implements Iida's CIL (Cognitive
Integration Layer).

Phase 1 Components (Read-only workspace and signal wiring):
- AstraState: Rich in-memory Global Workspace
- AstraStateSnapshot: Lightweight serializable snapshot
- Signal hierarchy: Normalized messages from subsystems
- IntegrationEventHub: Pub/sub message bus
- IdentityService: Unified PIM facade (read-only for Phase 1)

Based on INTEGRATION_LAYER_SPEC.md

This is purely vibe coded.
"""

# Core state
from .state import (
    AstraState,
    AstraStateSnapshot,
    SelfModelSnapshot,
    FocusItem,
    FocusType,
    BudgetStatus,
    ExecutionMode,
)

# Signal taxonomy
from .signals import (
    Signal,
    Priority,
    PerceptSignal,
    DissonanceSignal,
    GoalProposal,
    IntegrationEvent,
)

# Infrastructure
from .event_hub import IntegrationEventHub
from .identity_service import IdentityService
from .layer import IntegrationLayer

__all__ = [
    # State
    "AstraState",
    "AstraStateSnapshot",
    "SelfModelSnapshot",
    "FocusItem",
    "FocusType",
    "BudgetStatus",
    "ExecutionMode",
    # Signals
    "Signal",
    "Priority",
    "PerceptSignal",
    "DissonanceSignal",
    "GoalProposal",
    "IntegrationEvent",
    # Infrastructure
    "IntegrationEventHub",
    "IdentityService",
    "IntegrationLayer",
]
