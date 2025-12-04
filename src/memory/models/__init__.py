"""
HTN Self-Belief Decomposer database models.

This module provides SQLModel definitions for the belief extraction system:
- BeliefNode: Canonical belief concepts
- BeliefOccurrence: Evidence events linking nodes to source experiences
- TentativeLink: Uncertain identity resolution between nodes
- ConflictEdge: Contradiction/tension relationships
- StreamAssignment: Soft stream assignments with migration tracking
"""

from src.memory.models.belief_node import BeliefNode
from src.memory.models.belief_occurrence import BeliefOccurrence
from src.memory.models.tentative_link import TentativeLink
from src.memory.models.conflict_edge import ConflictEdge
from src.memory.models.stream_assignment import StreamAssignment

__all__ = [
    'BeliefNode',
    'BeliefOccurrence',
    'TentativeLink',
    'ConflictEdge',
    'StreamAssignment',
]
