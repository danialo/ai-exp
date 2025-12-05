"""
Memory models package.

This package re-exports all models from the original models.py
plus the new HTN Self-Belief Decomposer models.
"""

# Re-export everything from the original models.py
# Using a relative import to the parent module's models.py
import sys
import importlib.util

# Load the original models.py file directly since it's shadowed by this package
_models_path = __file__.replace('/models/__init__.py', '/models.py')
_spec = importlib.util.spec_from_file_location("_original_models", _models_path)
_original_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original_models)

# Re-export all original models
ExperienceType = _original_models.ExperienceType
Actor = _original_models.Actor
CaptureMethod = _original_models.CaptureMethod
EmbeddingRole = _original_models.EmbeddingRole
Stage = _original_models.Stage
SessionStatus = _original_models.SessionStatus
TraitType = _original_models.TraitType
TraitStability = _original_models.TraitStability
ContentModel = _original_models.ContentModel
ProvenanceSource = _original_models.ProvenanceSource
ProvenanceModel = _original_models.ProvenanceModel
ConfidenceModel = _original_models.ConfidenceModel
EmbeddingPointers = _original_models.EmbeddingPointers
VAD = _original_models.VAD
AffectModel = _original_models.AffectModel
ExperienceModel = _original_models.ExperienceModel
SignatureEmbeddingModel = _original_models.SignatureEmbeddingModel
AffectSnapshotModel = _original_models.AffectSnapshotModel
Experience = _original_models.Experience
SignatureEmbedding = _original_models.SignatureEmbedding
AffectSnapshot = _original_models.AffectSnapshot
Session = _original_models.Session
MemoryDecayMetrics = _original_models.MemoryDecayMetrics
experience_to_model = _original_models.experience_to_model
model_to_experience = _original_models.model_to_experience
signature_embedding_to_model = _original_models.signature_embedding_to_model
model_to_signature_embedding = _original_models.model_to_signature_embedding
affect_snapshot_to_model = _original_models.affect_snapshot_to_model
model_to_affect_snapshot = _original_models.model_to_affect_snapshot

# New HTN Self-Belief Decomposer models
from src.memory.models.belief_node import BeliefNode
from src.memory.models.belief_occurrence import BeliefOccurrence
from src.memory.models.tentative_link import TentativeLink
from src.memory.models.conflict_edge import ConflictEdge
from src.memory.models.stream_assignment import StreamAssignment

__all__ = [
    # Original models
    'ExperienceType',
    'Actor',
    'CaptureMethod',
    'EmbeddingRole',
    'Stage',
    'SessionStatus',
    'TraitType',
    'TraitStability',
    'ContentModel',
    'ProvenanceSource',
    'ProvenanceModel',
    'ConfidenceModel',
    'EmbeddingPointers',
    'VAD',
    'AffectModel',
    'ExperienceModel',
    'SignatureEmbeddingModel',
    'AffectSnapshotModel',
    'Experience',
    'SignatureEmbedding',
    'AffectSnapshot',
    'Session',
    'MemoryDecayMetrics',
    'experience_to_model',
    'model_to_experience',
    'signature_embedding_to_model',
    'model_to_signature_embedding',
    'affect_snapshot_to_model',
    'model_to_affect_snapshot',
    # Belief models
    'BeliefNode',
    'BeliefOccurrence',
    'TentativeLink',
    'ConflictEdge',
    'StreamAssignment',
]
