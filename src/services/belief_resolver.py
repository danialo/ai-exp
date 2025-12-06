"""
Belief Resolver for HTN Self-Belief Decomposer.

3-way concept resolution with concurrency safety:
- match: high similarity, return existing node
- no_match: low similarity, create new node
- uncertain: intermediate, may create TentativeLink
"""

import logging
import time
import uuid as uuid_module
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
from sqlmodel import Session, select

from src.utils.belief_config import BeliefSystemConfig, get_belief_config
from src.services.htn_belief_embedder import HTNBeliefEmbedder
from src.services.belief_canonicalizer import CanonicalAtom
from src.memory.models.belief_node import BeliefNode

logger = logging.getLogger(__name__)


class VectorIndexRequiredError(NotImplementedError):
    """Raised when node count exceeds linear scan threshold."""
    pass


class ConcurrencyError(Exception):
    """Raised when concurrent node creation fails after retries."""
    pass


@dataclass
class ResolutionResult:
    """
    Result of belief resolution.

    Attributes:
        outcome: match, no_match, uncertain
        match_confidence: Similarity score
        matched_node_id: ID of matched node (if match)
        candidate_ids: IDs of top candidates considered
        candidate_similarities: Similarities of candidates
        verifier_used: Whether LLM verifier was called
        verifier_result: Result from verifier if used
    """
    outcome: str  # match, no_match, uncertain
    match_confidence: float
    matched_node_id: Optional[uuid_module.UUID] = None
    candidate_ids: List[uuid_module.UUID] = field(default_factory=list)
    candidate_similarities: List[float] = field(default_factory=list)
    verifier_used: bool = False
    verifier_result: Optional[dict] = None


class BeliefResolver:
    """
    3-way concept resolution with concurrency safety.

    Resolves atoms to existing nodes or creates new ones:
    - match: similarity >= match_threshold
    - no_match: similarity <= no_match_threshold
    - uncertain: in between, may trigger verifier
    """

    def __init__(
        self,
        config: Optional[BeliefSystemConfig] = None,
        embedder: Optional[HTNBeliefEmbedder] = None,
        verifier: Optional[Any] = None,
        db_session: Optional[Session] = None,
    ):
        """
        Initialize the resolver.

        Args:
            config: Configuration object
            embedder: Embedding service
            verifier: Optional LLM verifier
            db_session: Database session
        """
        if config is None:
            config = get_belief_config()

        self.config = config.resolution
        self.embedder = embedder or HTNBeliefEmbedder(config)
        self.verifier = verifier
        self.db = db_session

        self.top_k = config.resolution.top_k
        self.match_threshold = config.resolution.match_threshold
        self.no_match_threshold = config.resolution.no_match_threshold
        self.verifier_enabled = config.resolution.verifier.enabled
        self.trigger_band = config.resolution.verifier.trigger_band

        self.max_retries = config.concurrency.max_retries
        self.retry_delay_ms = config.concurrency.retry_delay_ms

    def resolve(
        self,
        atom: CanonicalAtom,
        embedding: Optional[np.ndarray] = None
    ) -> ResolutionResult:
        """
        Resolve an atom to an existing node or determine it's new.

        Args:
            atom: Canonicalized atom to resolve
            embedding: Pre-computed embedding (optional)

        Returns:
            ResolutionResult with outcome and details
        """
        # Get candidates
        try:
            candidates = self._get_candidates(atom, embedding)
        except VectorIndexRequiredError:
            raise

        if not candidates:
            # No existing nodes, definitely new
            return ResolutionResult(
                outcome='no_match',
                match_confidence=0.0,
            )

        # Find best match
        best_node, best_similarity = candidates[0]
        all_candidate_ids = [c[0].belief_id for c in candidates]
        all_similarities = [c[1] for c in candidates]

        # Determine outcome
        if best_similarity >= self.match_threshold:
            return ResolutionResult(
                outcome='match',
                match_confidence=best_similarity,
                matched_node_id=best_node.belief_id,
                candidate_ids=all_candidate_ids,
                candidate_similarities=all_similarities,
            )

        if best_similarity <= self.no_match_threshold:
            return ResolutionResult(
                outcome='no_match',
                match_confidence=best_similarity,
                candidate_ids=all_candidate_ids,
                candidate_similarities=all_similarities,
            )

        # Uncertain - check if verifier should be called
        if self.verifier_enabled and self.verifier:
            lower, upper = self.trigger_band
            if lower <= best_similarity <= upper:
                verifier_result = self._call_verifier(atom, best_node)

                if verifier_result and verifier_result.same_concept:
                    return ResolutionResult(
                        outcome='match',
                        match_confidence=getattr(verifier_result, 'confidence', best_similarity),
                        matched_node_id=best_node.belief_id,
                        candidate_ids=all_candidate_ids,
                        candidate_similarities=all_similarities,
                        verifier_used=True,
                        verifier_result=verifier_result,
                    )

        # Still uncertain
        return ResolutionResult(
            outcome='uncertain',
            match_confidence=best_similarity,
            candidate_ids=all_candidate_ids,
            candidate_similarities=all_similarities,
            verifier_used=self.verifier is not None and self.verifier_enabled,
        )

    def _get_candidates(
        self,
        atom: CanonicalAtom,
        embedding: Optional[np.ndarray] = None
    ) -> List[Tuple[BeliefNode, float]]:
        """
        Get candidate nodes for matching.

        Uses linear scan if node count is small enough, otherwise
        requires vector index.
        """
        if not self.db:
            return []

        # Count nodes
        node_count = self.db.exec(select(BeliefNode)).all()
        count = len(node_count)

        if not self.embedder.should_use_linear_scan(count):
            raise VectorIndexRequiredError(
                f"Node count ({count}) exceeds linear_scan_max_nodes "
                f"({self.embedder.linear_scan_max}). Configure a vector index."
            )

        # Linear scan approach
        nodes = self.db.exec(select(BeliefNode)).all()

        if not nodes:
            return []

        # Compute similarities
        candidates = []

        if embedding is not None and self.embedder.enabled:
            # Use embeddings
            for node in nodes:
                if node.embedding:
                    node_embedding = self.embedder.deserialize(node.embedding)
                    sim = self.embedder.cosine_similarity(embedding, node_embedding)
                    candidates.append((node, sim))
                else:
                    # Fallback to text similarity
                    sim = self.embedder.text_similarity(
                        atom.canonical_text,
                        node.canonical_text
                    )
                    candidates.append((node, sim))
        else:
            # Use text similarity
            for node in nodes:
                sim = self.embedder.text_similarity(
                    atom.canonical_text,
                    node.canonical_text
                )
                candidates.append((node, sim))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return top_k
        return candidates[:self.top_k]

    def _call_verifier(
        self,
        atom: CanonicalAtom,
        candidate: BeliefNode
    ) -> Optional[dict]:
        """Call the LLM verifier to check semantic match."""
        if not self.verifier:
            return None

        try:
            return self.verifier.verify(
                atom.canonical_text,
                candidate.canonical_text
            )
        except Exception as e:
            logger.warning(f"Verifier call failed: {e}")
            return None

    def create_or_get_node(
        self,
        atom: CanonicalAtom,
        embedding: Optional[np.ndarray],
        resolution: ResolutionResult
    ) -> BeliefNode:
        """
        Create a new node or return existing based on resolution.

        Handles concurrency with unique constraint retries.

        Args:
            atom: Canonicalized atom
            embedding: Pre-computed embedding
            resolution: Resolution result from resolve()

        Returns:
            BeliefNode (existing or new)
        """
        if resolution.outcome == 'match' and resolution.matched_node_id:
            # Return existing node
            if self.db:
                node = self.db.get(BeliefNode, resolution.matched_node_id)
                if node:
                    return node

        # Create new node with concurrency handling
        return self._create_node_safe(atom, embedding)

    def _create_node_safe(
        self,
        atom: CanonicalAtom,
        embedding: Optional[np.ndarray]
    ) -> BeliefNode:
        """
        Create node with concurrency safety.

        Retries on unique constraint violations.
        """
        if not self.db:
            # Return in-memory node if no DB
            return BeliefNode(
                belief_id=uuid_module.uuid4(),
                canonical_text=atom.canonical_text,
                canonical_hash=atom.canonical_hash,
                belief_type=atom.belief_type,
                polarity=atom.polarity,
                embedding=self.embedder.serialize(embedding) if embedding is not None else None,
            )

        for attempt in range(self.max_retries):
            try:
                node = BeliefNode(
                    belief_id=uuid_module.uuid4(),
                    canonical_text=atom.canonical_text,
                    canonical_hash=atom.canonical_hash,
                    belief_type=atom.belief_type,
                    polarity=atom.polarity,
                    embedding=self.embedder.serialize(embedding) if embedding is not None else None,
                )

                self.db.add(node)
                self.db.commit()
                self.db.refresh(node)
                return node

            except Exception as e:
                error_str = str(e).lower()
                if 'unique' in error_str or 'constraint' in error_str:
                    self.db.rollback()

                    # Another process created it - fetch and return
                    existing = self.db.exec(
                        select(BeliefNode).where(
                            BeliefNode.canonical_hash == atom.canonical_hash
                        )
                    ).first()

                    if existing:
                        return existing

                    # Wait before retry
                    time.sleep(self.retry_delay_ms / 1000)
                else:
                    raise

        raise ConcurrencyError(
            f"Failed to create or fetch node after {self.max_retries} attempts"
        )


def resolve_atom(
    atom: CanonicalAtom,
    embedding: Optional[np.ndarray] = None,
    db_session: Optional[Session] = None
) -> ResolutionResult:
    """
    Convenience function to resolve an atom.

    Args:
        atom: Canonicalized atom
        embedding: Pre-computed embedding
        db_session: Database session

    Returns:
        ResolutionResult
    """
    resolver = BeliefResolver(db_session=db_session)
    return resolver.resolve(atom, embedding)
