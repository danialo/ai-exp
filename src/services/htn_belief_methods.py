"""
HTN Belief Methods - Orchestration Layer for HTN Self-Belief Decomposer.

Main entry point for extracting and storing beliefs from Experience rows.
Wires together all services in the correct sequence:

1. Source context classification (mode, weight, context_id)
2. Text segmentation into claim candidates
3. LLM atomization of candidates
4. Validation and filtering
5. Canonicalization and deduplication
6. Epistemics extraction (rules + LLM fallback)
7. Embedding computation
8. Resolution (match/no_match/uncertain)
9. Stream classification
10. Storage (BeliefNode, BeliefOccurrence, TentativeLink, ConflictEdge)
11. Activation and core score updates
12. Stream migration checks
"""

import logging
import uuid as uuid_module
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from src.utils.belief_config import BeliefSystemConfig, get_belief_config
from src.utils.extractor_version import get_extractor_version
from src.services.eval_event_logger import EvalEventBuilder

# Phase 2-3: Segmentation and canonicalization
from src.services.source_context_classifier import SourceContextClassifier, SourceContext
from src.services.belief_segmenter import BeliefSegmenter, ClaimCandidate
from src.services.belief_canonicalizer import BeliefCanonicalizer, CanonicalAtom

# Phase 4: Atomization
from src.services.belief_atomizer import BeliefAtomizer, RawAtom, AtomizerResult
from src.services.belief_atom_validator import BeliefAtomValidator, ValidationResult
from src.services.belief_atom_deduper import BeliefAtomDeduper, DedupResult

# Phase 5: Epistemics
from src.services.epistemics_rules import EpistemicsRulesEngine, EpistemicsResult, EpistemicFrame
from src.services.epistemics_llm import EpistemicsLLMFallback
from src.services.stream_classifier import StreamClassifier, StreamClassification

# Phase 6: Embedding and resolution
from src.services.htn_belief_embedder import HTNBeliefEmbedder
from src.services.belief_resolver import BeliefResolver, ResolutionResult, ConcurrencyError
from src.services.belief_match_verifier import BeliefMatchVerifier
from src.services.tentative_link_service import TentativeLinkService, TentativeLinkUpdate

# Phase 7: Conflict detection
from src.services.conflict_engine import ConflictEngine, ConflictEdge

# Phase 8: Scoring and migration
from src.services.activation_service import ActivationService
from src.services.core_score_service import CoreScoreService, CoreScoreResult
from src.services.stream_service import StreamService, MigrationResult

# Models
from src.memory.models.belief_node import BeliefNode
from src.memory.models.belief_occurrence import BeliefOccurrence
from src.memory.models.tentative_link import TentativeLink
from src.memory.models.stream_assignment import StreamAssignment

# SelfKnowledgeIndex integration (TASK 10.2)
from src.services.self_knowledge_index import SelfKnowledgeIndex

logger = logging.getLogger(__name__)


@dataclass
class AtomResult:
    """
    Result for a single processed atom.

    Attributes:
        atom: The canonical atom
        node: BeliefNode (matched or created)
        occurrence: Created BeliefOccurrence
        resolution: Resolution result
        epistemics: Epistemics extraction result
        stream: Stream classification
        is_new_node: Whether a new node was created
        tentative_link: Created tentative link (if uncertain)
        conflicts: Created conflict edges
    """
    atom: CanonicalAtom
    node: BeliefNode
    occurrence: BeliefOccurrence
    resolution: ResolutionResult
    epistemics: EpistemicsResult
    stream: StreamClassification
    is_new_node: bool
    tentative_link: Optional[TentativeLink] = None
    conflicts: List[ConflictEdge] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """
    Complete result of belief extraction from an experience.

    Attributes:
        experience_id: Source experience ID
        extractor_version: Version hash used
        source_context: Source classification result
        candidates: Claim candidates from segmentation
        raw_atoms: Raw atoms from LLM
        valid_atoms: Atoms after validation
        dedup_result: Deduplication result
        atom_results: Results for each processed atom
        errors: Any errors encountered
        stats: Processing statistics
    """
    experience_id: str
    extractor_version: str
    source_context: SourceContext
    candidates: List[ClaimCandidate]
    raw_atoms: List[RawAtom]
    valid_atoms: List[RawAtom]
    dedup_result: DedupResult
    atom_results: List[AtomResult]
    errors: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class HTNBeliefExtractor:
    """
    Orchestration layer for HTN Self-Belief Decomposer.

    Main entry point: extract_and_update_self_knowledge(experience)

    This class wires together all the individual services in the correct
    sequence to extract beliefs from Experience(type='self_definition') rows.
    """

    def __init__(
        self,
        config: Optional[BeliefSystemConfig] = None,
        llm_client: Optional[Any] = None,
        db_session: Optional[Session] = None,
        eval_logger: Optional[EvalEventBuilder] = None,
    ):
        """
        Initialize the HTN belief extractor.

        Args:
            config: Configuration object (loads default if None)
            llm_client: LLM client for atomization/epistemics/verification
            db_session: Database session for persistence
            eval_logger: Optional eval event logger for debugging
        """
        if config is None:
            config = get_belief_config()

        self.config = config
        self.llm = llm_client
        self.db = db_session
        self.eval_logger = eval_logger

        # Get extractor version hash
        self.version_hash: str = get_extractor_version(config)

        # Initialize all services
        self._init_services()

    def _init_services(self) -> None:
        """Initialize all required services."""
        # Phase 2-3: Source and segmentation
        self.source_classifier = SourceContextClassifier(self.config)
        self.segmenter = BeliefSegmenter()
        self.canonicalizer = BeliefCanonicalizer()

        # Phase 4: Atomization
        self.atomizer = BeliefAtomizer(self.config, self.llm)
        self.validator = BeliefAtomValidator()
        self.deduper = BeliefAtomDeduper(self.canonicalizer)

        # Phase 5: Epistemics
        self.epistemics_rules = EpistemicsRulesEngine(self.config)
        self.epistemics_llm = EpistemicsLLMFallback(self.config, self.llm) if self.llm else None
        self.stream_classifier = StreamClassifier(self.config)

        # Phase 6: Embedding and resolution
        self.embedder = HTNBeliefEmbedder(self.config)
        self.verifier = BeliefMatchVerifier(self.config, self.llm) if self.llm else None
        self.resolver = BeliefResolver(
            self.config,
            self.embedder,
            self.verifier,
            self.db
        )
        self.tentative_link_service = TentativeLinkService(self.config, self.db)

        # Phase 7: Conflict detection
        self.conflict_engine = ConflictEngine(self.config, self.embedder, self.db)

        # Phase 8: Scoring and migration
        self.activation_service = ActivationService(self.config, self.db)
        self.core_score_service = CoreScoreService(self.config, self.db)
        self.stream_service = StreamService(self.config, self.db)

        # SelfKnowledgeIndex integration (TASK 10.2)
        # Note: SelfKnowledgeIndex requires raw_store which may not be available
        # in all contexts (e.g., testing). Skip if not available.
        self.self_knowledge_index = None

    def extract_and_update_self_knowledge(
        self,
        experience: Any
    ) -> ExtractionResult:
        """
        Main entry point: extract beliefs from an experience.

        This is the complete pipeline that:
        1. Extracts atoms from the experience text
        2. Resolves atoms to existing or new BeliefNodes
        3. Stores occurrences and handles uncertain matches
        4. Updates derived state (activation, core score, stream)

        Args:
            experience: Experience object with content, affect, etc.

        Returns:
            ExtractionResult with all processing details
        """
        experience_id = str(getattr(experience, 'id', uuid_module.uuid4()))

        logger.info(f"Starting belief extraction for experience {experience_id}")

        errors: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {
            'start_time': datetime.now(timezone.utc).isoformat(),
        }

        # Step 1: Extract atoms
        atoms_result = self._extract_atoms(experience)
        candidates = atoms_result['candidates']
        raw_atoms = atoms_result['raw_atoms']
        valid_atoms = atoms_result['valid_atoms']
        dedup_result = atoms_result['dedup_result']
        source_context = atoms_result['source_context']
        errors.extend(atoms_result.get('errors', []))

        stats['candidates_count'] = len(candidates)
        stats['raw_atoms_count'] = len(raw_atoms)
        stats['valid_atoms_count'] = len(valid_atoms)
        stats['deduped_atoms_count'] = len(dedup_result.deduped_atoms)

        # Step 2: Resolve and store each atom
        atom_results: List[AtomResult] = []

        for atom in dedup_result.deduped_atoms:
            try:
                result = self._resolve_and_store_atom(
                    atom,
                    experience,
                    source_context
                )
                atom_results.append(result)
            except ConcurrencyError as e:
                logger.error(f"Concurrency error for atom '{atom.canonical_text}': {e}")
                errors.append({
                    'atom': atom.canonical_text,
                    'error_type': 'ConcurrencyError',
                    'details': str(e),
                })
            except Exception as e:
                logger.error(f"Error processing atom '{atom.canonical_text}': {e}")
                errors.append({
                    'atom': atom.canonical_text,
                    'error_type': type(e).__name__,
                    'details': str(e),
                })

        # Step 3: Update derived state for all affected nodes
        affected_nodes = set()
        for result in atom_results:
            affected_nodes.add(result.node.belief_id)

        for node_id in affected_nodes:
            try:
                self._update_derived_state(node_id)
            except Exception as e:
                logger.error(f"Error updating derived state for node {node_id}: {e}")
                errors.append({
                    'node_id': str(node_id),
                    'error_type': type(e).__name__,
                    'details': str(e),
                })

        stats['end_time'] = datetime.now(timezone.utc).isoformat()
        stats['nodes_created'] = sum(1 for r in atom_results if r.is_new_node)
        stats['nodes_matched'] = sum(1 for r in atom_results if not r.is_new_node)
        stats['tentative_links_created'] = sum(1 for r in atom_results if r.tentative_link)
        stats['conflicts_detected'] = sum(len(r.conflicts) for r in atom_results)

        # Step 4: Update SelfKnowledgeIndex (TASK 10.2)
        # Add claim for each atom by its belief type/topic
        if self.self_knowledge_index:
            for result in atom_results:
                try:
                    # Map belief_type to category and use canonical_text as topic
                    category = result.atom.belief_type.lower()
                    topic = result.atom.canonical_text[:50]  # Truncate for topic
                    self.self_knowledge_index.add_claim(category, topic, experience_id)
                except Exception as e:
                    logger.warning(f"Failed to add claim to SelfKnowledgeIndex: {e}")

        logger.info(
            f"Belief extraction complete for {experience_id}: "
            f"{stats['deduped_atoms_count']} atoms, "
            f"{stats['nodes_created']} new nodes, "
            f"{stats['nodes_matched']} matched"
        )

        return ExtractionResult(
            experience_id=experience_id,
            extractor_version=self.version_hash,
            source_context=source_context,
            candidates=candidates,
            raw_atoms=raw_atoms,
            valid_atoms=valid_atoms,
            dedup_result=dedup_result,
            atom_results=atom_results,
            errors=errors,
            stats=stats,
        )

    def _extract_atoms(self, experience: Any) -> Dict[str, Any]:
        """
        Extract atoms from an experience (Steps 1-9).

        1. Classify source context
        2. Get text content
        3. Segment into candidates
        4. Atomize with LLM
        5. Validate atoms
        6. Canonicalize and deduplicate

        Returns dict with candidates, raw_atoms, valid_atoms, dedup_result, source_context
        """
        errors: List[Dict] = []

        # Step 1: Classify source context
        source_context = self.source_classifier.classify(experience)

        # Step 2: Get text content
        content = getattr(experience, 'content', {})
        if isinstance(content, dict):
            text = content.get('text', '')
        else:
            text = str(content)

        if not text:
            return {
                'candidates': [],
                'raw_atoms': [],
                'valid_atoms': [],
                'dedup_result': DedupResult(deduped_atoms=[], duplicates_removed=0),
                'source_context': source_context,
                'errors': [],
            }

        # Step 3: Segment into candidates
        candidates = self.segmenter.segment(text)

        if not candidates:
            return {
                'candidates': [],
                'raw_atoms': [],
                'valid_atoms': [],
                'dedup_result': DedupResult(deduped_atoms=[], duplicates_removed=0),
                'source_context': source_context,
                'errors': [],
            }

        # Step 4: Atomize with LLM
        atomizer_result: AtomizerResult = self.atomizer.atomize(candidates)
        raw_atoms = atomizer_result.atoms
        errors.extend(atomizer_result.errors)

        # Step 5: Validate atoms
        validation_result: ValidationResult = self.validator.validate(raw_atoms)
        valid_atoms = validation_result.valid

        # Log validation failures for debugging
        if validation_result.invalid:
            for invalid in validation_result.invalid:
                logger.debug(f"Rejected atom: {invalid}")

        # Step 6: Canonicalize and deduplicate
        dedup_result: DedupResult = self.deduper.dedup(valid_atoms)

        return {
            'candidates': candidates,
            'raw_atoms': raw_atoms,
            'valid_atoms': valid_atoms,
            'dedup_result': dedup_result,
            'source_context': source_context,
            'errors': errors,
        }

    def _resolve_and_store_atom(
        self,
        atom: CanonicalAtom,
        experience: Any,
        source_context: SourceContext
    ) -> AtomResult:
        """
        Resolve an atom and store it (Steps 10-13).

        10. Extract epistemics
        11. Compute embedding
        12. Resolve to existing or new node
        13. Store occurrence, handle uncertain matches, detect conflicts

        Returns AtomResult with all details.
        """
        experience_id = str(getattr(experience, 'id', 'unknown'))

        # Step 10: Extract epistemics (rules first, LLM fallback if needed)
        epistemics_result = self.epistemics_rules.extract(atom.original_text)

        if epistemics_result.needs_llm_fallback and self.epistemics_llm:
            llm_result = self.epistemics_llm.extract(atom.original_text)
            if llm_result and llm_result.confidence > epistemics_result.confidence:
                epistemics_result = llm_result

        # Apply detected polarity from epistemics
        polarity = atom.polarity
        if epistemics_result.detected_polarity:
            polarity = epistemics_result.detected_polarity

        # Step 11: Compute embedding
        embedding = None
        if self.embedder.enabled:
            embedding = self.embedder.embed(atom.canonical_text)

        # Step 12: Resolve to existing or new node
        # Create a temporary atom with updated polarity for resolution
        resolve_atom = CanonicalAtom(
            original_text=atom.original_text,
            canonical_text=atom.canonical_text,
            canonical_hash=atom.canonical_hash,
            belief_type=atom.belief_type,
            polarity=polarity,
            spans=atom.spans,
            confidence=atom.confidence,
        )

        resolution = self.resolver.resolve(resolve_atom, embedding)

        # Get or create node
        is_new_node = resolution.outcome != 'match'
        node = self.resolver.create_or_get_node(resolve_atom, embedding, resolution)

        # Step 13a: Classify stream
        stream_classification = self.stream_classifier.classify(
            atom.belief_type,
            epistemics_result.frame
        )

        # Step 13b: Create or update stream assignment
        stream_assignment = None
        if is_new_node:
            stream_assignment = self.stream_service.assign_initial(
                node,
                stream_classification
            )
        else:
            stream_assignment = self.stream_service.get_assignment(node.belief_id)

        # Step 13c: Create occurrence
        occurrence = self._create_occurrence(
            node=node,
            atom=atom,
            experience_id=experience_id,
            source_context=source_context,
            epistemics_result=epistemics_result,
            resolution=resolution,
        )

        # Step 13d: Handle uncertain resolution
        tentative_link = None
        if resolution.outcome == 'uncertain' and resolution.candidate_ids:
            # Create tentative link to best candidate
            best_candidate_id = resolution.candidate_ids[0]
            best_candidate = self.db.get(BeliefNode, best_candidate_id) if self.db else None

            if best_candidate and best_candidate.belief_id != node.belief_id:
                tentative_link = self.tentative_link_service.create_or_update(
                    node_a=node,
                    node_b=best_candidate,
                    initial_confidence=resolution.match_confidence,
                    signals={
                        'resolution_outcome': resolution.outcome,
                        'similarity': resolution.match_confidence,
                        'verifier_used': resolution.verifier_used,
                        'verifier_result': resolution.verifier_result,
                    },
                    extractor_version=self.version_hash,
                )

        # Step 13e: Detect conflicts
        conflicts: List[ConflictEdge] = []
        if is_new_node or resolution.outcome == 'match':
            conflicts = self.conflict_engine.detect_conflicts(
                node,
                embedding,
                occurrence
            )

        return AtomResult(
            atom=atom,
            node=node,
            occurrence=occurrence,
            resolution=resolution,
            epistemics=epistemics_result,
            stream=stream_classification,
            is_new_node=is_new_node,
            tentative_link=tentative_link,
            conflicts=conflicts,
        )

    def _create_occurrence(
        self,
        node: BeliefNode,
        atom: CanonicalAtom,
        experience_id: str,
        source_context: SourceContext,
        epistemics_result: EpistemicsResult,
        resolution: ResolutionResult,
    ) -> BeliefOccurrence:
        """
        Create a BeliefOccurrence record.

        Handles unique constraint by updating existing if duplicate.
        """
        # Build epistemic frame dict
        frame = epistemics_result.frame
        epistemic_frame = {
            'temporal_scope': frame.temporal_scope,
            'modality': frame.modality,
            'degree': frame.degree,
            'conditional': frame.conditional,
        }

        # Build raw_span
        raw_span = None
        if atom.spans:
            if len(atom.spans) == 1:
                raw_span = {'start': atom.spans[0][0], 'end': atom.spans[0][1]}
            else:
                raw_span = [{'start': s[0], 'end': s[1]} for s in atom.spans]

        occurrence = BeliefOccurrence(
            occurrence_id=uuid_module.uuid4(),
            belief_id=node.belief_id,
            source_experience_id=experience_id,
            extractor_version=self.version_hash,
            raw_text=atom.original_text,
            raw_span=raw_span,
            source_weight=source_context.source_weight,
            atom_confidence=atom.confidence,
            epistemic_frame=epistemic_frame,
            epistemic_confidence=epistemics_result.confidence,
            match_confidence=resolution.match_confidence,
            context_id=source_context.context_id,
        )

        if self.db:
            # Check for existing (unique constraint handling)
            existing = self.db.exec(
                select(BeliefOccurrence).where(
                    BeliefOccurrence.belief_id == node.belief_id,
                    BeliefOccurrence.source_experience_id == experience_id,
                    BeliefOccurrence.extractor_version == self.version_hash,
                )
            ).first()

            if existing:
                # Update existing occurrence
                existing.raw_text = occurrence.raw_text
                existing.raw_span = occurrence.raw_span
                existing.source_weight = occurrence.source_weight
                existing.atom_confidence = occurrence.atom_confidence
                existing.epistemic_frame = occurrence.epistemic_frame
                existing.epistemic_confidence = occurrence.epistemic_confidence
                existing.match_confidence = occurrence.match_confidence
                existing.context_id = occurrence.context_id
                self.db.add(existing)
                self.db.commit()
                self.db.refresh(existing)
                return existing

            self.db.add(occurrence)
            self.db.commit()
            self.db.refresh(occurrence)

        return occurrence

    def _update_derived_state(self, node_id: uuid_module.UUID) -> None:
        """
        Update derived state for a node (Steps 14-17).

        14. Update activation
        15. Update core score
        16. Check stream migration
        """
        if not self.db:
            return

        node = self.db.get(BeliefNode, node_id)
        if not node:
            return

        stream_assignment = self.stream_service.get_assignment(node_id)

        # Step 14: Update activation
        self.activation_service.update_activation(node, stream_assignment)

        # Step 15: Update core score
        core_result: CoreScoreResult = self.core_score_service.update_core_score(
            node,
            self.conflict_engine
        )

        # Step 16: Check stream migration
        if stream_assignment:
            migration_result: MigrationResult = self.stream_service.check_migration(
                node,
                stream_assignment,
                core_result
            )

            if migration_result.migrated:
                logger.info(
                    f"Node {node_id} migrated from {migration_result.from_stream} "
                    f"to {migration_result.to_stream}"
                )


def extract_beliefs_from_experience(
    experience: Any,
    llm_client: Optional[Any] = None,
    db_session: Optional[Session] = None
) -> ExtractionResult:
    """
    Convenience function to extract beliefs from an experience.

    Args:
        experience: Experience object
        llm_client: Optional LLM client
        db_session: Optional database session

    Returns:
        ExtractionResult
    """
    extractor = HTNBeliefExtractor(
        llm_client=llm_client,
        db_session=db_session,
    )
    return extractor.extract_and_update_self_knowledge(experience)
