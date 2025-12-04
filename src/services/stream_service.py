"""
Stream Service for HTN Self-Belief Decomposer.

Manages stream assignments with migration ratchet:
- Assigns initial streams based on belief type + temporal scope
- Checks for STATE -> IDENTITY migration when thresholds met
- Implements ratchet to prevent demotion without explicit triggers
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import uuid as uuid_module

from sqlmodel import Session, select

from src.utils.belief_config import BeliefSystemConfig, get_belief_config
from src.memory.models.belief_node import BeliefNode
from src.memory.models.stream_assignment import StreamAssignment
from src.services.stream_classifier import StreamClassification, StreamClassifier
from src.services.core_score_service import CoreScoreResult

logger = logging.getLogger(__name__)


@dataclass
class MigrationResult:
    """
    Result of a migration check.

    Attributes:
        migrated: Whether migration occurred
        from_stream: Original stream (if migrated)
        to_stream: New stream (if migrated)
        assignment: The updated StreamAssignment
    """
    migrated: bool
    from_stream: Optional[str] = None
    to_stream: Optional[str] = None
    assignment: Optional[StreamAssignment] = None


class StreamService:
    """
    Manage stream assignments with migration ratchet.

    Streams:
    - identity: Core self-concept
    - state: Current emotional/mental state
    - meta: Beliefs about beliefs
    - relational: Relationships and social patterns

    Migration:
    - STATE -> IDENTITY when spread and diversity thresholds met
    - Ratchet prevents demotion without explicit triggers
    """

    def __init__(
        self,
        config: Optional[BeliefSystemConfig] = None,
        db_session: Optional[Session] = None
    ):
        """
        Initialize the stream service.

        Args:
            config: Configuration object
            db_session: Database session
        """
        if config is None:
            config = get_belief_config()

        self.config = config.migration
        self.db = db_session

        # Migration thresholds
        promote_config = config.migration.promote_state_to_identity
        self.migration_enabled = promote_config.enabled
        self.min_spread = promote_config.min_spread
        self.min_diversity = promote_config.min_diversity
        self.min_activation = promote_config.min_activation

        # Ratchet config
        ratchet_config = config.migration.ratchet
        self.ratchet_enabled = ratchet_config.enabled
        self.allow_demotion = ratchet_config.allow_demotion
        self.demotion_triggers = ratchet_config.demotion_triggers

        self.classifier = StreamClassifier(config)

    def assign_initial(
        self,
        node: BeliefNode,
        classification: StreamClassification
    ) -> StreamAssignment:
        """
        Create initial stream assignment from classifier output.

        Args:
            node: The belief node
            classification: Classification from StreamClassifier

        Returns:
            Created StreamAssignment
        """
        assignment = StreamAssignment(
            belief_id=node.belief_id,
            primary_stream=classification.primary_stream,
            secondary_stream=classification.secondary_stream,
            confidence=classification.confidence,
            migrated_from=None,
        )

        if self.db:
            self.db.add(assignment)
            self.db.commit()
            self.db.refresh(assignment)

        return assignment

    def get_assignment(
        self,
        node_id: uuid_module.UUID
    ) -> Optional[StreamAssignment]:
        """Get stream assignment for a node."""
        if not self.db:
            return None

        return self.db.exec(
            select(StreamAssignment).where(
                StreamAssignment.belief_id == node_id
            )
        ).first()

    def check_migration(
        self,
        node: BeliefNode,
        assignment: StreamAssignment,
        core_result: CoreScoreResult
    ) -> MigrationResult:
        """
        Check if STATE should migrate to IDENTITY.

        Migration conditions (all must be met):
        - current primary_stream == "state"
        - spread component >= min_spread
        - diversity component >= min_diversity
        - activation >= min_activation

        If migrating:
        - Update primary_stream to "identity"
        - Set migrated_from = "state"

        Args:
            node: The belief node
            assignment: Current stream assignment
            core_result: Result from core score computation

        Returns:
            MigrationResult indicating if migration occurred
        """
        if not self.migration_enabled:
            return MigrationResult(migrated=False, assignment=assignment)

        # Only migrate from state
        if assignment.primary_stream != 'state':
            return MigrationResult(migrated=False, assignment=assignment)

        # Check thresholds
        components = core_result.components
        spread = components.get('spread', 0.0)
        diversity = components.get('diversity', 0.0)

        if spread < self.min_spread:
            logger.debug(f"Migration blocked: spread {spread} < {self.min_spread}")
            return MigrationResult(migrated=False, assignment=assignment)

        if diversity < self.min_diversity:
            logger.debug(f"Migration blocked: diversity {diversity} < {self.min_diversity}")
            return MigrationResult(migrated=False, assignment=assignment)

        if node.activation < self.min_activation:
            logger.debug(f"Migration blocked: activation {node.activation} < {self.min_activation}")
            return MigrationResult(migrated=False, assignment=assignment)

        # All conditions met - migrate
        old_stream = assignment.primary_stream
        assignment.primary_stream = 'identity'
        assignment.migrated_from = old_stream
        assignment.updated_at = datetime.now(timezone.utc)

        if self.db:
            self.db.add(assignment)
            self.db.commit()

        logger.info(
            f"Migrated belief {node.belief_id} from {old_stream} to identity "
            f"(spread={spread:.2f}, diversity={diversity:.2f}, activation={node.activation:.2f})"
        )

        return MigrationResult(
            migrated=True,
            from_stream=old_stream,
            to_stream='identity',
            assignment=assignment,
        )

    def can_demote(
        self,
        node: BeliefNode,
        assignment: StreamAssignment,
        trigger: Optional[str] = None
    ) -> bool:
        """
        Check if demotion is allowed for a node.

        Only if:
        - ratchet.allow_demotion = true, OR
        - explicit demotion trigger present

        Args:
            node: The belief node
            assignment: Current stream assignment
            trigger: Optional explicit trigger

        Returns:
            True if demotion is allowed
        """
        if self.allow_demotion:
            return True

        if trigger and trigger in self.demotion_triggers:
            return True

        return False

    def demote_to_state(
        self,
        node: BeliefNode,
        assignment: StreamAssignment,
        trigger: str
    ) -> MigrationResult:
        """
        Demote a belief from identity to state.

        Requires explicit trigger.

        Args:
            node: The belief node
            assignment: Current stream assignment
            trigger: The demotion trigger

        Returns:
            MigrationResult
        """
        if not self.can_demote(node, assignment, trigger):
            logger.warning(
                f"Demotion blocked for {node.belief_id}: "
                f"trigger '{trigger}' not in allowed triggers"
            )
            return MigrationResult(migrated=False, assignment=assignment)

        if assignment.primary_stream != 'identity':
            return MigrationResult(migrated=False, assignment=assignment)

        old_stream = assignment.primary_stream
        assignment.primary_stream = 'state'
        assignment.migrated_from = f"{old_stream}:demoted:{trigger}"
        assignment.updated_at = datetime.now(timezone.utc)

        if self.db:
            self.db.add(assignment)
            self.db.commit()

        logger.info(
            f"Demoted belief {node.belief_id} from identity to state "
            f"(trigger={trigger})"
        )

        return MigrationResult(
            migrated=True,
            from_stream=old_stream,
            to_stream='state',
            assignment=assignment,
        )
