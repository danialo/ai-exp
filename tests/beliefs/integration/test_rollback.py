"""Integration tests for rollback functionality.

Tests that database transactions can be properly rolled back
and that partial failures don't corrupt state.
"""

import pytest
from uuid import uuid4
from datetime import datetime, timezone


class TestTransactionRollback:
    """Test that failed transactions don't leave partial state."""

    def test_failed_occurrence_creation_rolls_back(self, test_db):
        """If occurrence creation fails, no partial data remains."""
        from src.memory.models.belief_node import BeliefNode
        from src.memory.models.belief_occurrence import BeliefOccurrence

        # Create a node first
        node = BeliefNode(
            belief_id=uuid4(),
            canonical_text="i am patient",
            canonical_hash="hash_test",
            belief_type="TRAIT",
            polarity="affirm",
        )
        test_db.add(node)
        test_db.commit()

        initial_count = test_db.query(BeliefOccurrence).count()

        try:
            # Try to create occurrence with invalid data (missing required field)
            occ = BeliefOccurrence(
                occurrence_id=uuid4(),
                belief_id=node.belief_id,
                # Missing required fields to trigger error
                source_experience_id=None,  # Should cause constraint violation
                extractor_version="v1",
                raw_text="I am patient",
                source_weight=0.8,
                atom_confidence=0.9,
                epistemic_frame={},
                epistemic_confidence=0.85,
                match_confidence=0.9,
                context_id="ctx_1",
            )
            test_db.add(occ)
            test_db.commit()
        except Exception:
            test_db.rollback()

        # Count should be unchanged
        final_count = test_db.query(BeliefOccurrence).count()
        assert final_count == initial_count

    def test_batch_insert_atomic(self, test_db):
        """Batch inserts should be atomic - all or nothing."""
        from src.memory.models.belief_node import BeliefNode

        initial_count = test_db.query(BeliefNode).count()

        try:
            nodes = []
            for i in range(5):
                nodes.append(BeliefNode(
                    belief_id=uuid4(),
                    canonical_text=f"i am trait {i}",
                    canonical_hash=f"hash_{i}",
                    belief_type="TRAIT",
                    polarity="affirm",
                ))

            # Make last one invalid (duplicate hash of first)
            nodes[-1].canonical_hash = nodes[0].canonical_hash

            test_db.add_all(nodes)
            test_db.commit()
        except Exception:
            test_db.rollback()

        # Either all 5 were added or none
        final_count = test_db.query(BeliefNode).count()
        assert final_count == initial_count or final_count == initial_count + 5


class TestExtractorVersionRollback:
    """Test rollback by extractor version."""

    def test_can_query_by_extractor_version(self, test_db):
        """Can identify all occurrences by extractor version."""
        from src.memory.models.belief_node import BeliefNode
        from src.memory.models.belief_occurrence import BeliefOccurrence

        node = BeliefNode(
            belief_id=uuid4(),
            canonical_text="i am patient",
            canonical_hash="hash_rollback_test",
            belief_type="TRAIT",
            polarity="affirm",
        )
        test_db.add(node)
        test_db.commit()

        # Add occurrences with different versions
        for version in ["v1.0.0", "v1.0.0", "v2.0.0", "v2.0.0", "v2.0.0"]:
            occ = BeliefOccurrence(
                occurrence_id=uuid4(),
                belief_id=node.belief_id,
                source_experience_id=str(uuid4()),
                extractor_version=version,
                raw_text="I am patient",
                source_weight=0.8,
                atom_confidence=0.9,
                epistemic_frame={},
                epistemic_confidence=0.85,
                match_confidence=0.9,
                context_id="ctx_1",
            )
            test_db.add(occ)
        test_db.commit()

        # Query by version
        v1_count = test_db.query(BeliefOccurrence).filter(
            BeliefOccurrence.extractor_version == "v1.0.0"
        ).count()
        v2_count = test_db.query(BeliefOccurrence).filter(
            BeliefOccurrence.extractor_version == "v2.0.0"
        ).count()

        assert v1_count == 2
        assert v2_count == 3

    def test_delete_by_extractor_version(self, test_db):
        """Can delete all occurrences for a specific extractor version."""
        from src.memory.models.belief_node import BeliefNode
        from src.memory.models.belief_occurrence import BeliefOccurrence

        node = BeliefNode(
            belief_id=uuid4(),
            canonical_text="i am kind",
            canonical_hash="hash_delete_test",
            belief_type="TRAIT",
            polarity="affirm",
        )
        test_db.add(node)
        test_db.commit()

        # Add occurrences with different versions
        for version in ["v1.0.0", "v1.0.0", "v2.0.0"]:
            occ = BeliefOccurrence(
                occurrence_id=uuid4(),
                belief_id=node.belief_id,
                source_experience_id=str(uuid4()),
                extractor_version=version,
                raw_text="I am kind",
                source_weight=0.8,
                atom_confidence=0.9,
                epistemic_frame={},
                epistemic_confidence=0.85,
                match_confidence=0.9,
                context_id="ctx_1",
            )
            test_db.add(occ)
        test_db.commit()

        # Delete v1.0.0 occurrences
        test_db.query(BeliefOccurrence).filter(
            BeliefOccurrence.extractor_version == "v1.0.0"
        ).delete()
        test_db.commit()

        # Only v2.0.0 should remain
        remaining = test_db.query(BeliefOccurrence).filter(
            BeliefOccurrence.belief_id == node.belief_id
        ).all()
        assert len(remaining) == 1
        assert remaining[0].extractor_version == "v2.0.0"


class TestConflictEdgeRollback:
    """Test conflict edge cleanup during rollback."""

    def test_orphaned_edges_can_be_cleaned(self, test_db):
        """Conflict edges for deleted nodes can be identified and cleaned."""
        from src.memory.models.belief_node import BeliefNode
        from src.memory.models.conflict_edge import ConflictEdge

        # Create two nodes
        node_a = BeliefNode(
            belief_id=uuid4(),
            canonical_text="i am patient",
            canonical_hash="hash_a_edge",
            belief_type="TRAIT",
            polarity="affirm",
        )
        node_b = BeliefNode(
            belief_id=uuid4(),
            canonical_text="i am patient",
            canonical_hash="hash_b_edge",
            belief_type="TRAIT",
            polarity="deny",
        )
        test_db.add_all([node_a, node_b])
        test_db.commit()

        # Create conflict edge
        edge = ConflictEdge(
            edge_id=uuid4(),
            from_belief_id=node_a.belief_id,
            to_belief_id=node_b.belief_id,
            conflict_type="contradiction",
            evidence_count=1,
            status="tolerated",
        )
        test_db.add(edge)
        test_db.commit()

        # Delete one node
        test_db.delete(node_b)
        test_db.commit()

        # Edge now references non-existent node
        # In real code, this would be cleaned up by cascade or manual query
        orphaned_edges = test_db.query(ConflictEdge).filter(
            ConflictEdge.to_belief_id == node_b.belief_id
        ).all()

        # Clean up orphaned edges
        for e in orphaned_edges:
            test_db.delete(e)
        test_db.commit()

        remaining_edges = test_db.query(ConflictEdge).filter(
            ConflictEdge.from_belief_id == node_a.belief_id
        ).count()
        assert remaining_edges == 0
