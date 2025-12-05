"""Integration tests for extractor version tracking.

Tests that extractor versions are properly tracked and can be used
for rollback and migration.
"""

import pytest
from uuid import uuid4
from datetime import datetime, timezone


class TestVersionTracking:
    """Test extractor version is recorded correctly."""

    def test_occurrence_records_version(self, test_db):
        """Each occurrence records the extractor version."""
        from src.memory.models.belief_node import BeliefNode
        from src.memory.models.belief_occurrence import BeliefOccurrence
        from src.utils.extractor_version import get_extractor_version

        node = BeliefNode(
            belief_id=uuid4(),
            canonical_text="i am patient",
            canonical_hash="hash_version_test",
            belief_type="TRAIT",
            polarity="affirm",
        )
        test_db.add(node)
        test_db.commit()

        version = get_extractor_version()
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

        loaded = test_db.query(BeliefOccurrence).filter(
            BeliefOccurrence.occurrence_id == occ.occurrence_id
        ).first()
        assert loaded.extractor_version == version

    def test_version_format_valid(self):
        """Extractor version follows expected format."""
        from src.utils.extractor_version import get_extractor_version

        version = get_extractor_version()

        # Should be a string
        assert isinstance(version, str)

        # Should not be empty
        assert len(version) > 0

        # Common formats: "v1.0.0", "1.0.0", "htn_decomposer_v1"
        # At minimum, should contain alphanumeric chars
        assert any(c.isalnum() for c in version)


class TestVersionMigration:
    """Test migrating data between extractor versions."""

    def test_can_identify_old_version_data(self, test_db):
        """Can query for data from specific extractor versions."""
        from src.memory.models.belief_node import BeliefNode
        from src.memory.models.belief_occurrence import BeliefOccurrence

        node = BeliefNode(
            belief_id=uuid4(),
            canonical_text="i am patient",
            canonical_hash="hash_migrate_test",
            belief_type="TRAIT",
            polarity="affirm",
        )
        test_db.add(node)
        test_db.commit()

        # Create occurrences with different versions
        versions = ["v0.9.0", "v1.0.0", "v1.1.0"]
        for v in versions:
            occ = BeliefOccurrence(
                occurrence_id=uuid4(),
                belief_id=node.belief_id,
                source_experience_id=str(uuid4()),
                extractor_version=v,
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

        # Find all v0.x data (old version)
        old_data = test_db.query(BeliefOccurrence).filter(
            BeliefOccurrence.extractor_version.like("v0.%")
        ).all()
        assert len(old_data) == 1

    def test_can_reprocess_old_version(self, test_db):
        """Can mark old version data for reprocessing."""
        from src.memory.models.belief_node import BeliefNode
        from src.memory.models.belief_occurrence import BeliefOccurrence

        node = BeliefNode(
            belief_id=uuid4(),
            canonical_text="i am patient",
            canonical_hash="hash_reprocess_test",
            belief_type="TRAIT",
            polarity="affirm",
        )
        test_db.add(node)
        test_db.commit()

        old_version = "v0.5.0"
        occ = BeliefOccurrence(
            occurrence_id=uuid4(),
            belief_id=node.belief_id,
            source_experience_id=str(uuid4()),
            extractor_version=old_version,
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

        # In real code, we would:
        # 1. Query source_experience_ids from old version
        # 2. Delete old occurrences
        # 3. Reprocess experiences with new extractor

        source_ids = test_db.query(BeliefOccurrence.source_experience_id).filter(
            BeliefOccurrence.extractor_version == old_version
        ).all()

        assert len(source_ids) == 1


class TestVersionCompatibility:
    """Test that different versions can coexist."""

    def test_multiple_versions_same_belief(self, test_db):
        """Same belief can have occurrences from different versions."""
        from src.memory.models.belief_node import BeliefNode
        from src.memory.models.belief_occurrence import BeliefOccurrence

        node = BeliefNode(
            belief_id=uuid4(),
            canonical_text="i am patient",
            canonical_hash="hash_compat_test",
            belief_type="TRAIT",
            polarity="affirm",
        )
        test_db.add(node)
        test_db.commit()

        # Add occurrences from different versions
        versions = ["v1.0.0", "v1.1.0", "v2.0.0"]
        for v in versions:
            occ = BeliefOccurrence(
                occurrence_id=uuid4(),
                belief_id=node.belief_id,
                source_experience_id=str(uuid4()),
                extractor_version=v,
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

        # All should link to same belief
        occurrences = test_db.query(BeliefOccurrence).filter(
            BeliefOccurrence.belief_id == node.belief_id
        ).all()
        assert len(occurrences) == 3

        # Belief should have latest version info accessible
        versions_found = {o.extractor_version for o in occurrences}
        assert versions_found == {"v1.0.0", "v1.1.0", "v2.0.0"}

    def test_version_comparison(self):
        """Can compare versions for ordering."""
        from packaging.version import Version

        versions = ["v1.0.0", "v1.1.0", "v2.0.0", "v0.9.0"]

        # Strip 'v' prefix and parse
        parsed = [Version(v.lstrip("v")) for v in versions]
        sorted_versions = sorted(parsed)

        assert str(sorted_versions[0]) == "0.9.0"
        assert str(sorted_versions[-1]) == "2.0.0"
