"""Unit tests for TentativeLinkService."""

import pytest
import math
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from src.services.tentative_link_service import TentativeLinkService
from src.memory.models.tentative_link import TentativeLink


class TestConfidenceFormula:
    """Test confidence = sigmoid(a*support_both - b*support_one - c*age).

    Note: We test via update_confidence() since confidence computation is internal.
    """

    def _make_link(
        self,
        support_both: int = 10,
        support_one: int = 0,
        age_days: float = 0
    ) -> TentativeLink:
        """Create a link with specified parameters."""
        created_at = datetime.now(timezone.utc) - timedelta(days=age_days)
        return TentativeLink(
            link_id=uuid4(),
            from_belief_id=uuid4(),
            to_belief_id=uuid4(),
            confidence=0.5,
            status="pending",
            support_both=support_both,
            support_one=support_one,
            last_support_at=created_at,
            extractor_version="v1",
            created_at=created_at,
        )

    def test_high_support_both_increases_confidence(self):
        # Service without db - will compute but not persist
        service = TentativeLinkService(db_session=None)
        link = self._make_link(support_both=20, support_one=0, age_days=0)

        # Default params: a=1.2, b=0.9, c=0.06
        # raw = 1.2 * 20 - 0.9 * 0 - 0.06 * 0 = 24
        # sigmoid(24) ≈ 1.0
        result = service.update_confidence(link)
        assert result.new_confidence > 0.9

    def test_high_support_one_decreases_confidence(self):
        service = TentativeLinkService(db_session=None)
        link = self._make_link(support_both=5, support_one=20, age_days=0)

        # raw = 1.2 * 5 - 0.9 * 20 - 0 = 6 - 18 = -12
        # sigmoid(-12) ≈ 0
        result = service.update_confidence(link)
        assert result.new_confidence < 0.3

    def test_old_links_decay(self):
        service = TentativeLinkService(db_session=None)
        link_fresh = self._make_link(support_both=10, support_one=0, age_days=0)
        link_old = self._make_link(support_both=10, support_one=0, age_days=60)

        result_fresh = service.update_confidence(link_fresh)
        result_old = service.update_confidence(link_old)

        assert result_old.new_confidence < result_fresh.new_confidence

    def test_balanced_support_near_half(self):
        service = TentativeLinkService(db_session=None)
        # When support_both ≈ support_one * (a/b) and no age,
        # confidence should be near 0.5
        link = self._make_link(support_both=10, support_one=13, age_days=0)
        # raw = 1.2 * 10 - 0.9 * 13 = 12 - 11.7 = 0.3
        # sigmoid(0.3) ≈ 0.57
        result = service.update_confidence(link)
        assert 0.4 < result.new_confidence < 0.7


class TestAutoAcceptReject:
    """Test auto-accept and auto-reject thresholds.

    Note: We test without DB to avoid SQLite stripping timezone info
    from datetimes. The service handles DB persistence separately.
    """

    def test_high_confidence_auto_accepts(self):
        # Service without DB - just tests confidence logic
        service = TentativeLinkService(db_session=None)
        link = TentativeLink(
            link_id=uuid4(),
            from_belief_id=uuid4(),
            to_belief_id=uuid4(),
            confidence=0.5,
            status="pending",
            support_both=100,  # Very high support
            support_one=0,
            last_support_at=datetime.now(timezone.utc),
            extractor_version="v1",
            created_at=datetime.now(timezone.utc),
        )

        result = service.update_confidence(link)

        assert result.new_confidence >= 0.85
        assert link.status == "accepted"
        assert result.merge_required is True

    def test_low_confidence_auto_rejects(self):
        service = TentativeLinkService(db_session=None)
        link = TentativeLink(
            link_id=uuid4(),
            from_belief_id=uuid4(),
            to_belief_id=uuid4(),
            confidence=0.5,
            status="pending",
            support_both=0,
            support_one=50,  # Heavy counter-evidence
            last_support_at=datetime.now(timezone.utc),
            extractor_version="v1",
            created_at=datetime.now(timezone.utc),
        )

        result = service.update_confidence(link)

        assert result.new_confidence <= 0.15
        assert link.status == "rejected"


class TestLinkCreation:
    """Test tentative link creation."""

    def test_creates_new_link(self, test_db):
        from src.memory.models.belief_node import BeliefNode

        service = TentativeLinkService(db_session=test_db)

        node_a = BeliefNode(
            belief_id=uuid4(),
            canonical_text="i am patient",
            canonical_hash="hash_a_link",
            belief_type="TRAIT",
            polarity="affirm",
        )
        node_b = BeliefNode(
            belief_id=uuid4(),
            canonical_text="i am usually patient",
            canonical_hash="hash_b_link",
            belief_type="TRAIT",
            polarity="affirm",
        )
        test_db.add_all([node_a, node_b])
        test_db.commit()

        link = service.create_or_update(
            node_a=node_a,
            node_b=node_b,
            initial_confidence=0.82,
            signals={"similarity": 0.82},
            extractor_version="v1",
        )

        assert link.status == "pending"
        assert link.support_both == 1
        assert link.support_one == 0

    def test_updates_existing_link(self, test_db):
        from src.memory.models.belief_node import BeliefNode

        service = TentativeLinkService(db_session=test_db)

        node_a = BeliefNode(
            belief_id=uuid4(),
            canonical_text="i am patient",
            canonical_hash="hash_a_update",
            belief_type="TRAIT",
            polarity="affirm",
        )
        node_b = BeliefNode(
            belief_id=uuid4(),
            canonical_text="i am usually patient",
            canonical_hash="hash_b_update",
            belief_type="TRAIT",
            polarity="affirm",
        )
        test_db.add_all([node_a, node_b])
        test_db.commit()

        # Create first link
        link1 = service.create_or_update(
            node_a=node_a,
            node_b=node_b,
            initial_confidence=0.82,
            signals={"similarity": 0.82},
            extractor_version="v1",
        )

        # Create second link (should update existing)
        link2 = service.create_or_update(
            node_a=node_a,
            node_b=node_b,
            initial_confidence=0.85,
            signals={"similarity": 0.85},
            extractor_version="v1",
        )

        assert link1.link_id == link2.link_id
        assert link2.support_both == 2  # Incremented
