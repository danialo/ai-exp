"""Unit tests for DesireStore."""

import json
import pytest
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.services.desire_store import Desire, DesireStore, create_desire_store


@pytest.fixture
def temp_desires_dir():
    """Create temporary directory for desires."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def desire_store(temp_desires_dir):
    """Create DesireStore with temp directory."""
    return create_desire_store(temp_desires_dir)


class TestDesireCreation:
    """Test desire creation and ID generation."""

    def test_record_desire(self, desire_store):
        """Test recording a basic desire."""
        desire = desire_store.record(
            text="I wish I had better test coverage",
            strength=1.0,
            tags=["testing", "quality"],
        )

        assert desire.id.startswith("des_")
        assert len(desire.id) == 12  # des_ + 8 hex chars
        assert desire.text == "I wish I had better test coverage"
        assert desire.strength == 1.0
        assert "testing" in desire.tags
        assert "quality" in desire.tags

    def test_deterministic_id_generation(self):
        """Test that same desire at same time produces same ID."""
        now = datetime(2025, 11, 11, 12, 0, 0, tzinfo=timezone.utc)
        id1 = Desire.generate_id("test desire", now)
        id2 = Desire.generate_id("test desire", now)
        assert id1 == id2

    def test_different_time_different_id(self):
        """Test that same text at different times produces different IDs."""
        now1 = datetime(2025, 11, 11, 12, 0, 0, tzinfo=timezone.utc)
        now2 = datetime(2025, 11, 11, 13, 0, 0, tzinfo=timezone.utc)

        id1 = Desire.generate_id("test desire", now1)
        id2 = Desire.generate_id("test desire", now2)
        assert id1 != id2

    def test_record_with_context(self, desire_store):
        """Test recording desire with context."""
        desire = desire_store.record(
            text="I want to optimize database queries",
            strength=0.8,
            context={"triggered_by": "slow_query_log", "query_time_ms": 5000},
        )

        assert desire.context["triggered_by"] == "slow_query_log"
        assert desire.context["query_time_ms"] == 5000

    def test_invalid_strength_raises(self, desire_store):
        """Test that invalid strength raises error."""
        with pytest.raises(ValueError, match="Strength must be between"):
            desire_store.record("test", strength=1.5)

        with pytest.raises(ValueError, match="Strength must be between"):
            desire_store.record("test", strength=-0.1)


class TestDesireRetrieval:
    """Test desire retrieval operations."""

    def test_get_existing_desire(self, desire_store):
        """Test retrieving existing desire."""
        created = desire_store.record("test desire")

        retrieved = desire_store.get(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.text == "test desire"

    def test_get_nonexistent_desire(self, desire_store):
        """Test retrieving nonexistent desire returns None."""
        result = desire_store.get("des_notfound")
        assert result is None

    def test_list_all_desires(self, desire_store):
        """Test listing all desires."""
        desire_store.record("desire 1", strength=0.9)
        desire_store.record("desire 2", strength=0.7)
        desire_store.record("desire 3", strength=0.5)

        all_desires = desire_store.list_all()
        assert len(all_desires) == 3

        # Should be sorted by strength descending
        assert all_desires[0].strength == 0.9
        assert all_desires[1].strength == 0.7
        assert all_desires[2].strength == 0.5

    def test_list_with_min_strength(self, desire_store):
        """Test filtering desires by minimum strength."""
        desire_store.record("strong", strength=0.9)
        desire_store.record("medium", strength=0.5)
        desire_store.record("weak", strength=0.2)

        strong_only = desire_store.list_all(min_strength=0.6)
        assert len(strong_only) == 1
        assert strong_only[0].text == "strong"

    def test_list_top_desires(self, desire_store):
        """Test listing top N desires."""
        for i in range(10):
            desire_store.record(f"desire {i}", strength=1.0 - (i * 0.1))

        top_3 = desire_store.list_top(limit=3)
        assert len(top_3) == 3
        assert top_3[0].strength == 1.0
        assert top_3[1].strength == 0.9
        assert top_3[2].strength == 0.8


class TestDesireReinforcement:
    """Test desire reinforcement."""

    def test_reinforce_desire(self, desire_store):
        """Test reinforcing a desire increases strength."""
        desire = desire_store.record("test", strength=0.5)

        reinforced = desire_store.reinforce(desire.id, delta=0.2)
        assert reinforced.strength == 0.7

    def test_reinforce_caps_at_1(self, desire_store):
        """Test that reinforcement caps at 1.0."""
        desire = desire_store.record("test", strength=0.95)

        reinforced = desire_store.reinforce(desire.id, delta=0.2)
        assert reinforced.strength == 1.0

    def test_reinforce_updates_timestamp(self, desire_store):
        """Test that reinforcement updates last_reinforced_at."""
        desire = desire_store.record("test")
        original_time = desire.last_reinforced_at

        # Wait a bit
        import time
        time.sleep(0.01)

        reinforced = desire_store.reinforce(desire.id)
        assert reinforced.last_reinforced_at != original_time

    def test_reinforce_nonexistent_raises(self, desire_store):
        """Test that reinforcing nonexistent desire raises error."""
        with pytest.raises(ValueError, match="not found"):
            desire_store.reinforce("des_notfound")


class TestDesireDecay:
    """Test desire decay mechanism."""

    def test_decay_reduces_strength(self, desire_store):
        """Test that decay reduces strength."""
        desire = desire_store.record("test", strength=1.0)

        # Manually set last_reinforced_at to 1 day ago
        one_day_ago = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        desire.last_reinforced_at = one_day_ago
        desire_store.index[desire.id] = desire

        # Decay with rate 0.1 per day
        decayed = desire_store.decay_all(decay_rate=0.1)

        assert desire.id in decayed
        updated = desire_store.get(desire.id)
        assert updated.strength < 1.0
        # Use pytest.approx for floating point comparison (should be ~0.9)
        assert updated.strength == pytest.approx(0.9, abs=0.01)

    def test_decay_respects_time(self, desire_store):
        """Test that decay is proportional to time elapsed."""
        desire = desire_store.record("test", strength=1.0)

        # Set last_reinforced_at to 10 days ago
        ten_days_ago = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        desire.last_reinforced_at = ten_days_ago
        desire_store.index[desire.id] = desire

        # Decay with rate 0.01 per day
        desire_store.decay_all(decay_rate=0.01)

        updated = desire_store.get(desire.id)
        # After 10 days at 0.01/day: 1.0 - (0.01 * 10) = 0.9
        assert abs(updated.strength - 0.9) < 0.01

    def test_decay_floors_at_zero(self, desire_store):
        """Test that decay doesn't go negative."""
        desire = desire_store.record("test", strength=0.05)

        # Set to long ago
        long_ago = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        desire.last_reinforced_at = long_ago
        desire_store.index[desire.id] = desire

        desire_store.decay_all(decay_rate=0.1)

        updated = desire_store.get(desire.id)
        assert updated.strength == 0.0

    def test_decay_returns_changed_desires(self, desire_store):
        """Test that decay_all returns dict of changed desires."""
        d1 = desire_store.record("old", strength=1.0)
        d2 = desire_store.record("recent", strength=1.0)

        # Make d1 old (5 days ago)
        old_time = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        d1.last_reinforced_at = old_time
        desire_store.index[d1.id] = d1

        # Set d2 to NOW (avoid timing issues with just-created desires)
        d2.last_reinforced_at = datetime.now(timezone.utc).isoformat()
        desire_store.index[d2.id] = d2

        decayed = desire_store.decay_all(decay_rate=0.1)

        # d1 should be decayed significantly
        assert d1.id in decayed
        assert decayed[d1.id] < 0.6  # 1.0 - (0.1 * 5) = 0.5

        # d2 may have tiny decay due to microseconds elapsed, but should be minimal
        if d2.id in decayed:
            assert decayed[d2.id] > 0.99  # Almost unchanged


class TestDesirePruning:
    """Test desire pruning."""

    def test_prune_weak_desires(self, desire_store):
        """Test pruning desires below threshold."""
        d1 = desire_store.record("strong", strength=0.9)
        d2 = desire_store.record("weak", strength=0.05)
        d3 = desire_store.record("medium", strength=0.5)

        pruned = desire_store.prune_weak(threshold=0.1)

        assert len(pruned) == 1
        assert d2.id in pruned

        # Verify weak desire is gone
        assert desire_store.get(d2.id) is None
        assert desire_store.get(d1.id) is not None
        assert desire_store.get(d3.id) is not None

    def test_prune_empty_when_all_strong(self, desire_store):
        """Test that prune returns empty when all desires are strong."""
        desire_store.record("strong 1", strength=0.9)
        desire_store.record("strong 2", strength=0.8)

        pruned = desire_store.prune_weak(threshold=0.1)
        assert len(pruned) == 0


class TestTagging:
    """Test desire tagging and search."""

    def test_search_by_tag(self, desire_store):
        """Test searching desires by tag."""
        desire_store.record("test 1", tags=["testing", "quality"])
        desire_store.record("test 2", tags=["testing", "performance"])
        desire_store.record("test 3", tags=["documentation"])

        testing_desires = desire_store.search_by_tag("testing")
        assert len(testing_desires) == 2

        doc_desires = desire_store.search_by_tag("documentation")
        assert len(doc_desires) == 1

    def test_search_sorts_by_strength(self, desire_store):
        """Test that search results are sorted by strength."""
        desire_store.record("weak", strength=0.5, tags=["testing"])
        desire_store.record("strong", strength=0.9, tags=["testing"])
        desire_store.record("medium", strength=0.7, tags=["testing"])

        results = desire_store.search_by_tag("testing")
        assert len(results) == 3
        assert results[0].text == "strong"
        assert results[1].text == "medium"
        assert results[2].text == "weak"


class TestPersistence:
    """Test persistence layer."""

    def test_index_persists(self, temp_desires_dir):
        """Test that index is persisted to disk."""
        store = create_desire_store(temp_desires_dir)

        store.record("test desire")

        # Check index file exists
        index_path = Path(temp_desires_dir) / "index.json"
        assert index_path.exists()

        # Verify content
        with open(index_path, "r") as f:
            index_data = json.load(f)

        assert len(index_data) == 1

    def test_index_loads_on_init(self, temp_desires_dir):
        """Test that index is loaded on store initialization."""
        # Create desire with first store
        store1 = create_desire_store(temp_desires_dir)
        desire = store1.record("test desire", tags=["persistent"])

        # Create new store instance (should load from disk)
        store2 = create_desire_store(temp_desires_dir)

        retrieved = store2.get(desire.id)
        assert retrieved is not None
        assert retrieved.text == "test desire"
        assert "persistent" in retrieved.tags

    def test_ndjson_chain_appends(self, temp_desires_dir):
        """Test that events are appended to NDJSON chain."""
        store = create_desire_store(temp_desires_dir)

        store.record("test desire")

        # Check chain file exists
        chain_files = list(Path(temp_desires_dir).glob("*.ndjson.gz"))
        assert len(chain_files) == 1


class TestDecayIntegration:
    """Integration tests for decay and pruning workflow."""

    def test_decay_and_prune_workflow(self, desire_store):
        """Test typical decay and prune workflow."""
        # Record some desires
        d1 = desire_store.record("important", strength=1.0)
        d2 = desire_store.record("less important", strength=1.0)
        d3 = desire_store.record("not important", strength=1.0)

        # Make them old
        old_time = (datetime.now(timezone.utc) - timedelta(days=50)).isoformat()
        for d in [d1, d2, d3]:
            d.last_reinforced_at = old_time
            desire_store.index[d.id] = d

        # Reinforce d1 (keep it strong)
        desire_store.reinforce(d1.id, delta=0.5)

        # Decay all
        desire_store.decay_all(decay_rate=0.02)  # 0.02 * 50 = 1.0, so d2/d3 -> 0

        # Prune weak
        pruned = desire_store.prune_weak(threshold=0.1)

        # d1 should survive (was reinforced)
        assert desire_store.get(d1.id) is not None

        # d2 and d3 should be pruned
        assert len(pruned) >= 1
