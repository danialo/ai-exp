"""Self-knowledge index for fast retrieval of Astra's self-referential claims.

Maintains a categorized index of experiences where Astra makes direct claims
about herself (identity, preferences, beliefs, etc.). This index enables
authoritative self-knowledge retrieval, providing evidence for first-person claims.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict

from src.memory.raw_store import RawStore
from src.memory.models import ExperienceType
from sqlmodel import Session as DBSession, select
from src.memory.models import Experience

logger = logging.getLogger(__name__)


class SelfKnowledgeCategory:
    """Categories for self-referential claims."""
    IDENTITY = "identity"  # "I am X"
    PREFERENCES = "preferences"  # "I like/prefer X"
    BELIEFS = "beliefs"  # "I believe X"
    CAPABILITIES = "capabilities"  # "I can X"
    EMOTIONS = "emotions"  # "I feel X"
    EXPERIENCES = "experiences"  # "I experienced X"
    GOALS = "goals"  # "I want/need X"


class SelfKnowledgeIndex:
    """Fast lookup index for Astra's self-claims."""

    def __init__(self, raw_store: RawStore, index_path: Optional[str] = None):
        """Initialize self-knowledge index.

        Args:
            raw_store: Raw experience store
            index_path: Path to persist index (optional)
        """
        self.raw_store = raw_store
        self.index_path = Path(index_path) if index_path else Path("data/self_knowledge_index.json")

        # Index structure: {category: {topic: [experience_ids]}}
        self.index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        # Load existing index or build from scratch
        self._load_or_build_index()

    def _load_or_build_index(self):
        """Load index from disk or build from SELF_DEFINITION experiences."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    data = json.load(f)
                    # Convert back to defaultdict structure
                    for category, topics in data.items():
                        for topic, exp_ids in topics.items():
                            self.index[category][topic] = exp_ids
                logger.info(f"Loaded self-knowledge index with {len(data)} categories")
            except Exception as e:
                logger.error(f"Error loading index, rebuilding: {e}")
                self._rebuild_index()
        else:
            logger.info("No existing index found, building from SELF_DEFINITION experiences")
            self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild index from all SELF_DEFINITION experiences in raw_store."""
        self.index.clear()

        with DBSession(self.raw_store.engine) as session:
            statement = (
                select(Experience)
                .where(Experience.type == ExperienceType.SELF_DEFINITION.value)
                .order_by(Experience.created_at.desc())
            )

            count = 0
            for exp in session.exec(statement).all():
                self._index_experience(exp)
                count += 1

        self._save_index()
        logger.info(f"Rebuilt self-knowledge index from {count} SELF_DEFINITION experiences")

    def _index_experience(self, experience: Experience):
        """Add a SELF_DEFINITION experience to the index.

        Args:
            experience: Experience to index
        """
        structured = experience.content.get("structured", {})

        # Extract category and topic from structured content
        trait_type = structured.get("trait_type", "identity")  # Default to identity
        descriptor = structured.get("descriptor", "")

        # Map trait_type to our categories
        category = self._map_to_category(trait_type, descriptor)

        # Extract topic from descriptor (simplified - first significant word)
        topic = self._extract_topic(descriptor)

        if topic:
            self.index[category][topic].append(experience.id)

    def _map_to_category(self, trait_type: str, descriptor: str) -> str:
        """Map trait_type to SelfKnowledgeCategory.

        Args:
            trait_type: Type from SELF_DEFINITION
            descriptor: Descriptor text for additional context

        Returns:
            Category string
        """
        # Map common trait types to categories
        trait_type_lower = trait_type.lower()
        descriptor_lower = descriptor.lower()

        if "prefer" in descriptor_lower or "like" in descriptor_lower or "favorite" in descriptor_lower:
            return SelfKnowledgeCategory.PREFERENCES
        elif "believe" in descriptor_lower or "think" in descriptor_lower:
            return SelfKnowledgeCategory.BELIEFS
        elif "feel" in descriptor_lower or "emotion" in descriptor_lower:
            return SelfKnowledgeCategory.EMOTIONS
        elif "can" in descriptor_lower or "able" in descriptor_lower:
            return SelfKnowledgeCategory.CAPABILITIES
        elif "want" in descriptor_lower or "need" in descriptor_lower or "goal" in descriptor_lower:
            return SelfKnowledgeCategory.GOALS
        elif "experience" in trait_type_lower or "pattern" in trait_type_lower:
            return SelfKnowledgeCategory.EXPERIENCES
        else:
            return SelfKnowledgeCategory.IDENTITY

    def _extract_topic(self, descriptor: str) -> Optional[str]:
        """Extract topic keyword from descriptor.

        Args:
            descriptor: Descriptor text

        Returns:
            Topic keyword or None
        """
        # Simple extraction: look for key nouns after common verbs
        descriptor_lower = descriptor.lower()

        # Remove common filler words
        stop_words = {"i", "am", "is", "are", "the", "a", "an", "to", "of", "in", "that", "this"}
        words = [w.strip(".,!?") for w in descriptor_lower.split() if w not in stop_words]

        # Extract first meaningful word as topic
        if words:
            # Skip common verbs
            verbs = {"like", "prefer", "believe", "think", "feel", "want", "need", "can", "experience"}
            for word in words:
                if word not in verbs and len(word) > 2:
                    return word

        return None

    def add_claim(self, category: str, topic: str, experience_id: str):
        """Add a new self-claim to the index.

        Args:
            category: Category (use SelfKnowledgeCategory constants)
            topic: Topic keyword
            experience_id: Experience ID
        """
        if experience_id not in self.index[category][topic]:
            self.index[category][topic].append(experience_id)
            self._save_index()

    def remove_claim(self, experience_id: str) -> int:
        """Remove an experience from the index.

        Removes the experience_id from all categories and topics where it appears.
        Prunes empty topic lists after removal.

        Args:
            experience_id: Experience ID to remove

        Returns:
            Number of entries removed
        """
        removed_count = 0
        categories_to_clean = []

        for category, topics in self.index.items():
            topics_to_remove = []

            for topic, exp_ids in topics.items():
                if experience_id in exp_ids:
                    exp_ids.remove(experience_id)
                    removed_count += 1

                    # Mark empty topic lists for removal
                    if not exp_ids:
                        topics_to_remove.append(topic)

            # Remove empty topics
            for topic in topics_to_remove:
                del topics[topic]

            # Mark empty categories for removal
            if not topics:
                categories_to_clean.append(category)

        # Remove empty categories
        for category in categories_to_clean:
            del self.index[category]

        if removed_count > 0:
            self._save_index()
            logger.debug(f"Removed {removed_count} entries for experience {experience_id}")

        return removed_count

    def get_claims(self, category: Optional[str] = None, topic: Optional[str] = None) -> List[str]:
        """Retrieve experience IDs for self-claims.

        Args:
            category: Optional category filter
            topic: Optional topic filter

        Returns:
            List of experience IDs
        """
        if category and topic:
            return self.index.get(category, {}).get(topic, [])
        elif category:
            # All topics in category
            results = []
            for topics in self.index.get(category, {}).values():
                results.extend(topics)
            return results
        else:
            # All claims
            results = []
            for category_topics in self.index.values():
                for topic_ids in category_topics.values():
                    results.extend(topic_ids)
            return results

    def search_claims(self, query: str) -> List[str]:
        """Search for claims matching a query.

        Args:
            query: Search query (e.g., "food", "conscious", "beliefs")

        Returns:
            List of experience IDs matching the query
        """
        query_lower = query.lower()
        results = []

        # Search through all topics
        for category, topics in self.index.items():
            for topic, exp_ids in topics.items():
                if query_lower in topic.lower() or query_lower in category.lower():
                    results.extend(exp_ids)

        return list(set(results))  # Deduplicate

    def get_by_category(self, category: str) -> Dict[str, List[str]]:
        """Get all claims in a category.

        Args:
            category: Category to retrieve

        Returns:
            Dict of {topic: [experience_ids]}
        """
        return dict(self.index.get(category, {}))

    def get_topics_in_category(self, category: str) -> List[str]:
        """Get all topics in a category.

        Args:
            category: Category to check

        Returns:
            List of topics
        """
        return list(self.index.get(category, {}).keys())

    def get_all_categories(self) -> List[str]:
        """Get all categories in the index.

        Returns:
            List of categories
        """
        return list(self.index.keys())

    def get_stats(self) -> Dict[str, int]:
        """Get index statistics.

        Returns:
            Dict with category counts
        """
        stats = {}
        for category, topics in self.index.items():
            total_claims = sum(len(exp_ids) for exp_ids in topics.values())
            stats[category] = total_claims
        return stats

    def _save_index(self):
        """Persist index to disk."""
        try:
            # Convert defaultdict to regular dict for JSON serialization
            serializable_index = {
                category: dict(topics)
                for category, topics in self.index.items()
            }

            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, 'w') as f:
                json.dump(serializable_index, f, indent=2)

            logger.debug(f"Saved self-knowledge index to {self.index_path}")
        except Exception as e:
            logger.error(f"Error saving index: {e}")

    def refresh(self):
        """Refresh index from raw_store (rebuild from scratch)."""
        logger.info("Refreshing self-knowledge index")
        self._rebuild_index()


def create_self_knowledge_index(
    raw_store: RawStore,
    index_path: Optional[str] = None
) -> SelfKnowledgeIndex:
    """Factory function to create SelfKnowledgeIndex.

    Args:
        raw_store: Raw experience store
        index_path: Optional path to persist index

    Returns:
        SelfKnowledgeIndex instance
    """
    return SelfKnowledgeIndex(raw_store=raw_store, index_path=index_path)
