"""Desire store for autonomous wish tracking with strength decay.

Persistence:
- Append-only NDJSON chain: var/desires/YYYY-MM.ndjson.gz
- Compact KV index: var/desires/index.json

Desires are vague wishes that Astra can express before they become concrete goals.
They decay over time unless reinforced, creating a natural priority system.

Example flow:
1. Astra records desire: "I wish I had better test coverage"
2. Desire starts at strength 1.0
3. Over time, strength decays: 1.0 -> 0.95 -> 0.9 -> ...
4. If desire is acted upon (converted to goal), it gets reinforced
5. Weak desires (strength < 0.1) are automatically pruned
"""

import gzip
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Desire:
    """A vague wish or aspiration before it becomes a concrete goal."""

    id: str  # des_<sha8>
    text: str  # Free-form desire text
    strength: float  # 0.0 to 1.0, decays over time
    created_at: str  # ISO8601 timestamp
    last_reinforced_at: str  # ISO8601 timestamp
    tags: List[str] = field(default_factory=list)  # Optional categorization
    context: Dict = field(default_factory=dict)  # Optional context

    @staticmethod
    def generate_id(text: str, created_at: datetime) -> str:
        """Generate deterministic desire ID from text and timestamp.

        Args:
            text: Desire text
            created_at: Creation timestamp

        Returns:
            Deterministic ID: des_<sha8>
        """
        # Include timestamp to allow duplicate text at different times
        composite = f"{text}:{created_at.isoformat()}"
        hash_digest = hashlib.sha256(composite.encode()).hexdigest()
        return f"des_{hash_digest[:8]}"


class DesireStore:
    """Store for tracking desires with strength decay and persistence."""

    def __init__(self, desires_dir: str = "var/desires"):
        """Initialize desire store.

        Args:
            desires_dir: Directory for desire persistence
        """
        self.desires_dir = Path(desires_dir)
        self.desires_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.desires_dir / "index.json"

        # In-memory index: desire_id -> Desire
        self.index: Dict[str, Desire] = {}

        # Load index from disk
        self._load_index()

    def _load_index(self) -> None:
        """Load desire index from disk."""
        if not self.index_path.exists():
            logger.info("No desire index found, starting fresh")
            return

        try:
            with open(self.index_path, "r") as f:
                index_data = json.load(f)

            for desire_id, desire_dict in index_data.items():
                self.index[desire_id] = Desire(**desire_dict)

            logger.info(f"Loaded {len(self.index)} desires from index")

        except Exception as e:
            logger.error(f"Failed to load desire index: {e}")
            # Continue with empty index

    def _save_index(self) -> None:
        """Save desire index to disk."""
        try:
            # Convert Desire objects to dicts
            index_data = {
                desire_id: asdict(desire)
                for desire_id, desire in self.index.items()
            }

            # Write atomically (write to temp, then rename)
            temp_path = self.index_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(index_data, f, indent=2)

            temp_path.replace(self.index_path)

            logger.debug(f"Saved {len(self.index)} desires to index")

        except Exception as e:
            logger.error(f"Failed to save desire index: {e}")

    def _append_to_chain(self, event: Dict) -> None:
        """Append event to NDJSON chain.

        Args:
            event: Event to append (recorded, reinforced, decayed, pruned)
        """
        now = datetime.now(timezone.utc)
        year_month = now.strftime("%Y-%m")
        chain_path = self.desires_dir / f"{year_month}.ndjson.gz"

        try:
            # Add timestamp to event
            event["_timestamp"] = now.isoformat()

            # Append to gzipped NDJSON
            with gzip.open(chain_path, "at", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

            logger.debug(f"Appended event to chain: {chain_path}")

        except Exception as e:
            logger.error(f"Failed to append to chain: {e}")

    def record(
        self,
        text: str,
        strength: float = 1.0,
        tags: Optional[List[str]] = None,
        context: Optional[Dict] = None,
    ) -> Desire:
        """Record a new desire.

        Args:
            text: Free-form desire text
            strength: Initial strength (0.0 to 1.0)
            tags: Optional categorization tags
            context: Optional context dictionary

        Returns:
            Created desire

        Raises:
            ValueError: If strength is out of range
        """
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {strength}")

        now = datetime.now(timezone.utc)
        desire_id = Desire.generate_id(text, now)

        # Create desire
        desire = Desire(
            id=desire_id,
            text=text,
            strength=strength,
            created_at=now.isoformat(),
            last_reinforced_at=now.isoformat(),
            tags=tags or [],
            context=context or {},
        )

        # Add to index
        self.index[desire_id] = desire
        self._save_index()

        # Append to chain
        self._append_to_chain(
            {"event": "desire_recorded", "desire_id": desire_id, "desire": asdict(desire)}
        )

        logger.info(f"Recorded desire: {desire_id} (text: {text[:50]}...)")
        return desire

    def get(self, desire_id: str) -> Optional[Desire]:
        """Get desire by ID.

        Args:
            desire_id: Desire ID

        Returns:
            Desire or None if not found
        """
        return self.index.get(desire_id)

    def list_all(self, min_strength: float = 0.0) -> List[Desire]:
        """List all desires above minimum strength.

        Args:
            min_strength: Minimum strength threshold (0.0 to 1.0)

        Returns:
            List of desires sorted by strength descending
        """
        desires = [
            d for d in self.index.values()
            if d.strength >= min_strength
        ]
        return sorted(desires, key=lambda d: d.strength, reverse=True)

    def list_top(self, limit: int = 10, min_strength: float = 0.0) -> List[Desire]:
        """List top N desires by strength.

        Args:
            limit: Maximum number of desires to return
            min_strength: Minimum strength threshold

        Returns:
            List of top desires sorted by strength descending
        """
        all_desires = self.list_all(min_strength=min_strength)
        return all_desires[:limit]

    def reinforce(self, desire_id: str, delta: float = 0.1) -> Desire:
        """Reinforce a desire, increasing its strength.

        Args:
            desire_id: Desire ID
            delta: Amount to increase strength by (default: 0.1)

        Returns:
            Updated desire

        Raises:
            ValueError: If desire not found
        """
        desire = self.get(desire_id)
        if not desire:
            raise ValueError(f"Desire not found: {desire_id}")

        # Increase strength (capped at 1.0)
        desire.strength = min(1.0, desire.strength + delta)
        desire.last_reinforced_at = datetime.now(timezone.utc).isoformat()

        self._save_index()
        self._append_to_chain(
            {
                "event": "desire_reinforced",
                "desire_id": desire_id,
                "delta": delta,
                "new_strength": desire.strength,
            }
        )

        logger.info(f"Reinforced desire {desire_id} by {delta} -> {desire.strength}")
        return desire

    def decay_all(self, decay_rate: float = 0.01) -> Dict[str, float]:
        """Apply decay to all desires.

        Args:
            decay_rate: Amount to decrease strength by (default: 0.01 per day)

        Returns:
            Dict mapping desire_id to new strength
        """
        now = datetime.now(timezone.utc)
        decayed = {}

        for desire_id, desire in list(self.index.items()):
            # Calculate time since last reinforcement
            last_reinforced = datetime.fromisoformat(desire.last_reinforced_at)
            days_since = (now - last_reinforced).total_seconds() / 86400

            # Apply decay: strength -= decay_rate * days_since
            decay_amount = decay_rate * days_since
            new_strength = max(0.0, desire.strength - decay_amount)

            if new_strength != desire.strength:
                desire.strength = new_strength
                desire.last_reinforced_at = now.isoformat()  # Update timestamp
                decayed[desire_id] = new_strength

        if decayed:
            self._save_index()
            self._append_to_chain(
                {
                    "event": "desires_decayed",
                    "decay_rate": decay_rate,
                    "decayed_count": len(decayed),
                }
            )
            logger.info(f"Decayed {len(decayed)} desires by rate {decay_rate}")

        return decayed

    def prune_weak(self, threshold: float = 0.1) -> List[str]:
        """Remove desires below strength threshold.

        Args:
            threshold: Strength threshold for pruning (default: 0.1)

        Returns:
            List of pruned desire IDs
        """
        pruned_ids = []

        for desire_id, desire in list(self.index.items()):
            if desire.strength < threshold:
                del self.index[desire_id]
                pruned_ids.append(desire_id)

        if pruned_ids:
            self._save_index()
            self._append_to_chain(
                {
                    "event": "desires_pruned",
                    "threshold": threshold,
                    "pruned_ids": pruned_ids,
                    "pruned_count": len(pruned_ids),
                }
            )
            logger.info(f"Pruned {len(pruned_ids)} weak desires (threshold: {threshold})")

        return pruned_ids

    def search_by_tag(self, tag: str) -> List[Desire]:
        """Search desires by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of desires with matching tag, sorted by strength descending
        """
        matching = [
            d for d in self.index.values()
            if tag in d.tags
        ]
        return sorted(matching, key=lambda d: d.strength, reverse=True)


def create_desire_store(desires_dir: str = "var/desires") -> DesireStore:
    """Factory function to create DesireStore.

    Args:
        desires_dir: Directory for desire persistence

    Returns:
        Initialized DesireStore
    """
    return DesireStore(desires_dir=desires_dir)
