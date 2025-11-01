"""
Provenance Trust Manager - Learn actor reliability from outcomes.

Maintains EWMA-smoothed trust scores per actor (user, agent) based on
delayed credit assignment from multi-component outcome rewards.
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrustState:
    """Trust state for a single actor."""
    trust: float  # T_actor ∈ [0, 1]
    sample_count: int  # Number of outcomes observed
    last_reward: float  # Most recent reward received
    last_update_ts: float  # Timestamp of last update


@dataclass
class TrustConfig:
    """Configuration for trust learning."""
    enabled: bool = True
    alpha_0: float = 0.3  # Initial EWMA step size
    k_samples: int = 50  # Half-step by k samples
    r_min: float = 0.1  # Minimum |r| to trigger update
    persist_interval: int = 300  # Persist every 5 minutes


class ProvenanceTrust:
    """
    Manages outcome-driven trust learning for provenance actors.

    Trust scores are learned from delayed rewards based on coherence,
    conflict reduction, stability, and user validation.
    """

    def __init__(self, data_dir: Path, config: Optional[TrustConfig] = None):
        """
        Initialize provenance trust manager.

        Args:
            data_dir: Directory for trust state persistence
            config: Configuration (defaults to TrustConfig())
        """
        self.data_dir = Path(data_dir)
        self.config = config or TrustConfig()

        # Trust state per actor
        self.trust: Dict[str, TrustState] = {}

        # Initialize default actors
        self._init_default_actors()

        # Load persisted state if exists
        self._load_state()

        # Last persist timestamp
        self._last_persist_ts = time.time()

        logger.info(f"ProvenanceTrust initialized (alpha_0={self.config.alpha_0}, k={self.config.k_samples})")

    def _init_default_actors(self):
        """Initialize default actors with neutral trust (0.5)."""
        default_actors = ["user", "agent"]
        for actor in default_actors:
            if actor not in self.trust:
                self.trust[actor] = TrustState(
                    trust=0.5,
                    sample_count=0,
                    last_reward=0.0,
                    last_update_ts=0.0
                )

    def get_trust(self, actor: str) -> float:
        """
        Get current trust score for actor.

        Args:
            actor: Actor identifier (user, agent)

        Returns:
            Trust score ∈ [0, 1], defaults to 0.5 for unknown actors
        """
        if actor not in self.trust:
            return 0.5
        return self.trust[actor].trust

    def get_trust_multiplier(self, actor: str) -> float:
        """
        Get trust as a gain multiplier centered at 1.0.

        Args:
            actor: Actor identifier

        Returns:
            Multiplier ∈ [0.5, 1.5] (maps trust [0,1] to gain)
        """
        T = self.get_trust(actor)
        return 0.5 + T  # Maps [0,1] → [0.5, 1.5]

    def update_trust(
        self,
        actor: str,
        reward: float,
        now: Optional[float] = None
    ) -> None:
        """
        Update trust for actor based on outcome reward.

        Args:
            actor: Actor identifier
            reward: Reward ∈ [-1, 1] from outcome evaluation
            now: Current timestamp (default: time.time())
        """
        if not self.config.enabled:
            return

        if now is None:
            now = time.time()

        # Gate on minimum reward magnitude
        if abs(reward) < self.config.r_min:
            logger.debug(f"Skipping trust update for {actor}: |r|={abs(reward):.3f} < {self.config.r_min}")
            return

        # Initialize actor if new
        if actor not in self.trust:
            self.trust[actor] = TrustState(
                trust=0.5,
                sample_count=0,
                last_reward=0.0,
                last_update_ts=0.0
            )

        state = self.trust[actor]

        # Compute diminishing step size
        alpha = self.config.alpha_0 / (1 + state.sample_count / self.config.k_samples)

        # Map reward to target trust (q ∈ [0, 1])
        q = 0.5 * (reward + 1.0)

        # EWMA update
        old_trust = state.trust
        state.trust = (1 - alpha) * state.trust + alpha * q
        state.sample_count += 1
        state.last_reward = reward
        state.last_update_ts = now

        logger.info(
            f"Trust updated: {actor} {old_trust:.3f} → {state.trust:.3f} "
            f"(r={reward:.3f}, α={alpha:.3f}, n={state.sample_count})"
        )

        # Periodic persistence
        if now - self._last_persist_ts > self.config.persist_interval:
            self.persist()

    def get_state(self, actor: str) -> Optional[TrustState]:
        """Get full trust state for actor."""
        return self.trust.get(actor)

    def get_all_states(self) -> Dict[str, TrustState]:
        """Get all trust states."""
        return self.trust.copy()

    def persist(self) -> None:
        """Persist trust state to disk."""
        try:
            # Create persona directory if needed
            persona_dir = self.data_dir / "persona"
            persona_dir.mkdir(parents=True, exist_ok=True)

            path = persona_dir / "trust.json"

            # Serialize trust state
            data = {
                "version": 1,
                "last_persist_ts": time.time(),
                "trust": {
                    actor: asdict(state)
                    for actor, state in self.trust.items()
                }
            }

            # Atomic write
            tmp_path = path.with_suffix('.tmp')
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
            tmp_path.replace(path)

            self._last_persist_ts = time.time()
            logger.debug(f"Persisted trust state: {len(self.trust)} actors")

        except Exception as e:
            logger.error(f"Failed to persist trust state: {e}")

    def _load_state(self) -> None:
        """Load trust state from disk."""
        try:
            path = self.data_dir / "persona" / "trust.json"
            if not path.exists():
                logger.info("No persisted trust state found, using defaults")
                return

            with open(path) as f:
                data = json.load(f)

            # Validate version
            if data.get("version") != 1:
                logger.warning(f"Unknown trust state version: {data.get('version')}")
                return

            # Load trust states
            for actor, state_dict in data.get("trust", {}).items():
                self.trust[actor] = TrustState(**state_dict)

            logger.info(f"Loaded trust state: {len(self.trust)} actors")

        except Exception as e:
            logger.error(f"Failed to load trust state: {e}")

    def get_telemetry(self) -> Dict:
        """Get telemetry for status endpoint."""
        return {
            "actors": {
                actor: {
                    "trust": state.trust,
                    "multiplier": self.get_trust_multiplier(actor),
                    "sample_count": state.sample_count,
                    "last_reward": state.last_reward,
                    "last_update": state.last_update_ts,
                }
                for actor, state in self.trust.items()
            },
            "config": {
                "alpha_0": self.config.alpha_0,
                "k_samples": self.config.k_samples,
                "r_min": self.config.r_min,
            }
        }


def create_provenance_trust(
    data_dir: Path,
    config: Optional[TrustConfig] = None
) -> ProvenanceTrust:
    """
    Factory function to create provenance trust manager.

    Args:
        data_dir: Directory for persistence
        config: Optional configuration

    Returns:
        ProvenanceTrust instance
    """
    return ProvenanceTrust(data_dir=data_dir, config=config)
