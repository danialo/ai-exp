"""
Temporal Anchor - Maintains identity coherence across consolidation cycles.

Part of Memory Consolidation Layer (Phase 4).

Process:
1. After each consolidation, recompute identity embedding
2. Compare to origin_anchor
3. If drift > threshold, log to identity_ledger
4. Store new anchor as live_anchor
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from src.services.identity_ledger import append_event, LedgerEvent

logger = logging.getLogger(__name__)


@dataclass
class AnchorConfig:
    """Configuration for temporal anchoring."""
    # Drift thresholds
    drift_warning_threshold: float = 0.15  # Log warning
    drift_critical_threshold: float = 0.30  # Trigger dissonance check

    # Update weights
    anchor_update_weight_new: float = 0.1  # 10% new, 90% existing

    # Embedding
    embedding_dim: int = 768  # Default embedding dimension


@dataclass
class AnchorUpdate:
    """Record of an anchor update."""
    timestamp: datetime
    drift_from_origin: float
    drift_from_previous: float
    narratives_incorporated: int
    action_taken: str  # "updated", "warning", "dissonance_triggered"


class TemporalAnchor:
    """
    Manages identity continuity across consolidation cycles.

    Tracks how much Astra's identity embedding drifts from the origin
    anchor established at initialization.
    """

    def __init__(
        self,
        identity_service=None,
        embedding_provider=None,
        config: Optional[AnchorConfig] = None,
    ):
        """Initialize temporal anchor.

        Args:
            identity_service: IdentityService for anchor access
            embedding_provider: Embedding provider for narrative encoding
            config: Anchor configuration
        """
        self.identity_service = identity_service
        self.embedding_provider = embedding_provider
        self.config = config or AnchorConfig()

        # Track update history
        self.update_history: List[AnchorUpdate] = []
        self._previous_anchor: Optional[np.ndarray] = None

        logger.info("TemporalAnchor initialized")

    def get_origin_anchor(self) -> Optional[np.ndarray]:
        """Get the origin anchor (baseline identity)."""
        if not self.identity_service:
            return None

        try:
            snapshot = self.identity_service.get_snapshot()
            if snapshot.origin_anchor:
                return np.array(snapshot.origin_anchor)
        except Exception as e:
            logger.warning(f"Could not get origin anchor: {e}")

        return None

    def get_live_anchor(self) -> Optional[np.ndarray]:
        """Get the current live anchor."""
        if not self.identity_service:
            return None

        try:
            snapshot = self.identity_service.get_snapshot()
            if snapshot.live_anchor:
                return np.array(snapshot.live_anchor)
        except Exception as e:
            logger.warning(f"Could not get live anchor: {e}")

        return None

    def compute_drift(
        self,
        anchor1: np.ndarray,
        anchor2: np.ndarray
    ) -> float:
        """Compute cosine distance between two anchors."""
        if anchor1 is None or anchor2 is None:
            return 0.0

        if len(anchor1) != len(anchor2):
            logger.warning(f"Anchor dimension mismatch: {len(anchor1)} vs {len(anchor2)}")
            return 0.0

        # Cosine distance = 1 - cosine_similarity
        norm1 = np.linalg.norm(anchor1)
        norm2 = np.linalg.norm(anchor2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(anchor1, anchor2) / (norm1 * norm2)
        return float(1.0 - similarity)

    async def update_after_consolidation(
        self,
        new_narratives: List[str],
    ) -> Dict[str, Any]:
        """
        Update anchor after consolidation cycle.

        Args:
            new_narratives: List of new narrative texts to incorporate

        Returns:
            Summary of update including drift metrics
        """
        result = {
            "narratives_incorporated": len(new_narratives),
            "drift_from_origin": 0.0,
            "drift_from_previous": 0.0,
            "action": "none",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Get current anchors
        origin_anchor = self.get_origin_anchor()
        live_anchor = self.get_live_anchor()

        if origin_anchor is None:
            logger.warning("No origin anchor available - skipping update")
            result["action"] = "skipped_no_origin"
            return result

        if live_anchor is None:
            live_anchor = origin_anchor.copy()
            logger.info("No live anchor - initializing from origin")

        # Store previous for drift calculation
        previous_anchor = self._previous_anchor if self._previous_anchor is not None else live_anchor.copy()

        # Compute new embedding from narratives (if embedding provider available)
        if self.embedding_provider and new_narratives:
            try:
                # Concatenate narratives for embedding
                combined_text = " ".join(new_narratives[:10])  # Limit for embedding
                narrative_embedding = self.embedding_provider.get_embedding(combined_text)

                if narrative_embedding is not None:
                    narrative_embedding = np.array(narrative_embedding)

                    # Ensure dimension match
                    if len(narrative_embedding) == len(live_anchor):
                        # Weighted update: 90% existing, 10% new
                        weight = self.config.anchor_update_weight_new
                        updated_anchor = (1 - weight) * live_anchor + weight * narrative_embedding

                        # Normalize
                        norm = np.linalg.norm(updated_anchor)
                        if norm > 0:
                            updated_anchor = updated_anchor / norm

                        live_anchor = updated_anchor
                    else:
                        logger.warning(f"Embedding dimension mismatch: {len(narrative_embedding)} vs {len(live_anchor)}")

            except Exception as e:
                logger.warning(f"Could not embed narratives: {e}")

        # Calculate drift metrics
        drift_from_origin = self.compute_drift(live_anchor, origin_anchor)
        drift_from_previous = self.compute_drift(live_anchor, previous_anchor)

        result["drift_from_origin"] = round(drift_from_origin, 4)
        result["drift_from_previous"] = round(drift_from_previous, 4)

        # Determine action based on drift
        action = "updated"

        if drift_from_origin >= self.config.drift_critical_threshold:
            action = "dissonance_triggered"
            self._log_critical_drift(drift_from_origin, new_narratives)
            logger.warning(f"CRITICAL: Identity drift {drift_from_origin:.3f} exceeds threshold")

        elif drift_from_origin >= self.config.drift_warning_threshold:
            action = "warning"
            self._log_drift_warning(drift_from_origin, new_narratives)
            logger.info(f"WARNING: Identity drift {drift_from_origin:.3f} approaching threshold")

        result["action"] = action

        # Update live anchor in identity service
        if self.identity_service:
            try:
                self._update_identity_service_anchor(live_anchor.tolist())
            except Exception as e:
                logger.error(f"Failed to update identity service anchor: {e}")

        # Store for next comparison
        self._previous_anchor = live_anchor.copy()

        # Record update
        update = AnchorUpdate(
            timestamp=datetime.now(timezone.utc),
            drift_from_origin=drift_from_origin,
            drift_from_previous=drift_from_previous,
            narratives_incorporated=len(new_narratives),
            action_taken=action,
        )
        self.update_history.append(update)

        logger.info(f"Anchor update: drift_origin={drift_from_origin:.3f}, "
                   f"drift_prev={drift_from_previous:.3f}, action={action}")

        return result

    def _update_identity_service_anchor(self, new_anchor: List[float]):
        """Update live anchor in identity service."""
        if not self.identity_service:
            return

        # The identity service gets anchor from awareness_loop
        # We need to update awareness_loop's live_anchor
        if hasattr(self.identity_service, 'awareness_loop') and self.identity_service.awareness_loop:
            try:
                if hasattr(self.identity_service.awareness_loop, 'set_live_anchor'):
                    self.identity_service.awareness_loop.set_live_anchor(np.array(new_anchor))
                elif hasattr(self.identity_service.awareness_loop, 'live_anchor'):
                    self.identity_service.awareness_loop.live_anchor = np.array(new_anchor)
            except Exception as e:
                logger.warning(f"Could not update awareness_loop anchor: {e}")

    def _log_drift_warning(self, drift: float, narratives: List[str]):
        """Log a drift warning to identity ledger."""
        try:
            append_event(LedgerEvent(
                ts=datetime.now(timezone.utc).timestamp(),
                schema=2,
                event="identity_drift_warning",
                beliefs_touched=[],
                evidence_refs=[],
                meta={
                    "drift": drift,
                    "threshold": self.config.drift_warning_threshold,
                    "narratives_count": len(narratives),
                    "narrative_preview": narratives[0][:100] if narratives else "",
                },
            ))
        except Exception as e:
            logger.warning(f"Could not log drift warning: {e}")

    def _log_critical_drift(self, drift: float, narratives: List[str]):
        """Log critical drift to identity ledger and potentially trigger dissonance."""
        try:
            append_event(LedgerEvent(
                ts=datetime.now(timezone.utc).timestamp(),
                schema=2,
                event="identity_drift_critical",
                beliefs_touched=[],
                evidence_refs=[],
                meta={
                    "drift": drift,
                    "threshold": self.config.drift_critical_threshold,
                    "narratives_count": len(narratives),
                    "action": "dissonance_check_recommended",
                },
            ))
        except Exception as e:
            logger.warning(f"Could not log critical drift: {e}")

    def get_drift_history(self, limit: int = 10) -> List[Dict]:
        """Get recent drift history."""
        recent = self.update_history[-limit:]
        return [
            {
                "timestamp": u.timestamp.isoformat(),
                "drift_from_origin": u.drift_from_origin,
                "drift_from_previous": u.drift_from_previous,
                "narratives_incorporated": u.narratives_incorporated,
                "action": u.action_taken,
            }
            for u in recent
        ]

    def get_current_drift(self) -> float:
        """Get current drift from origin anchor."""
        origin = self.get_origin_anchor()
        live = self.get_live_anchor()
        return self.compute_drift(live, origin) if origin is not None and live is not None else 0.0


def create_temporal_anchor(
    identity_service=None,
    embedding_provider=None,
    config: Optional[AnchorConfig] = None,
) -> TemporalAnchor:
    """Factory function to create TemporalAnchor."""
    return TemporalAnchor(
        identity_service=identity_service,
        embedding_provider=embedding_provider,
        config=config,
    )
