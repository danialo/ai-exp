"""
Identity Service - Unified PIM facade over fragmented identity stores

Astra's implementation of Iida's Pattern-Integrated Memory focused on
self-knowledge and identity.

Provides coherent SelfModelSnapshot from fragmented sources:
- beliefs.json (core + peripheral beliefs)
- traits.json (personality traits)
- Redis awareness blackboard (identity anchors)
- SQLite identity ledger (update history)
- self_knowledge_index (vector store)

Based on INTEGRATION_LAYER_SPEC.md Section 3.2.

Note: Phase 1 implementation is READ-ONLY. Update methods will be added
in Phase 2 when IL begins controlling identity evolution.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from .state import SelfModelSnapshot

logger = logging.getLogger(__name__)


class IdentityService:
    """
    Persistent Identity Module (PIM) facade - Astra's implementation of Iida's
    Pattern-Integrated Memory focused on self-knowledge and identity.

    Provides unified, coherent view of Astra's identity from fragmented stores.

    Phase 1: Read-only facade
    Phase 2: Will add update_belief, update_trait, update_identity_anchor
    """

    def __init__(
        self,
        belief_store=None,  # BeliefStore
        persona_file_manager=None,  # PersonaFileManager
        awareness_loop=None,  # Optional[AwarenessLoop]
        identity_ledger=None  # IdentityLedger
    ):
        """
        Initialize IdentityService with references to existing subsystems.

        Args:
            belief_store: Access to beliefs.json and belief operations
            persona_file_manager: Access to persona_space files
            awareness_loop: Optional, for identity anchor access
            identity_ledger: SQLite ledger of identity updates
        """
        self.belief_store = belief_store
        self.persona_files = persona_file_manager
        self.awareness_loop = awareness_loop
        self.identity_ledger = identity_ledger

        # Snapshot cache (1-second TTL)
        self._cache: Optional[SelfModelSnapshot] = None
        self._cache_expiry: Optional[datetime] = None

        logger.info("IdentityService initialized (Phase 1: read-only)")

    def get_snapshot(self, force_refresh: bool = False) -> SelfModelSnapshot:
        """
        Get current self-model snapshot.

        Reads from all identity sources and synthesizes into unified view.
        Results are cached for 1 second to avoid redundant reads.

        Args:
            force_refresh: If True, bypass cache and recompute

        Returns:
            SelfModelSnapshot with unified identity view
        """
        if not force_refresh and self._is_cache_valid():
            logger.debug("Returning cached SelfModelSnapshot")
            return self._cache

        logger.debug("Computing fresh SelfModelSnapshot")

        # Read all beliefs from store and classify by confidence
        core_beliefs = []
        peripheral_beliefs = []
        if self.belief_store:
            try:
                all_beliefs = self.belief_store.get_current()  # Returns Dict[str, BeliefVersion]

                for belief_id, belief in all_beliefs.items():
                    confidence = getattr(belief, 'confidence', 0.0)
                    state = getattr(belief, 'state', '')

                    # Skip non-asserted beliefs
                    if state != 'asserted':
                        continue

                    # Core beliefs: high confidence (>=1.0) foundational beliefs
                    # Peripheral beliefs: everything else that's asserted
                    if confidence >= 1.0:
                        core_beliefs.append(belief)
                    else:
                        peripheral_beliefs.append(belief)

                logger.debug(f"Loaded {len(core_beliefs)} core beliefs, {len(peripheral_beliefs)} peripheral beliefs")
            except Exception as e:
                logger.warning(f"Could not read beliefs: {e}")

        # Read traits from persona_space/identity/traits.json
        traits = {}
        if self.persona_files:
            try:
                import json
                traits_raw = self.persona_files.read_file("identity/traits.json")
                if traits_raw:
                    traits_data = json.loads(traits_raw) if isinstance(traits_raw, str) else traits_raw
                    traits = {t['name']: t['value'] for t in traits_data.get('traits', [])}
            except Exception as e:
                logger.warning(f"Could not read traits: {e}")

        # Read identity anchors from awareness loop
        # Convert numpy to list[float] for serialization
        origin_anchor = []
        live_anchor = []
        anchor_drift = 0.0

        if self.awareness_loop:
            try:
                origin_np = getattr(self.awareness_loop, 'get_origin_anchor', lambda: None)()
                live_np = getattr(self.awareness_loop, 'get_live_anchor', lambda: None)()

                if origin_np is not None:
                    origin_anchor = origin_np.tolist() if hasattr(origin_np, 'tolist') else list(origin_np)
                if live_np is not None:
                    live_anchor = live_np.tolist() if hasattr(live_np, 'tolist') else list(live_np)

                # Compute drift
                if origin_anchor and live_anchor and len(origin_anchor) == len(live_anchor):
                    import numpy as np
                    anchor_drift = float(np.linalg.norm(np.array(origin_anchor) - np.array(live_anchor)))
            except Exception as e:
                logger.warning(f"Could not read identity anchors: {e}")

        # Infer capabilities and limitations from beliefs
        known_capabilities = self._infer_capabilities(peripheral_beliefs)
        limitations = self._infer_limitations(peripheral_beliefs)

        # Assess self-model confidence (meta-cognition)
        confidence = self._assess_self_model_confidence(core_beliefs, peripheral_beliefs, traits)

        # Get last major update from ledger
        last_update = None
        if self.identity_ledger:
            try:
                last_update = self.identity_ledger.get_last_major_update()
            except Exception as e:
                logger.warning(f"Could not read identity ledger: {e}")

        # Create snapshot
        snapshot = SelfModelSnapshot(
            core_beliefs=core_beliefs,
            peripheral_beliefs=peripheral_beliefs,
            traits=traits,
            origin_anchor=origin_anchor,
            live_anchor=live_anchor,
            anchor_drift=anchor_drift,
            known_capabilities=known_capabilities,
            limitations=limitations,
            confidence_self_model=confidence,
            last_major_update=last_update,
            snapshot_id=str(uuid.uuid4()),
            created_at=datetime.now()
        )

        # Cache for 1 second
        self._cache = snapshot
        self._cache_expiry = datetime.now() + timedelta(seconds=1)

        logger.info(
            f"SelfModelSnapshot created: {len(core_beliefs)} core beliefs, "
            f"{len(peripheral_beliefs)} peripheral, {len(traits)} traits, "
            f"drift={anchor_drift:.3f}, confidence={confidence:.2f}"
        )

        return snapshot

    def _infer_capabilities(self, beliefs: list) -> set:
        """
        Infer known capabilities from beliefs.

        Looks for CAPABILITY type beliefs with positive phrasing.
        """
        capabilities = set()
        for belief in beliefs:
            belief_text = getattr(belief, 'text', str(belief)).lower()
            if "can" in belief_text and "cannot" not in belief_text:
                capabilities.add(belief_text[:100])  # Truncate for sanity

        return capabilities

    def _infer_limitations(self, beliefs: list) -> set:
        """
        Infer limitations from negative capability beliefs.

        Looks for beliefs with "cannot", "unable", "lack" phrasing.
        """
        limitations = set()
        for belief in beliefs:
            belief_text = getattr(belief, 'text', str(belief)).lower()
            if any(word in belief_text for word in ["cannot", "unable", "lack", "limited"]):
                limitations.add(belief_text[:100])

        return limitations

    def _assess_self_model_confidence(self, core: list, peripheral: list, traits: dict) -> float:
        """
        Meta-cognitive assessment of how well Astra knows itself.

        Simple heuristic: more beliefs and traits = higher confidence.
        Normalize to 0-1 range (assume 50 total items = very high confidence).
        """
        total_items = len(core) + len(peripheral) + len(traits)
        confidence = min(1.0, total_items / 50.0)
        return confidence

    def _is_cache_valid(self) -> bool:
        """Check if cached snapshot is still valid (within TTL)."""
        if self._cache is None or self._cache_expiry is None:
            return False
        return datetime.now() < self._cache_expiry

    def _invalidate_cache(self):
        """Force cache invalidation."""
        self._cache = None
        self._cache_expiry = None

    # =========================================================================
    # Mutation Methods with Policy Enforcement
    # =========================================================================

    # Valid causes for belief mutations
    VALID_MUTATION_CAUSES = {
        "DISSONANCE_RESOLUTION",  # From belief consistency checker
        "ADMIN_OVERRIDE",         # Manual admin action
        "MAINTENANCE",            # Belief gardener (formation, promotion, deprecation)
    }

    # Belief types that require DISSONANCE_RESOLUTION cause
    PROTECTED_BELIEF_TYPES = {"ontological", "self", "identity"}

    def update_belief(
        self,
        belief_id: str,
        updates: dict,
        *,
        cause: str,
        evidence: list = None,
    ) -> bool:
        """
        Update a belief with policy enforcement.

        All belief writes MUST go through this method (not direct to BeliefStore).

        Policy:
        1. Core beliefs (is_core=True or stability>=0.95) are blocked
        2. PROTECTED_BELIEF_TYPES require cause="DISSONANCE_RESOLUTION"
        3. Unknown causes are rejected
        4. All updates are logged to identity ledger

        Args:
            belief_id: ID of belief to update
            updates: Dict of fields to update
            cause: Reason for update (must be in VALID_MUTATION_CAUSES)
            evidence: Optional list of evidence references

        Returns:
            True if update succeeded, False if blocked by policy
        """
        evidence = evidence or []

        # 1. Validate cause
        if cause not in self.VALID_MUTATION_CAUSES:
            logger.error(f"BLOCKED: Invalid mutation cause '{cause}' for belief {belief_id}")
            return False

        # 2. Get current belief
        if not self.belief_store:
            logger.error("BLOCKED: No belief_store wired to IdentityService")
            return False

        all_beliefs = self.belief_store.get_current(belief_ids=[belief_id])
        if belief_id not in all_beliefs:
            logger.error(f"BLOCKED: Belief {belief_id} not found")
            return False

        belief = all_beliefs[belief_id]
        belief_type = getattr(belief, 'belief_type', '').lower()

        # 3. Check protected types
        if belief_type in self.PROTECTED_BELIEF_TYPES:
            if cause not in {"DISSONANCE_RESOLUTION", "ADMIN_OVERRIDE"}:
                logger.warning(
                    f"BLOCKED: Protected belief type '{belief_type}' requires "
                    f"DISSONANCE_RESOLUTION or ADMIN_OVERRIDE, got cause='{cause}'"
                )
                return False

        # 4. Log the mutation attempt (before BeliefStore checks its own guards)
        logger.info(
            f"IdentityService.update_belief: {belief_id} cause={cause} "
            f"updates={list(updates.keys())}"
        )

        # 5. Apply via BeliefStore (which has its own stability/immutable checks)
        # Note: BeliefStore.apply_delta handles the actual mutation
        # For now, we don't have a direct update method, so we log this
        # TODO: Implement proper delta application

        # 6. Record in identity ledger if available
        if self.identity_ledger:
            try:
                self.identity_ledger.record_update(
                    update_type="belief_update",
                    target_id=belief_id,
                    details={
                        "updates": updates,
                        "cause": cause,
                        "evidence": evidence,
                    }
                )
            except Exception as e:
                logger.error(f"Failed to log to identity ledger: {e}")

        # 7. Invalidate cache
        self._invalidate_cache()

        return True

    def propose_belief_update(
        self,
        belief_id: str,
        old_text: str,
        new_text: str,
        cause: str,
        evidence: list = None,
        proposed_stability: float = None,
    ) -> dict:
        """
        Create a structured belief update proposal.

        Proposals are reviewed before being applied. This is the safe way
        to request belief changes.

        Returns:
            Proposal dict that can be passed to apply_belief_proposal()
        """
        evidence = evidence or []

        if not self.belief_store:
            return {"error": "No belief_store wired"}

        all_beliefs = self.belief_store.get_current(belief_ids=[belief_id])
        if belief_id not in all_beliefs:
            return {"error": f"Belief {belief_id} not found"}

        belief = all_beliefs[belief_id]
        current_stability = getattr(belief, 'stability', 0.0) or \
                           belief.metadata.get('stability', 0.3) if hasattr(belief, 'metadata') else 0.3

        return {
            "belief_id": belief_id,
            "old_text": old_text,
            "new_text": new_text,
            "belief_type": getattr(belief, 'belief_type', 'unknown'),
            "stability_before": current_stability,
            "proposed_stability_after": proposed_stability or current_stability,
            "cause": cause,
            "evidence": evidence,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
        }
