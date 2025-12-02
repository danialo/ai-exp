"""Versioned belief store with immutable history and optimistic locking.

Provides:
- Immutable belief history with version tracking
- Optimistic locking for concurrent updates
- Append-only delta log with daily rotation
- SHA-256 integrity checking
- State-based lifecycle (tentative → asserted → deprecated)
"""

import gzip
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.services.identity_ledger import belief_versioned_event

logger = logging.getLogger(__name__)


class BeliefState(str, Enum):
    """Belief lifecycle states."""
    TENTATIVE = "tentative"  # Newly formed, insufficient evidence
    ASSERTED = "asserted"  # Strong evidence, high confidence
    DEPRECATED = "deprecated"  # Superseded by newer belief


class DeltaOp(str, Enum):
    """Delta operation types."""
    CREATE = "create"
    UPDATE = "update"
    DEPRECATE = "deprecate"
    REINFORCE = "reinforce"


@dataclass
class BeliefVersion:
    """A versioned belief snapshot."""
    belief_id: str
    ver: int
    statement: str
    state: BeliefState
    confidence: float  # [0.0, 1.0]
    evidence_refs: List[str]  # Experience IDs supporting this belief
    updated_by: str  # "slow"|"review"|"user"|"migration"|"contrarian"
    ts: float  # Unix timestamp
    belief_type: str  # ontological|axiological|epistemological|experiential
    immutable: bool  # Core beliefs are immutable
    rationale: str
    metadata: Dict[str, Any]
    hash: str  # SHA-256 of canonical JSON (without hash field)
    # Safety fields (Phase 1.5 identity protection)
    stability: float = 0.0  # [0.0, 1.0] - beliefs >= 0.95 cannot be mutated
    is_core: bool = False  # Core identity beliefs get triple protection

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of belief (excluding hash field)."""
        data = {k: v for k, v in asdict(self).items() if k != "hash"}
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def verify_hash(self) -> bool:
        """Verify hash matches content."""
        return self.hash == self.compute_hash()


@dataclass
class BeliefDelta:
    """A delta operation on a belief."""
    belief_id: str
    from_ver: int
    to_ver: int
    op: DeltaOp
    confidence_delta: float
    evidence_refs_added: List[str]
    evidence_refs_removed: List[str]
    state_change: Optional[str]  # "tentative->asserted"
    updated_by: str
    ts: float
    reason: str  # Why this delta was applied


class MutationMode(str, Enum):
    """Mutation modes for gradual rollout."""
    DISABLED = "disabled"           # No mutations allowed
    REINFORCE_ONLY = "reinforce"    # Only reinforce existing beliefs
    CREATE_LIMITED = "create_limited"  # Creates + reinforces with limits
    FULL = "full"                   # All operations (use with caution)


@dataclass
class ScanContext:
    """Context for a single gardener scan - tracks rate limits and idempotency."""
    scan_id: str
    creates: int = 0
    updates: int = 0
    reinforcements: int = 0
    applied_keys: set = None  # (op, belief_id) tuples already applied this scan

    def __post_init__(self):
        if self.applied_keys is None:
            self.applied_keys = set()


class BeliefStore:
    """Versioned belief store with append-only history."""

    # Guardrails
    MIN_CONFIDENCE_STEP = 0.02
    MAX_CONFIDENCE_STEP = 0.15
    MIN_EVIDENCE_FOR_ASSERTED = 2

    # === NAMESPACE GUARDRAILS ===
    # Namespaces where creates are allowed
    ALLOWED_CREATE_NAMESPACES = {"auto.", "peripheral."}
    # Namespaces that are NEVER writable (except by migration actor)
    PROTECTED_NAMESPACES = {"core."}

    # === RATE LIMITS (per scan) ===
    MAX_CREATES_PER_SCAN = 10
    MAX_UPDATES_PER_SCAN = 30
    MAX_REINFORCEMENTS_PER_SCAN = 50
    MAX_CREATES_PER_CATEGORY = 5  # Limit per category (emotional spam vector)

    # === STATEMENT GUARDRAILS ===
    MIN_STATEMENT_TOKENS = 3      # Too short = noise
    MAX_STATEMENT_TOKENS = 50     # Too long = rambling junk

    # Patterns that should never become beliefs (capability/policy spam)
    BLOCKED_PATTERNS = [
        "i can browse", "i can delete", "i can create files",
        "i can execute", "i can search", "i can read",
        "i have access to", "i have tools", "my tools include",
    ]

    def __init__(self, data_dir: Path):
        """Initialize belief store.

        Args:
            data_dir: Base directory for belief storage
        """
        self.data_dir = Path(data_dir)
        self.beliefs_dir = self.data_dir / "beliefs"
        self.beliefs_dir.mkdir(parents=True, exist_ok=True)

        self.current_file = self.beliefs_dir / "current.json"
        self.index_file = self.beliefs_dir / "index.json"
        self.backup_dir = self.beliefs_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()

        # === MUTATION CONTROL ===
        # Read mode from environment (supports gradual rollout)
        # BELIEF_MUTATION_MODE: disabled | reinforce | create_limited | full
        env_mode = os.environ.get("BELIEF_MUTATION_MODE", "create_limited").lower()
        mode_map = {
            "disabled": MutationMode.DISABLED,
            "reinforce": MutationMode.REINFORCE_ONLY,
            "create_limited": MutationMode.CREATE_LIMITED,
            "full": MutationMode.FULL,
        }
        self.mutation_mode = mode_map.get(env_mode, MutationMode.CREATE_LIMITED)

        # Current scan context (set by gardener before mutations)
        self._current_scan: Optional[ScanContext] = None

        # Category counts for current scan
        self._category_creates: Dict[str, int] = {}

        # Write telemetry for current scan
        self._scan_telemetry = {
            "writes_attempted": 0,
            "writes_applied": 0,
            "writes_blocked": 0,
            "writes_idempotent_skip": 0,
            "writes_failed": 0,
            "blocked_reasons": {},
        }

        logger.warning(f"BeliefStore initialized with mutation_mode={self.mutation_mode.value}")

        # Initialize if needed
        if not self.current_file.exists():
            self._initialize_empty_store()

    # === MUTATION MODE CONTROL ===

    def set_mutation_mode(self, mode: MutationMode):
        """Set mutation mode."""
        old_mode = self.mutation_mode
        self.mutation_mode = mode
        logger.warning(f"BeliefStore mutation_mode changed: {old_mode.value} -> {mode.value}")

    def enable_mutations(self):
        """Enable limited mutations (create_limited mode)."""
        self.set_mutation_mode(MutationMode.CREATE_LIMITED)

    def disable_mutations(self):
        """Disable all mutations."""
        self.set_mutation_mode(MutationMode.DISABLED)

    # === SCAN CONTEXT MANAGEMENT ===

    def begin_scan(self, scan_id: str) -> None:
        """Begin a new scan context. Call before gardener mutations."""
        # Backup current beliefs before scan
        self._create_scan_backup(scan_id)

        self._current_scan = ScanContext(scan_id=scan_id)
        self._category_creates = {}
        self._scan_telemetry = {
            "writes_attempted": 0,
            "writes_applied": 0,
            "writes_blocked": 0,
            "writes_idempotent_skip": 0,
            "writes_failed": 0,
            "blocked_reasons": {},
        }
        logger.info(f"[BeliefStore] Scan began: {scan_id}")

    def end_scan(self) -> Dict[str, Any]:
        """End current scan context. Returns telemetry."""
        if not self._current_scan:
            return {}

        telemetry = {
            **self._scan_telemetry,
            "scan_id": self._current_scan.scan_id,
            "creates": self._current_scan.creates,
            "updates": self._current_scan.updates,
            "reinforcements": self._current_scan.reinforcements,
            "category_creates": dict(self._category_creates),
        }

        logger.info(
            f"[BeliefStore] Scan ended: {self._current_scan.scan_id} | "
            f"applied={telemetry['writes_applied']} blocked={telemetry['writes_blocked']} "
            f"creates={telemetry['creates']} reinforcements={telemetry['reinforcements']}"
        )

        self._current_scan = None
        return telemetry

    def _create_scan_backup(self, scan_id: str) -> None:
        """Create backup before scan."""
        if self.current_file.exists():
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"{timestamp}_{scan_id[:8]}.json"
            import shutil
            shutil.copy(self.current_file, backup_file)
            logger.debug(f"[BeliefStore] Backup created: {backup_file.name}")

    # Stability threshold - beliefs with stability >= this cannot be mutated
    STABILITY_THRESHOLD = 0.95

    # === GUARDRAIL CHECKS ===

    def _record_block(self, reason: str) -> None:
        """Record a blocked write in telemetry."""
        self._scan_telemetry["writes_blocked"] += 1
        self._scan_telemetry["blocked_reasons"][reason] = \
            self._scan_telemetry["blocked_reasons"].get(reason, 0) + 1

    def _check_mutation_allowed(
        self,
        belief_id: str,
        operation: str,
        statement: str = "",
        category: str = "",
        updated_by: str = "system",
    ) -> tuple[bool, str]:
        """
        Check if mutation is allowed by mode, namespace, rate limits, and content guardrails.

        Returns:
            (allowed: bool, reason: str) - reason is empty if allowed
        """
        self._scan_telemetry["writes_attempted"] += 1

        # 1. Mode check
        if self.mutation_mode == MutationMode.DISABLED:
            reason = "mutations_disabled"
            logger.warning(f"BLOCKED: {operation} for {belief_id} - {reason}")
            self._record_block(reason)
            return False, reason

        # 2. Namespace protection (core.* is sacred)
        for protected in self.PROTECTED_NAMESPACES:
            if belief_id.startswith(protected) and updated_by != "migration":
                reason = f"protected_namespace:{protected}"
                logger.warning(f"BLOCKED: {operation} for {belief_id} - {reason}")
                self._record_block(reason)
                return False, reason

        # 3. Mode-specific operation checks
        is_create = "create" in operation.lower()
        is_reinforce = "reinforce" in operation.lower()
        is_update = "update" in operation.lower() and not is_reinforce

        if self.mutation_mode == MutationMode.REINFORCE_ONLY:
            if is_create or is_update:
                reason = "mode_reinforce_only"
                logger.info(f"BLOCKED: {operation} for {belief_id} - {reason}")
                self._record_block(reason)
                return False, reason

        # 4. Namespace allowlist for creates
        if is_create:
            allowed_ns = any(belief_id.startswith(ns) for ns in self.ALLOWED_CREATE_NAMESPACES)
            if not allowed_ns:
                reason = f"create_namespace_not_allowed"
                logger.warning(f"BLOCKED: {operation} for {belief_id} - {reason}")
                self._record_block(reason)
                return False, reason

        # 5. Rate limits (if scan context exists)
        if self._current_scan:
            # Idempotency check
            idempotency_key = (operation, belief_id)
            if idempotency_key in self._current_scan.applied_keys:
                self._scan_telemetry["writes_idempotent_skip"] += 1
                return False, "idempotent_skip"

            # Per-scan rate limits
            if is_create and self._current_scan.creates >= self.MAX_CREATES_PER_SCAN:
                reason = "rate_limit_creates"
                logger.warning(f"BLOCKED: {operation} for {belief_id} - {reason} ({self._current_scan.creates}/{self.MAX_CREATES_PER_SCAN})")
                self._record_block(reason)
                return False, reason

            if is_reinforce and self._current_scan.reinforcements >= self.MAX_REINFORCEMENTS_PER_SCAN:
                reason = "rate_limit_reinforcements"
                logger.info(f"BLOCKED: {operation} for {belief_id} - {reason}")
                self._record_block(reason)
                return False, reason

            if is_update and self._current_scan.updates >= self.MAX_UPDATES_PER_SCAN:
                reason = "rate_limit_updates"
                logger.info(f"BLOCKED: {operation} for {belief_id} - {reason}")
                self._record_block(reason)
                return False, reason

            # Per-category rate limit for creates
            if is_create and category:
                cat_count = self._category_creates.get(category, 0)
                if cat_count >= self.MAX_CREATES_PER_CATEGORY:
                    reason = f"rate_limit_category:{category}"
                    logger.info(f"BLOCKED: {operation} for {belief_id} - {reason} ({cat_count}/{self.MAX_CREATES_PER_CATEGORY})")
                    self._record_block(reason)
                    return False, reason

        # 6. Statement content guardrails (for creates)
        if is_create and statement:
            # Token length check
            tokens = statement.lower().split()
            if len(tokens) < self.MIN_STATEMENT_TOKENS:
                reason = "statement_too_short"
                logger.debug(f"BLOCKED: {operation} for {belief_id} - {reason}")
                self._record_block(reason)
                return False, reason

            if len(tokens) > self.MAX_STATEMENT_TOKENS:
                reason = "statement_too_long"
                logger.debug(f"BLOCKED: {operation} for {belief_id} - {reason}")
                self._record_block(reason)
                return False, reason

            # Blocked pattern check
            statement_lower = statement.lower()
            for pattern in self.BLOCKED_PATTERNS:
                if pattern in statement_lower:
                    reason = f"blocked_pattern:{pattern}"
                    logger.debug(f"BLOCKED: {operation} for {belief_id} - {reason}")
                    self._record_block(reason)
                    return False, reason

        return True, ""

    def _record_write_success(self, operation: str, belief_id: str, category: str = "") -> None:
        """Record successful write and update counters."""
        self._scan_telemetry["writes_applied"] += 1

        if self._current_scan:
            # Record for idempotency
            self._current_scan.applied_keys.add((operation, belief_id))

            # Update counters
            if "create" in operation.lower():
                self._current_scan.creates += 1
                if category:
                    self._category_creates[category] = self._category_creates.get(category, 0) + 1
            elif "reinforce" in operation.lower():
                self._current_scan.reinforcements += 1
            else:
                self._current_scan.updates += 1

    def _check_belief_mutable(self, belief_id: str, belief_data: dict, operation: str) -> bool:
        """
        Check if a specific belief can be mutated based on its properties.

        Blocks mutation if:
        - immutable flag is True
        - stability >= STABILITY_THRESHOLD (0.95)
        - is_core flag is True

        Returns False and logs if blocked.
        """
        # Check immutable flag
        if belief_data.get("immutable", False):
            logger.error(f"BLOCKED: Cannot {operation} immutable belief {belief_id}")
            return False

        # Check stability (metadata or top-level)
        stability = belief_data.get("stability") or belief_data.get("metadata", {}).get("stability", 0.0)
        if stability >= self.STABILITY_THRESHOLD:
            logger.error(f"BLOCKED: Cannot {operation} high-stability belief {belief_id} (stability={stability})")
            return False

        # Check is_core flag
        if belief_data.get("is_core", False) or belief_data.get("metadata", {}).get("is_core", False):
            logger.error(f"BLOCKED: Cannot {operation} core belief {belief_id}")
            return False

        return True

    def _initialize_empty_store(self):
        """Initialize empty belief store."""
        with self._lock:
            with open(self.current_file, "w") as f:
                json.dump({}, f, indent=2)
            with open(self.index_file, "w") as f:
                json.dump({}, f, indent=2)
            logger.info("Initialized empty belief store")

    def _day_stamp(self, ts: Optional[float] = None) -> str:
        """Get YYYYMMDD stamp."""
        t = datetime.utcfromtimestamp(ts or time.time())
        return t.strftime("%Y%m%d")

    def _log_file_for_day(self, day: str) -> Path:
        """Get log file path for a given day."""
        return self.beliefs_dir / f"log-{day}.ndjson.gz"

    def get_current(self, belief_ids: Optional[List[str]] = None) -> Dict[str, BeliefVersion]:
        """Get current beliefs.

        Args:
            belief_ids: Optional list of belief IDs to fetch (all if None)

        Returns:
            Dict mapping belief_id to BeliefVersion
        """
        with self._lock:
            with open(self.current_file, "r") as f:
                data = json.load(f)

            beliefs = {}
            for bid, belief_data in data.items():
                if belief_ids is None or bid in belief_ids:
                    beliefs[bid] = BeliefVersion(**belief_data)

            return beliefs

    def get_history(self, belief_id: str, limit: Optional[int] = None) -> List[BeliefDelta]:
        """Get history of deltas for a belief.

        Args:
            belief_id: Belief ID
            limit: Maximum number of deltas to return (most recent first)

        Returns:
            List of BeliefDelta in reverse chronological order
        """
        deltas = []

        # Scan all log files (newest first)
        log_files = sorted(self.beliefs_dir.glob("log-*.ndjson.gz"), reverse=True)

        for log_file in log_files:
            try:
                with gzip.open(log_file, "rt") as f:
                    for line in f:
                        delta_data = json.loads(line)
                        if delta_data.get("belief_id") == belief_id:
                            deltas.append(BeliefDelta(**delta_data))
                            if limit and len(deltas) >= limit:
                                return deltas
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")

        return deltas

    def apply_delta(
        self,
        belief_id: str,
        from_ver: int,
        op: DeltaOp,
        confidence_delta: float = 0.0,
        evidence_refs_added: Optional[List[str]] = None,
        evidence_refs_removed: Optional[List[str]] = None,
        state_change: Optional[str] = None,
        updated_by: str = "system",
        reason: str = "",
    ) -> bool:
        """Apply a delta to a belief with optimistic locking.

        Args:
            belief_id: Belief ID
            from_ver: Expected current version (for optimistic lock)
            op: Operation type
            confidence_delta: Change in confidence
            evidence_refs_added: Evidence references to add
            evidence_refs_removed: Evidence references to remove
            state_change: State transition (e.g., "tentative->asserted")
            updated_by: Actor applying this delta
            reason: Explanation for this change

        Returns:
            True if delta applied successfully, False if version mismatch or blocked

        Raises:
            ValueError: If delta violates guardrails
        """
        # SAFETY: Check guardrails (mode, namespace, rate limits)
        op_name = f"apply_delta({op.value})" if hasattr(op, 'value') else f"apply_delta({op})"
        allowed, block_reason = self._check_mutation_allowed(
            belief_id=belief_id,
            operation=op_name,
            updated_by=updated_by,
        )
        if not allowed:
            # Raise ValueError so callers can distinguish blocked from version mismatch
            # Treat idempotent_skip as success (already in desired state)
            if block_reason == "idempotent_skip":
                return True
            raise ValueError(f"blocked:{block_reason}")

        evidence_refs_added = evidence_refs_added or []
        evidence_refs_removed = evidence_refs_removed or []

        # Validate confidence delta
        if abs(confidence_delta) > 0:
            if abs(confidence_delta) < self.MIN_CONFIDENCE_STEP:
                raise ValueError(f"Confidence step too small: {confidence_delta}")
            if abs(confidence_delta) > self.MAX_CONFIDENCE_STEP:
                raise ValueError(f"Confidence step too large: {confidence_delta}")

        with self._lock:
            # Load current state
            with open(self.current_file, "r") as f:
                current = json.load(f)

            # Check if belief exists
            if belief_id not in current:
                if op != DeltaOp.CREATE:
                    raise ValueError(f"Belief {belief_id} does not exist")
                # For CREATE, from_ver should be 0
                if from_ver != 0:
                    raise ValueError(f"CREATE requires from_ver=0, got {from_ver}")
            else:
                # SAFETY: Check if belief is mutable (stability, immutable, is_core)
                if not self._check_belief_mutable(belief_id, current[belief_id], f"apply_delta({op})"):
                    return False

                # Check version (optimistic lock)
                current_belief = BeliefVersion(**current[belief_id])
                if current_belief.ver != from_ver:
                    logger.warning(
                        f"Version mismatch for {belief_id}: expected {from_ver}, got {current_belief.ver}"
                    )
                    return False

            # Apply delta
            to_ver = from_ver + 1

            if op == DeltaOp.CREATE:
                # Create new belief (requires external data)
                raise NotImplementedError("Use create_belief() instead")

            # Get current belief
            current_belief = BeliefVersion(**current[belief_id])

            # Apply changes
            new_confidence = max(0.0, min(1.0, current_belief.confidence + confidence_delta))
            new_evidence = list(set(current_belief.evidence_refs) - set(evidence_refs_removed))
            new_evidence.extend(evidence_refs_added)
            new_evidence = list(set(new_evidence))  # Deduplicate

            new_state = current_belief.state
            if state_change:
                # Validate state transition
                if "->" in state_change:
                    from_state, to_state = state_change.split("->")
                    if from_state != current_belief.state:
                        raise ValueError(f"Invalid state transition: current={current_belief.state}, expected={from_state}")

                    # Check evidence requirements for tentative->asserted
                    if from_state == "tentative" and to_state == "asserted":
                        if len(new_evidence) < self.MIN_EVIDENCE_FOR_ASSERTED:
                            raise ValueError(
                                f"Cannot transition to asserted with only {len(new_evidence)} evidence refs (need {self.MIN_EVIDENCE_FOR_ASSERTED})"
                            )

                    new_state = BeliefState(to_state)

            # Create updated belief
            updated_belief = BeliefVersion(
                belief_id=belief_id,
                ver=to_ver,
                statement=current_belief.statement,
                state=new_state,
                confidence=new_confidence,
                evidence_refs=new_evidence,
                updated_by=updated_by,
                ts=time.time(),
                belief_type=current_belief.belief_type,
                immutable=current_belief.immutable,
                rationale=current_belief.rationale,
                metadata=current_belief.metadata,
                hash="",  # Will be computed
            )

            # Compute hash
            updated_belief.hash = updated_belief.compute_hash()

            # Update current
            current[belief_id] = asdict(updated_belief)

            # Update index
            with open(self.index_file, "r") as f:
                index = json.load(f)
            index[belief_id] = to_ver

            # Write atomically
            with open(self.current_file, "w") as f:
                json.dump(current, f, indent=2)
            with open(self.index_file, "w") as f:
                json.dump(index, f, indent=2)

            # Append delta to log
            delta = BeliefDelta(
                belief_id=belief_id,
                from_ver=from_ver,
                to_ver=to_ver,
                op=op,
                confidence_delta=confidence_delta,
                evidence_refs_added=evidence_refs_added,
                evidence_refs_removed=evidence_refs_removed,
                state_change=state_change,
                updated_by=updated_by,
                ts=time.time(),
                reason=reason,
            )

            self._append_delta_to_log(delta)

            # Log to identity ledger
            try:
                belief_versioned_event(
                    belief_id=belief_id,
                    from_ver=from_ver,
                    to_ver=to_ver,
                    reason_changed=reason,
                    confidence=new_confidence,
                    cause=f"delta_{op.value}",
                    evidence_refs=new_evidence,
                )
            except Exception as e:
                logger.error(f"Failed to log belief version to ledger: {e}")

            logger.info(
                f"Applied delta to {belief_id}: ver {from_ver}->{to_ver}, op={op}, confidence_delta={confidence_delta}"
            )

            # Record success for telemetry
            self._record_write_success(op_name, belief_id)

            return True

    def create_belief(
        self,
        belief_id: str,
        statement: str,
        state: BeliefState,
        confidence: float,
        evidence_refs: List[str],
        belief_type: str,
        immutable: bool,
        rationale: str,
        metadata: Optional[Dict[str, Any]] = None,
        updated_by: str = "system",
    ) -> bool:
        """Create a new belief.

        Args:
            belief_id: Unique belief ID
            statement: First-person belief statement
            state: Initial state
            confidence: Initial confidence [0.0, 1.0]
            evidence_refs: Initial evidence references
            belief_type: Type of belief
            immutable: Whether this is a core belief
            rationale: Why this belief exists
            metadata: Additional metadata
            updated_by: Actor creating this belief

        Returns:
            True if created successfully, False if already exists or blocked
        """
        # Extract category from belief_id (e.g., "auto.emotional.xxx" -> "emotional")
        category = ""
        parts = belief_id.split(".")
        if len(parts) >= 2:
            category = parts[1]  # auto.emotional.xxx -> emotional

        # SAFETY: Check all guardrails (mode, namespace, rate limits, content)
        allowed, block_reason = self._check_mutation_allowed(
            belief_id=belief_id,
            operation="create_belief",
            statement=statement,
            category=category,
            updated_by=updated_by,
        )
        if not allowed:
            return False

        metadata = metadata or {}

        with self._lock:
            # Check if belief already exists
            with open(self.current_file, "r") as f:
                current = json.load(f)

            if belief_id in current:
                logger.info(f"Belief {belief_id} already exists - skipping create")
                return False

            # Create belief at ver=1
            belief = BeliefVersion(
                belief_id=belief_id,
                ver=1,
                statement=statement,
                state=state,
                confidence=confidence,
                evidence_refs=evidence_refs,
                updated_by=updated_by,
                ts=time.time(),
                belief_type=belief_type,
                immutable=immutable,
                rationale=rationale,
                metadata=metadata,
                hash="",
            )

            # Compute hash
            belief.hash = belief.compute_hash()

            # Add to current
            current[belief_id] = asdict(belief)

            # Update index
            with open(self.index_file, "r") as f:
                index = json.load(f)
            index[belief_id] = 1

            # Write atomically
            with open(self.current_file, "w") as f:
                json.dump(current, f, indent=2)
            with open(self.index_file, "w") as f:
                json.dump(index, f, indent=2)

            # Append delta
            delta = BeliefDelta(
                belief_id=belief_id,
                from_ver=0,
                to_ver=1,
                op=DeltaOp.CREATE,
                confidence_delta=confidence,
                evidence_refs_added=evidence_refs,
                evidence_refs_removed=[],
                state_change=None,
                updated_by=updated_by,
                ts=time.time(),
                reason=f"Created: {statement[:50]}",
            )

            self._append_delta_to_log(delta)

            # Log to identity ledger
            try:
                belief_versioned_event(
                    belief_id=belief_id,
                    from_ver=0,
                    to_ver=1,
                    reason_changed=f"Created: {statement[:50]}",
                    confidence=confidence,
                    cause=f"create_{updated_by}",
                    evidence_refs=evidence_refs,
                )
            except Exception as e:
                logger.error(f"Failed to log belief creation to ledger: {e}")

            logger.info(f"Created belief {belief_id} at ver=1")

            # Record success for telemetry
            self._record_write_success("create_belief", belief_id, category)

            return True

    def deprecate_belief(
        self,
        belief_id: str,
        from_ver: int,
        replacement_id: Optional[str] = None,
        updated_by: str = "system",
        reason: str = "",
    ) -> bool:
        """Deprecate a belief (soft delete with pointer to replacement).

        Args:
            belief_id: Belief to deprecate
            from_ver: Expected current version
            replacement_id: Optional ID of replacement belief
            updated_by: Actor deprecating this belief
            reason: Explanation

        Returns:
            True if deprecated successfully
        """
        # SAFETY: Check kill switch
        if not self._check_mutation_allowed(belief_id, "deprecate_belief"):
            return False

        metadata_update = {"deprecated": True}
        if replacement_id:
            metadata_update["replacement_id"] = replacement_id

        # Get current belief to determine its state for proper transition
        with self._lock:
            with open(self.current_file, "r") as f:
                current = json.load(f)

            if belief_id not in current:
                logger.warning(f"Cannot deprecate non-existent belief: {belief_id}")
                return False

            # SAFETY: Check if belief is mutable (stability, immutable, is_core)
            if not self._check_belief_mutable(belief_id, current[belief_id], "deprecate_belief"):
                return False

            current_belief = BeliefVersion(**current[belief_id])
            current_state = current_belief.state

        # Build appropriate state transition based on current state
        state_change = f"{current_state}->deprecated"

        return self.apply_delta(
            belief_id=belief_id,
            from_ver=from_ver,
            op=DeltaOp.DEPRECATE,
            state_change=state_change,
            updated_by=updated_by,
            reason=reason,
        )

    def _append_delta_to_log(self, delta: BeliefDelta):
        """Append delta to daily log file."""
        day = self._day_stamp(delta.ts)
        log_file = self._log_file_for_day(day)

        with gzip.open(log_file, "at") as f:
            f.write(json.dumps(asdict(delta)) + "\n")

    def verify_integrity(self) -> Dict[str, bool]:
        """Verify integrity of current beliefs.

        Returns:
            Dict with verification results
        """
        results = {"hash_valid": True, "index_consistent": True}

        with self._lock:
            # Load current and index
            with open(self.current_file, "r") as f:
                current = json.load(f)
            with open(self.index_file, "r") as f:
                index = json.load(f)

            # Check hashes
            for bid, belief_data in current.items():
                belief = BeliefVersion(**belief_data)
                if not belief.verify_hash():
                    logger.error(f"Hash verification failed for {bid}")
                    results["hash_valid"] = False

            # Check index consistency
            for bid, ver in index.items():
                if bid not in current:
                    logger.error(f"Index references missing belief: {bid}")
                    results["index_consistent"] = False
                elif current[bid]["ver"] != ver:
                    logger.error(f"Index version mismatch for {bid}: index={ver}, current={current[bid]['ver']}")
                    results["index_consistent"] = False

        return results


def create_belief_store(data_dir: Path) -> BeliefStore:
    """Factory function to create belief store.

    Args:
        data_dir: Base directory for storage

    Returns:
        BeliefStore instance
    """
    return BeliefStore(data_dir)
