"""
Memory Pruner - Archives or deletes decayed memories to manage storage.

Part of Memory Consolidation Layer (Phase 4).

Pruning Rules:
- PRUNE if decay_factor < 0.1 AND emotional_salience < 0.3 AND not referenced by belief
- KEEP if emotional_salience > 0.5 (emotional memories resist pruning)
- KEEP if referenced by active belief (evidence)
- ARCHIVE if part of consolidated narrative (move to cold storage)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import json

from sqlmodel import Session as DBSession, select

from src.memory.raw_store import RawStore
from src.memory.models import (
    Experience,
    ExperienceModel,
    ExperienceType,
    MemoryDecayMetrics,
    experience_to_model,
)

logger = logging.getLogger(__name__)


@dataclass
class PrunerConfig:
    """Configuration for memory pruning."""
    # Decay thresholds
    decay_threshold: float = 0.1  # Prune if decay < this
    salience_preserve_threshold: float = 0.5  # Keep if salience > this

    # Age requirements
    min_age_days_for_prune: int = 30  # Don't prune recent memories

    # Access requirements
    min_days_since_access: int = 30  # Must not be accessed recently

    # Self-definition pruning settings
    self_def_prune_after_days: int = 30  # Days without reinforcement before pruning
    self_def_protect_core: bool = True  # Never prune core stability traits

    # Archive settings
    archive_dir: str = "data/memory_archive"
    archive_consolidated: bool = True  # Archive instead of delete consolidated

    # Safety limits
    max_prune_per_run: int = 100  # Safety limit per run


@dataclass
class PruneCandidate:
    """A memory candidate for pruning."""
    experience_id: str
    experience_type: str
    decay_factor: float
    emotional_salience: float
    access_count: int
    last_accessed: Optional[datetime]
    age_days: float
    consolidated: bool
    decision: str  # "keep", "archive", "delete"
    reason: str


class MemoryPruner:
    """
    Manages memory pruning to control storage growth.

    Implements careful pruning rules that protect:
    - Emotionally significant memories
    - Memories referenced by beliefs
    - Consolidated narratives (archived, not deleted)
    """

    def __init__(
        self,
        raw_store: RawStore,
        belief_store=None,
        config: Optional[PrunerConfig] = None,
    ):
        """Initialize memory pruner.

        Args:
            raw_store: Experience raw store
            belief_store: Optional belief store for reference checking
            config: Pruning configuration
        """
        self.raw_store = raw_store
        self.belief_store = belief_store
        self.config = config or PrunerConfig()

        # Ensure archive directory exists
        self.archive_dir = Path(self.config.archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"MemoryPruner initialized (archive={self.archive_dir})")

    def get_referenced_experience_ids(self) -> Set[str]:
        """Get experience IDs referenced by active beliefs."""
        if not self.belief_store:
            return set()

        referenced = set()

        try:
            all_beliefs = self.belief_store.get_current()
            for belief in all_beliefs.values():
                evidence_refs = getattr(belief, 'evidence_refs', [])
                if evidence_refs:
                    referenced.update(evidence_refs)
        except Exception as e:
            logger.warning(f"Could not get belief references: {e}")

        return referenced

    def identify_prune_candidates(self) -> List[PruneCandidate]:
        """Identify experiences that may be candidates for pruning."""
        candidates = []
        now = datetime.now(timezone.utc)
        cutoff_date = now - timedelta(days=self.config.min_age_days_for_prune)

        # Get referenced experience IDs (protected)
        referenced_ids = self.get_referenced_experience_ids()

        with DBSession(self.raw_store.engine) as db:
            # Get all decay metrics
            metrics_stmt = select(MemoryDecayMetrics).where(
                MemoryDecayMetrics.decay_factor < self.config.decay_threshold
            )
            metrics = db.exec(metrics_stmt).all()

            for metric in metrics:
                # Get the experience
                exp = db.get(Experience, metric.experience_id)
                if not exp:
                    continue

                # Calculate age
                exp_created = exp.created_at
                if exp_created.tzinfo is None:
                    exp_created = exp_created.replace(tzinfo=timezone.utc)
                age_days = (now - exp_created).days

                # Skip recent experiences
                if age_days < self.config.min_age_days_for_prune:
                    continue

                # Determine decision
                decision, reason = self._evaluate_candidate(
                    exp,
                    metric,
                    referenced_ids,
                    age_days,
                )

                candidates.append(PruneCandidate(
                    experience_id=exp.id,
                    experience_type=exp.type,
                    decay_factor=metric.decay_factor,
                    emotional_salience=metric.emotional_salience,
                    access_count=metric.access_count,
                    last_accessed=metric.last_accessed,
                    age_days=age_days,
                    consolidated=exp.consolidated,
                    decision=decision,
                    reason=reason,
                ))

        logger.info(f"Identified {len(candidates)} prune candidates")
        return candidates

    def identify_self_definition_prune_candidates(self) -> List[PruneCandidate]:
        """
        Identify self-definition experiences that should be pruned.

        Pruning strategy based on "reinforcement by repetition":
        - Count how many times each statement text appears
        - Statements appearing multiple times are "reinforced" - keep the oldest, prune duplicates
        - Single-occurrence statements older than threshold are pruned (never reinforced)
        - Core stability traits are protected (if self_def_protect_core=True)
        """
        candidates = []
        now = datetime.now(timezone.utc)
        cutoff_date = now - timedelta(days=self.config.self_def_prune_after_days)

        with DBSession(self.raw_store.engine) as db:
            # Get all SELF_DEFINITION experiences
            stmt = select(Experience).where(
                Experience.type == ExperienceType.SELF_DEFINITION.value
            )
            self_defs = db.exec(stmt).all()

            # Build statement -> list of (exp_id, created_at, exp) mapping
            from collections import defaultdict
            statement_groups: dict[str, list] = defaultdict(list)

            for exp in self_defs:
                content = exp.content or {}
                text = content.get("text", "").strip()
                if text:
                    statement_groups[text].append(exp)

            # Process each statement group
            for statement_text, exps in statement_groups.items():
                # Sort by created_at (oldest first)
                exps_sorted = sorted(exps, key=lambda e: e.created_at)

                # Check stability on first occurrence
                first_exp = exps_sorted[0]
                content = first_exp.content or {}
                structured = content.get("structured", {})
                stability = structured.get("stability", "surface")

                # Rule 1: Protect core traits
                if self.config.self_def_protect_core and stability == "core":
                    logger.debug(f"Protected core trait: {statement_text[:50]}")
                    continue

                if len(exps_sorted) >= 2:
                    # Statement has been "reinforced" (appears multiple times)
                    # Keep the OLDEST one (canonical), prune all duplicates
                    for exp in exps_sorted[1:]:  # Skip the oldest, prune the rest
                        exp_created = exp.created_at
                        if exp_created.tzinfo is None:
                            exp_created = exp_created.replace(tzinfo=timezone.utc)
                        age_days = (now - exp_created).days

                        candidates.append(PruneCandidate(
                            experience_id=exp.id,
                            experience_type=exp.type,
                            decay_factor=0.5,
                            emotional_salience=0.0,
                            access_count=0,
                            last_accessed=None,
                            age_days=age_days,
                            consolidated=exp.consolidated,
                            decision="delete",
                            reason=f"duplicate_of_reinforced_statement_{len(exps_sorted)}_occurrences",
                        ))
                else:
                    # Single occurrence - check if old enough to prune
                    exp = exps_sorted[0]
                    exp_created = exp.created_at
                    if exp_created.tzinfo is None:
                        exp_created = exp_created.replace(tzinfo=timezone.utc)
                    age_days = (now - exp_created).days

                    if exp_created < cutoff_date:
                        candidates.append(PruneCandidate(
                            experience_id=exp.id,
                            experience_type=exp.type,
                            decay_factor=0.5,
                            emotional_salience=0.0,
                            access_count=0,
                            last_accessed=None,
                            age_days=age_days,
                            consolidated=exp.consolidated,
                            decision="delete",
                            reason=f"single_occurrence_older_than_{self.config.self_def_prune_after_days}_days",
                        ))

        logger.info(f"Identified {len(candidates)} self-definition prune candidates")
        return candidates

    def _evaluate_candidate(
        self,
        exp: Experience,
        metric: MemoryDecayMetrics,
        referenced_ids: Set[str],
        age_days: float,
    ) -> tuple[str, str]:
        """Evaluate a candidate and determine action."""
        # Rule 1: Protected by belief reference
        if exp.id in referenced_ids:
            return "keep", "referenced_by_belief"

        # Rule 2: High emotional salience
        if metric.emotional_salience > self.config.salience_preserve_threshold:
            return "keep", "high_emotional_salience"

        # Rule 3: Consolidated experiences get archived, not deleted
        if exp.consolidated and self.config.archive_consolidated:
            return "archive", "consolidated_archive"

        # Rule 4: NARRATIVE and LEARNING_PATTERN are never deleted
        if exp.type in (ExperienceType.NARRATIVE.value, ExperienceType.LEARNING_PATTERN.value):
            return "archive", "protected_type"

        # Rule 5: Check access recency
        if metric.last_accessed:
            last_access = metric.last_accessed
            if last_access.tzinfo is None:
                last_access = last_access.replace(tzinfo=timezone.utc)
            days_since_access = (datetime.now(timezone.utc) - last_access).days
            if days_since_access < self.config.min_days_since_access:
                return "keep", "recently_accessed"

        # Default: safe to delete
        return "delete", "decay_threshold_met"

    async def prune(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute memory pruning.

        Args:
            dry_run: If True, only report what would be pruned

        Returns:
            Summary of pruning actions
        """
        # Get general memory candidates
        candidates = self.identify_prune_candidates()

        # Get self-definition candidates (based on last_reinforced, not decay metrics)
        self_def_candidates = self.identify_self_definition_prune_candidates()

        results = {
            "candidates_found": len(candidates),
            "self_def_candidates_found": len(self_def_candidates),
            "kept": 0,
            "archived": 0,
            "deleted": 0,
            "self_defs_deleted": 0,
            "dry_run": dry_run,
            "details": [],
        }

        # Group general candidates by decision
        to_archive = [c for c in candidates if c.decision == "archive"]
        to_delete = [c for c in candidates if c.decision == "delete"]
        to_keep = [c for c in candidates if c.decision == "keep"]

        results["kept"] = len(to_keep)

        # Apply safety limit to general candidates
        to_archive = to_archive[:self.config.max_prune_per_run]
        to_delete = to_delete[:self.config.max_prune_per_run - len(to_archive)]

        # Apply safety limit to self-definition candidates (separate budget)
        self_def_to_delete = self_def_candidates[:self.config.max_prune_per_run]

        if not dry_run:
            # Archive general candidates
            for candidate in to_archive:
                if self._archive_experience(candidate.experience_id):
                    results["archived"] += 1
                    results["details"].append({
                        "id": candidate.experience_id,
                        "action": "archived",
                        "reason": candidate.reason,
                    })

            # Delete general candidates
            for candidate in to_delete:
                if self._delete_experience(candidate.experience_id):
                    results["deleted"] += 1
                    results["details"].append({
                        "id": candidate.experience_id,
                        "action": "deleted",
                        "reason": candidate.reason,
                    })

            # Delete self-definition candidates
            for candidate in self_def_to_delete:
                if self._delete_experience(candidate.experience_id):
                    results["self_defs_deleted"] += 1
                    results["details"].append({
                        "id": candidate.experience_id,
                        "type": "self_definition",
                        "action": "deleted",
                        "reason": candidate.reason,
                    })
        else:
            # Dry run - just report
            results["would_archive"] = len(to_archive)
            results["would_delete"] = len(to_delete)
            results["would_delete_self_defs"] = len(self_def_to_delete)
            results["details"] = [
                {"id": c.experience_id, "would": c.decision, "reason": c.reason}
                for c in to_archive + to_delete
            ][:20]  # Limit details
            results["self_def_details"] = [
                {"id": c.experience_id, "would": "delete", "reason": c.reason}
                for c in self_def_to_delete
            ][:10]  # Limit self-def details

        logger.info(f"Pruning complete: {results}")
        return results

    def _archive_experience(self, experience_id: str) -> bool:
        """Archive an experience to cold storage."""
        try:
            with DBSession(self.raw_store.engine) as db:
                exp = db.get(Experience, experience_id)
                if not exp:
                    return False

                # Convert to dict for JSON storage
                exp_model = experience_to_model(exp)
                exp_dict = exp_model.model_dump(mode='json')

                # Write to archive file (one file per day)
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                archive_file = self.archive_dir / f"archive_{today}.jsonl"

                with open(archive_file, "a") as f:
                    f.write(json.dumps(exp_dict) + "\n")

                # Soft-delete from main store (tombstone)
                # For now, just mark as archived via special flag
                # TODO: Implement proper tombstone
                logger.debug(f"Archived experience {experience_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to archive {experience_id}: {e}")
            return False

    def _delete_experience(self, experience_id: str) -> bool:
        """Delete an experience (after archiving metadata)."""
        try:
            # First archive metadata before deletion
            self._archive_experience(experience_id)

            with DBSession(self.raw_store.engine) as db:
                exp = db.get(Experience, experience_id)
                if exp:
                    db.delete(exp)
                    db.commit()
                    logger.debug(f"Deleted experience {experience_id}")
                    return True
                return False

        except Exception as e:
            logger.error(f"Failed to delete {experience_id}: {e}")
            return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get current storage statistics."""
        with DBSession(self.raw_store.engine) as db:
            # Count by type
            all_exp = db.exec(select(Experience)).all()
            type_counts = {}
            for exp in all_exp:
                exp_type = exp.type
                type_counts[exp_type] = type_counts.get(exp_type, 0) + 1

            # Count consolidated
            consolidated = sum(1 for exp in all_exp if exp.consolidated)

            # Get decay metrics summary
            metrics = db.exec(select(MemoryDecayMetrics)).all()
            if metrics:
                avg_decay = sum(m.decay_factor for m in metrics) / len(metrics)
                low_decay_count = sum(1 for m in metrics if m.decay_factor < self.config.decay_threshold)
            else:
                avg_decay = 1.0
                low_decay_count = 0

        return {
            "total_experiences": len(all_exp),
            "by_type": type_counts,
            "consolidated": consolidated,
            "unconsolidated": len(all_exp) - consolidated,
            "average_decay_factor": round(avg_decay, 3),
            "low_decay_count": low_decay_count,
            "archive_dir": str(self.archive_dir),
        }


def create_memory_pruner(
    raw_store: RawStore,
    belief_store=None,
    config: Optional[PrunerConfig] = None,
) -> MemoryPruner:
    """Factory function to create MemoryPruner."""
    return MemoryPruner(
        raw_store=raw_store,
        belief_store=belief_store,
        config=config,
    )
