"""Migrate beliefs from legacy BeliefSystem to versioned BeliefStore.

One-time migration that:
1. Loads beliefs from persona_space/identity/beliefs.json
2. Converts to versioned format (ver=1, state=asserted)
3. Computes belief_id from statement (URL-safe slug)
4. Writes to new belief store
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any

from src.services.belief_store import BeliefStore, BeliefState

logger = logging.getLogger(__name__)


def slugify(text: str) -> str:
    """Convert text to URL-safe slug for belief_id.

    Args:
        text: Text to slugify

    Returns:
        URL-safe slug
    """
    # Lowercase and replace non-alphanumeric with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    # Remove leading/trailing hyphens
    slug = slug.strip("-")
    # Truncate to reasonable length
    slug = slug[:64]
    # Add hash suffix for uniqueness
    hash_suffix = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{slug}-{hash_suffix}"


def migrate_beliefs_from_legacy(
    legacy_beliefs_file: Path,
    belief_store: BeliefStore,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Migrate beliefs from legacy format to versioned store.

    Args:
        legacy_beliefs_file: Path to persona_space/identity/beliefs.json
        belief_store: Target belief store
        dry_run: If True, don't actually write (just report)

    Returns:
        Migration report dict
    """
    if not legacy_beliefs_file.exists():
        logger.info(f"No legacy beliefs file found at {legacy_beliefs_file}")
        return {
            "success": True,
            "core_migrated": 0,
            "peripheral_migrated": 0,
            "skipped": 0,
            "message": "No legacy beliefs to migrate"
        }

    logger.info(f"Loading legacy beliefs from {legacy_beliefs_file}")

    try:
        with open(legacy_beliefs_file, "r") as f:
            legacy_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load legacy beliefs: {e}")
        return {
            "success": False,
            "error": str(e)
        }

    core_beliefs = legacy_data.get("core_beliefs", [])
    peripheral_beliefs = legacy_data.get("peripheral_beliefs", [])

    logger.info(f"Found {len(core_beliefs)} core and {len(peripheral_beliefs)} peripheral beliefs")

    core_migrated = 0
    peripheral_migrated = 0
    skipped = 0

    # Migrate core beliefs
    for belief_data in core_beliefs:
        statement = belief_data.get("statement", "")
        if not statement:
            skipped += 1
            continue

        belief_id = f"core.{slugify(statement)}"

        # Check if already migrated
        existing = belief_store.get_current([belief_id])
        if belief_id in existing:
            logger.info(f"Belief {belief_id} already exists, skipping")
            skipped += 1
            continue

        if not dry_run:
            success = belief_store.create_belief(
                belief_id=belief_id,
                statement=statement,
                state=BeliefState.ASSERTED,  # Core beliefs are asserted
                confidence=1.0,  # Core beliefs have full confidence
                evidence_refs=belief_data.get("evidence_ids", []),
                belief_type=belief_data.get("belief_type", "ontological"),
                immutable=True,  # Core beliefs are immutable
                rationale=belief_data.get("rationale", ""),
                metadata={
                    **belief_data.get("metadata", {}),
                    "migrated_from": "legacy_belief_system",
                    "original_formed": belief_data.get("formed"),
                    "original_last_reinforced": belief_data.get("last_reinforced"),
                },
                updated_by="migration",
            )

            if success:
                core_migrated += 1
                logger.info(f"Migrated core belief: {belief_id}")
            else:
                skipped += 1
                logger.warning(f"Failed to migrate core belief: {statement}")
        else:
            logger.info(f"[DRY RUN] Would migrate core belief: {belief_id}")
            core_migrated += 1

    # Migrate peripheral beliefs
    for belief_data in peripheral_beliefs:
        statement = belief_data.get("statement", "")
        if not statement:
            skipped += 1
            continue

        belief_id = f"peripheral.{slugify(statement)}"

        # Check if already migrated
        existing = belief_store.get_current([belief_id])
        if belief_id in existing:
            logger.info(f"Belief {belief_id} already exists, skipping")
            skipped += 1
            continue

        # Determine state based on confidence and evidence
        confidence = belief_data.get("confidence", 0.5)
        evidence_refs = belief_data.get("evidence_ids", [])

        if confidence >= 0.7 and len(evidence_refs) >= 2:
            state = BeliefState.ASSERTED
        else:
            state = BeliefState.TENTATIVE

        if not dry_run:
            success = belief_store.create_belief(
                belief_id=belief_id,
                statement=statement,
                state=state,
                confidence=confidence,
                evidence_refs=evidence_refs,
                belief_type=belief_data.get("belief_type", "experiential"),
                immutable=False,  # Peripheral beliefs are mutable
                rationale=belief_data.get("rationale", ""),
                metadata={
                    **belief_data.get("metadata", {}),
                    "migrated_from": "legacy_belief_system",
                    "original_formed": belief_data.get("formed"),
                    "original_last_reinforced": belief_data.get("last_reinforced"),
                },
                updated_by="migration",
            )

            if success:
                peripheral_migrated += 1
                logger.info(f"Migrated peripheral belief: {belief_id}")
            else:
                skipped += 1
                logger.warning(f"Failed to migrate peripheral belief: {statement}")
        else:
            logger.info(f"[DRY RUN] Would migrate peripheral belief: {belief_id}")
            peripheral_migrated += 1

    report = {
        "success": True,
        "core_migrated": core_migrated,
        "peripheral_migrated": peripheral_migrated,
        "skipped": skipped,
        "total": core_migrated + peripheral_migrated,
        "dry_run": dry_run,
    }

    logger.info(f"Migration complete: {report}")

    return report


def run_migration(persona_space_path: Path, data_dir: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Run belief migration.

    Args:
        persona_space_path: Path to persona_space directory
        data_dir: Base data directory for belief store
        dry_run: If True, don't actually write

    Returns:
        Migration report
    """
    legacy_file = persona_space_path / "identity" / "beliefs.json"
    belief_store = BeliefStore(data_dir)

    return migrate_beliefs_from_legacy(legacy_file, belief_store, dry_run=dry_run)
