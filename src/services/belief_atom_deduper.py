"""
Belief Atom Deduper for HTN Self-Belief Decomposer.

Deduplicates atoms within a single experience before storage.
Groups by (canonical_hash, polarity, belief_type) and keeps the
highest confidence atom while merging spans.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.services.belief_canonicalizer import BeliefCanonicalizer, CanonicalAtom
from src.services.belief_atomizer import RawAtom


@dataclass
class DedupResult:
    """
    Result of atom deduplication.

    Attributes:
        deduped_atoms: Deduplicated atoms
        duplicates_removed: Count of duplicates merged
        merge_log: Log of merges for debugging
    """
    deduped_atoms: List[CanonicalAtom]
    duplicates_removed: int
    merge_log: List[Dict] = field(default_factory=list)


class BeliefAtomDeduper:
    """
    Deduplicate atoms within a single experience.

    Groups atoms by (canonical_hash, polarity, belief_type).
    For each group:
    - Keep atom with highest confidence
    - Union all spans into a list
    - Log the merge for tracing
    """

    def __init__(self, canonicalizer: Optional[BeliefCanonicalizer] = None):
        """
        Initialize the deduper.

        Args:
            canonicalizer: Canonicalizer for text normalization.
                          If None, creates new instance.
        """
        self.canonicalizer = canonicalizer or BeliefCanonicalizer()

    def dedup(self, atoms: List[RawAtom]) -> DedupResult:
        """
        Deduplicate atoms within a single experience.

        Args:
            atoms: Raw atoms to deduplicate

        Returns:
            DedupResult with deduped atoms, count, and log
        """
        if not atoms:
            return DedupResult(
                deduped_atoms=[],
                duplicates_removed=0,
                merge_log=[],
            )

        # First canonicalize all atoms
        canonical_atoms = []
        for atom in atoms:
            canonical = self.canonicalizer.canonicalize(atom.atom_text)
            canonical_hash = self.canonicalizer.compute_hash(canonical)

            canonical_atoms.append({
                'raw': atom,
                'canonical_text': canonical,
                'canonical_hash': canonical_hash,
            })

        # Group by (canonical_hash, polarity, belief_type)
        groups: Dict[Tuple[str, str, str], List[Dict]] = {}

        for ca in canonical_atoms:
            key = (
                ca['canonical_hash'],
                ca['raw'].polarity,
                ca['raw'].belief_type,
            )
            if key not in groups:
                groups[key] = []
            groups[key].append(ca)

        # Process each group
        deduped: List[CanonicalAtom] = []
        merge_log: List[Dict] = []
        duplicates_removed = 0

        for key, group in groups.items():
            if len(group) == 1:
                # No duplicates
                ca = group[0]
                deduped.append(CanonicalAtom(
                    original_text=ca['raw'].atom_text,
                    canonical_text=ca['canonical_text'],
                    canonical_hash=ca['canonical_hash'],
                    belief_type=ca['raw'].belief_type,
                    polarity=ca['raw'].polarity,
                    spans=ca['raw'].spans,
                    confidence=ca['raw'].confidence,
                ))
            else:
                # Merge duplicates
                duplicates_removed += len(group) - 1

                # Find highest confidence atom
                best = max(group, key=lambda x: x['raw'].confidence)

                # Collect all spans
                all_spans = []
                merged_from = []

                for ca in group:
                    if ca['raw'].spans:
                        all_spans.extend(ca['raw'].spans)
                    if ca != best:
                        merged_from.append(ca['raw'].atom_text)

                # Create merged atom
                merged = CanonicalAtom(
                    original_text=best['raw'].atom_text,
                    canonical_text=best['canonical_text'],
                    canonical_hash=best['canonical_hash'],
                    belief_type=best['raw'].belief_type,
                    polarity=best['raw'].polarity,
                    spans=all_spans if all_spans else None,
                    confidence=best['raw'].confidence,
                )
                deduped.append(merged)

                # Log the merge
                merge_log.append({
                    'kept_text': best['raw'].atom_text,
                    'merged_from': merged_from,
                    'combined_spans': all_spans,
                    'group_size': len(group),
                })

        # Sort by canonical_text for deterministic ordering
        deduped.sort(key=lambda a: a.canonical_text)

        return DedupResult(
            deduped_atoms=deduped,
            duplicates_removed=duplicates_removed,
            merge_log=merge_log,
        )


def dedup_atoms(atoms: List[RawAtom]) -> DedupResult:
    """
    Convenience function to deduplicate atoms.

    Args:
        atoms: Raw atoms to deduplicate

    Returns:
        DedupResult
    """
    deduper = BeliefAtomDeduper()
    return deduper.dedup(atoms)
