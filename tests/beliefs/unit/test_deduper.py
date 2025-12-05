"""Unit tests for BeliefAtomDeduper."""

import pytest
from dataclasses import dataclass, field
from typing import List, Optional
from src.services.belief_atom_deduper import BeliefAtomDeduper
from src.services.belief_canonicalizer import BeliefCanonicalizer


@dataclass
class MockAtom:
    """Mock atom for testing deduplication."""
    atom_text: str
    belief_type: str
    polarity: str
    confidence: float
    spans: Optional[List[tuple]] = None
    canonical_text: str = ""
    canonical_hash: str = ""
    original_text: str = ""


class TestDeduplication:
    """Test basic deduplication functionality."""

    @pytest.fixture
    def deduper(self):
        return BeliefAtomDeduper(BeliefCanonicalizer())

    @pytest.fixture
    def canon(self):
        return BeliefCanonicalizer()

    def _prepare(self, atoms, canon):
        """Prepare atoms with canonical text and hash."""
        for a in atoms:
            a.canonical_text = canon.canonicalize(a.atom_text)
            a.canonical_hash = canon.compute_hash(a.canonical_text)
            a.original_text = a.atom_text
        return atoms

    def test_no_duplicates_unchanged(self, deduper, canon):
        atoms = self._prepare([
            MockAtom("I am patient", "TRAIT", "affirm", 0.9, [(0, 13)]),
            MockAtom("I am kind", "TRAIT", "affirm", 0.85, [(14, 23)]),
        ], canon)

        result = deduper.dedup(atoms)
        assert len(result.deduped_atoms) == 2
        assert result.duplicates_removed == 0

    def test_exact_duplicates_merged(self, deduper, canon):
        atoms = self._prepare([
            MockAtom("I am patient", "TRAIT", "affirm", 0.9, [(0, 13)]),
            MockAtom("I am patient", "TRAIT", "affirm", 0.85, [(50, 63)]),
            MockAtom("I am patient", "TRAIT", "affirm", 0.7, [(100, 113)]),
        ], canon)

        result = deduper.dedup(atoms)
        assert len(result.deduped_atoms) == 1
        assert result.duplicates_removed == 2

    def test_highest_confidence_kept(self, deduper, canon):
        atoms = self._prepare([
            MockAtom("I am patient", "TRAIT", "affirm", 0.7, [(0, 13)]),
            MockAtom("I'm patient", "TRAIT", "affirm", 0.95, [(50, 60)]),
            MockAtom("i am patient", "TRAIT", "affirm", 0.8, [(100, 113)]),
        ], canon)

        result = deduper.dedup(atoms)
        assert len(result.deduped_atoms) == 1
        assert result.deduped_atoms[0].confidence == 0.95


class TestSpanPreservation:
    """Test that all spans are preserved when merging."""

    @pytest.fixture
    def deduper(self):
        return BeliefAtomDeduper(BeliefCanonicalizer())

    @pytest.fixture
    def canon(self):
        return BeliefCanonicalizer()

    def _prepare(self, atoms, canon):
        for a in atoms:
            a.canonical_text = canon.canonicalize(a.atom_text)
            a.canonical_hash = canon.compute_hash(a.canonical_text)
            a.original_text = a.atom_text
        return atoms

    def test_all_spans_preserved(self, deduper, canon):
        atoms = self._prepare([
            MockAtom("I am patient", "TRAIT", "affirm", 0.9, [(0, 13)]),
            MockAtom("I am patient", "TRAIT", "affirm", 0.85, [(50, 63)]),
            MockAtom("I am patient", "TRAIT", "affirm", 0.7, [(100, 113)]),
        ], canon)

        result = deduper.dedup(atoms)
        merged = result.deduped_atoms[0]

        assert (0, 13) in merged.spans
        assert (50, 63) in merged.spans
        assert (100, 113) in merged.spans


class TestNonMerging:
    """Test cases where atoms should NOT be merged."""

    @pytest.fixture
    def deduper(self):
        return BeliefAtomDeduper(BeliefCanonicalizer())

    @pytest.fixture
    def canon(self):
        return BeliefCanonicalizer()

    def _prepare(self, atoms, canon):
        for a in atoms:
            a.canonical_text = canon.canonicalize(a.atom_text)
            a.canonical_hash = canon.compute_hash(a.canonical_text)
            a.original_text = a.atom_text
        return atoms

    def test_different_polarity_not_merged(self, deduper, canon):
        atoms = self._prepare([
            MockAtom("I am patient", "TRAIT", "affirm", 0.9, [(0, 13)]),
            MockAtom("I am not patient", "TRAIT", "deny", 0.85, [(50, 66)]),
        ], canon)

        result = deduper.dedup(atoms)
        assert len(result.deduped_atoms) == 2

    def test_different_type_not_merged(self, deduper, canon):
        atoms = self._prepare([
            MockAtom("I am honest", "TRAIT", "affirm", 0.9, [(0, 11)]),
            MockAtom("I am honest", "VALUE", "affirm", 0.85, [(50, 61)]),
        ], canon)

        result = deduper.dedup(atoms)
        assert len(result.deduped_atoms) == 2


class TestInvariants:
    """Test invariants that must always hold."""

    @pytest.fixture
    def deduper(self):
        return BeliefAtomDeduper(BeliefCanonicalizer())

    @pytest.fixture
    def canon(self):
        return BeliefCanonicalizer()

    def _prepare(self, atoms, canon):
        for a in atoms:
            a.canonical_text = canon.canonicalize(a.atom_text)
            a.canonical_hash = canon.compute_hash(a.canonical_text)
            a.original_text = a.atom_text
        return atoms

    def test_output_lte_input(self, deduper, canon):
        atoms = self._prepare([
            MockAtom("I am A", "TRAIT", "affirm", 0.9, [(0, 6)]),
            MockAtom("I am A", "TRAIT", "affirm", 0.8, [(10, 16)]),
            MockAtom("I am B", "TRAIT", "affirm", 0.7, [(20, 26)]),
        ], canon)

        result = deduper.dedup(atoms)
        assert len(result.deduped_atoms) <= len(atoms)

    def test_unique_keys_in_output(self, deduper, canon):
        atoms = self._prepare([
            MockAtom("I am A", "TRAIT", "affirm", 0.9, None),
            MockAtom("I am A", "TRAIT", "affirm", 0.8, None),
            MockAtom("I am A", "VALUE", "affirm", 0.7, None),
            MockAtom("I am A", "TRAIT", "deny", 0.6, None),
        ], canon)

        result = deduper.dedup(atoms)
        keys = [(a.canonical_hash, a.polarity, a.belief_type) for a in result.deduped_atoms]
        assert len(keys) == len(set(keys))

    def test_empty_input_empty_output(self, deduper, canon):
        result = deduper.dedup([])
        assert len(result.deduped_atoms) == 0
        assert result.duplicates_removed == 0
