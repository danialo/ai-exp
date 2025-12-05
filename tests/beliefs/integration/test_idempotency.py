"""Integration tests for idempotency guarantees.

Processing the same experience twice should produce identical results.
"""

import pytest
from uuid import uuid4
from datetime import datetime, timezone


class TestCanonicalizationIdempotency:
    """Canonicalization should be idempotent."""

    def test_canonicalization_idempotent(self):
        """Canonicalizing twice produces same result."""
        from src.services.belief_canonicalizer import BeliefCanonicalizer

        canon = BeliefCanonicalizer()

        test_cases = [
            "I'm very patient",
            "I   AM   HAPPY",
            "i dont like mornings",
            "We're excited about this!",
        ]

        for text in test_cases:
            first = canon.canonicalize(text)
            second = canon.canonicalize(first)
            assert first == second, f"Not idempotent: {text} -> {first} -> {second}"

    def test_hash_idempotent(self):
        """Same text always produces same hash."""
        from src.services.belief_canonicalizer import BeliefCanonicalizer

        canon = BeliefCanonicalizer()

        texts = [
            "I am patient",
            "I'm very thoughtful",
            "I always overthink things",
        ]

        for text in texts:
            canonical = canon.canonicalize(text)
            hash1 = canon.compute_hash(canonical)
            hash2 = canon.compute_hash(canonical)
            assert hash1 == hash2


class TestResolutionIdempotency:
    """Same atom resolved twice should match to same node."""

    def test_resolve_same_atom_twice(self, test_db, fake_embedder):
        """Resolving identical atom twice gives same result."""
        from src.services.belief_resolver import BeliefResolver
        from src.services.belief_canonicalizer import BeliefCanonicalizer
        from dataclasses import dataclass

        @dataclass
        class MockAtom:
            canonical_text: str
            canonical_hash: str
            belief_type: str
            polarity: str
            original_text: str = ""
            spans: list = None
            confidence: float = 0.9

        canon = BeliefCanonicalizer()
        resolver = BeliefResolver(embedder=fake_embedder)

        text = "i am patient"
        atom = MockAtom(
            canonical_text=text,
            canonical_hash=canon.compute_hash(text),
            belief_type="TRAIT",
            polarity="affirm",
        )

        # First resolution (creates new node)
        result1 = resolver.resolve(atom)

        # Second resolution (should match existing)
        result2 = resolver.resolve(atom)

        # Both should have same outcome structure
        assert result1.outcome == result2.outcome


class TestDeduperIdempotency:
    """Deduplication should be idempotent."""

    def test_dedup_same_atoms_twice(self):
        """Deduping same set of atoms twice gives same result."""
        from src.services.belief_canonicalizer import BeliefCanonicalizer
        from src.services.belief_atom_deduper import BeliefAtomDeduper
        from dataclasses import dataclass

        @dataclass
        class MockAtom:
            canonical_text: str
            canonical_hash: str
            belief_type: str
            polarity: str
            original_text: str = ""
            atom_text: str = ""
            spans: list = None
            confidence: float = 0.9

        canon = BeliefCanonicalizer()
        deduper = BeliefAtomDeduper(canon)

        # Create atoms with different confidences
        atoms1 = [
            MockAtom(
                canonical_text="i am patient",
                canonical_hash=canon.compute_hash("i am patient"),
                belief_type="TRAIT",
                polarity="affirm",
                atom_text="I am patient",
                confidence=0.9,
            ),
            MockAtom(
                canonical_text="i am patient",
                canonical_hash=canon.compute_hash("i am patient"),
                belief_type="TRAIT",
                polarity="affirm",
                atom_text="I'm patient",
                confidence=0.8,
            ),
        ]

        atoms2 = [
            MockAtom(
                canonical_text="i am patient",
                canonical_hash=canon.compute_hash("i am patient"),
                belief_type="TRAIT",
                polarity="affirm",
                atom_text="I am patient",
                confidence=0.9,
            ),
            MockAtom(
                canonical_text="i am patient",
                canonical_hash=canon.compute_hash("i am patient"),
                belief_type="TRAIT",
                polarity="affirm",
                atom_text="I'm patient",
                confidence=0.8,
            ),
        ]

        result1 = deduper.dedup(atoms1)
        result2 = deduper.dedup(atoms2)

        assert len(result1.deduped_atoms) == len(result2.deduped_atoms)


class TestEpistemicsIdempotency:
    """Epistemic extraction should be deterministic."""

    def test_same_text_same_frame(self):
        """Same text always produces same epistemic frame."""
        from src.services.epistemics_rules import EpistemicsRulesEngine

        engine = EpistemicsRulesEngine()

        test_cases = [
            "I always feel happy",
            "I used to be shy",
            "I might be wrong about this",
            "I'm becoming more confident",
        ]

        for text in test_cases:
            result1 = engine.extract(text)
            result2 = engine.extract(text)

            assert result1.frame.temporal_scope == result2.frame.temporal_scope
            assert result1.frame.modality == result2.frame.modality
            assert result1.confidence == result2.confidence

    def test_signals_deterministic(self):
        """Same text produces same signals."""
        from src.services.epistemics_rules import EpistemicsRulesEngine

        engine = EpistemicsRulesEngine()

        text = "I always tend to overthink things"

        result1 = engine.extract(text)
        result2 = engine.extract(text)

        assert len(result1.signals) == len(result2.signals)
