"""Unit tests for BeliefCanonicalizer."""

import pytest
from src.services.belief_canonicalizer import BeliefCanonicalizer


class TestContractionExpansion:
    """Test contraction expansion in canonicalization."""

    @pytest.fixture
    def canon(self):
        return BeliefCanonicalizer()

    @pytest.mark.parametrize("input_text,expected", [
        ("I'm happy", "i am happy"),
        ("I've been thinking", "i have been thinking"),
        ("I don't like crowds", "i do not like crowds"),
        ("I can't swim", "i cannot swim"),
        ("I won't give up", "i will not give up"),
        ("I couldn't do it", "i could not do it"),
        ("I wouldn't say that", "i would not say that"),
        ("I shouldn't worry", "i should not worry"),
        ("I wasn't there", "i was not there"),
        ("I haven't tried", "i have not tried"),
    ])
    def test_contraction_expansion(self, canon, input_text, expected):
        assert canon.canonicalize(input_text) == expected


class TestNormalization:
    """Test whitespace and punctuation normalization."""

    @pytest.fixture
    def canon(self):
        return BeliefCanonicalizer()

    @pytest.mark.parametrize("input_text,expected", [
        ("  I am   happy  ", "i am happy"),
        ("I\tam\nhappy", "i am happy"),
        ("I am happy.", "i am happy"),
        ("I am happy!", "i am happy"),
        ("I am happy?", "i am happy"),
        ("I AM HAPPY", "i am happy"),
        ("Really???", "really"),
    ])
    def test_normalization(self, canon, input_text, expected):
        assert canon.canonicalize(input_text) == expected

    def test_no_double_spaces(self, canon):
        inputs = ["I    am    happy", "I\t\tam", "  spaced   out  "]
        for text in inputs:
            assert "  " not in canon.canonicalize(text)

    def test_no_trailing_punctuation(self, canon):
        inputs = ["I am happy.", "I am happy!", "I am happy?", "Really???"]
        for text in inputs:
            out = canon.canonicalize(text)
            assert not out.endswith((".", "!", "?", ";"))

    def test_always_lowercase(self, canon):
        inputs = ["I AM HAPPY", "I Am Mixed", "SHOUTING"]
        for text in inputs:
            out = canon.canonicalize(text)
            assert out == out.lower()


class TestIdempotency:
    """Test that canonicalization is idempotent."""

    @pytest.fixture
    def canon(self):
        return BeliefCanonicalizer()

    def test_idempotent(self, canon):
        inputs = [
            "I'm very happy!",
            "  I   don't   like   crowds...  ",
            "I've ALWAYS been curious",
        ]
        for text in inputs:
            once = canon.canonicalize(text)
            twice = canon.canonicalize(once)
            assert once == twice, f"Not idempotent: {text}"


class TestHashing:
    """Test hash computation."""

    @pytest.fixture
    def canon(self):
        return BeliefCanonicalizer()

    def test_hash_deterministic(self, canon):
        text = "I am thoughtful"
        h1 = canon.compute_hash(canon.canonicalize(text))
        h2 = canon.compute_hash(canon.canonicalize(text))
        assert h1 == h2
        assert len(h1) == 32  # MD5 hex is 32 chars

    def test_hash_differs_for_different_text(self, canon):
        h1 = canon.compute_hash("i am happy")
        h2 = canon.compute_hash("i am sad")
        assert h1 != h2

    def test_hash_same_for_equivalent_text(self, canon):
        # These should canonicalize to the same text
        t1 = canon.canonicalize("I'm patient")
        t2 = canon.canonicalize("I am patient")
        assert canon.compute_hash(t1) == canon.compute_hash(t2)
