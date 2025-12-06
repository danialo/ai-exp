"""
Pytest fixtures for HTN Self-Belief Decomposer test suite.

Fixtures use StaticPool for in-memory SQLite (shared connection),
NullPool for file-backed SQLite (concurrency tests).
"""

import hashlib
import pytest
import numpy as np
from datetime import datetime, timezone
from uuid import uuid4
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.pool import StaticPool, NullPool

# Ensure SQLModel models are registered before create_all()
import src.memory.models.belief_node  # noqa: F401
import src.memory.models.belief_occurrence  # noqa: F401
import src.memory.models.tentative_link  # noqa: F401
import src.memory.models.conflict_edge  # noqa: F401
import src.memory.models.stream_assignment  # noqa: F401


# ============================================================================
# DATABASE FIXTURES
# ============================================================================

@pytest.fixture(scope="function")
def test_db():
    """
    Fresh shared in-memory SQLite per test.

    Why StaticPool:
    - sqlite:///:memory: is per-connection, which breaks if code opens a second Session.
    - sqlite:// + StaticPool forces one shared connection, making the schema stable.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        yield session
        session.rollback()


@pytest.fixture(scope="function")
def file_db(tmp_path):
    """
    File-backed SQLite for concurrency tests.

    Why NullPool:
    - avoids pooled connections crossing threads
    - each Session gets its own connection

    Returns a session for consistency with test_db fixture.
    The session.bind provides access to the engine/path if needed.
    """
    db_path = tmp_path / "test_concurrent.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        yield session
        session.rollback()


# ============================================================================
# CONFIG FIXTURE
# ============================================================================

class ConfigObject:
    """Minimal config object that supports dot-access."""
    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, ConfigObject(v))
            else:
                setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


@pytest.fixture
def test_config(tmp_path):
    """Create test configuration with all required settings."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    prompt_files = {
        "atomizer_system": "You are a belief extractor. Output only valid JSON.",
        "atomizer_user": "Extract beliefs from: {input_text}",
        "repair_json": "Fix this JSON: {broken_json}",
        "epistemics_fallback": "Analyze temporal scope: {atom_text}",
        "verifier": "Are these the same concept? A: {atom_text} B: {candidate_text}",
    }

    for name, content in prompt_files.items():
        (prompts_dir / f"{name}_v1.txt").write_text(content)

    config_dict = {
        "extractor": {
            "atomizer_model": "test-model",
            "epistemics_model": "test-model",
            "verifier_model": "test-model",
            "temperature": 0,
            "max_json_repair_attempts": 1,
        },
        "prompts": {
            "atomizer_system": str(prompts_dir / "atomizer_system_v1.txt"),
            "atomizer_user": str(prompts_dir / "atomizer_user_v1.txt"),
            "repair_json": str(prompts_dir / "repair_json_v1.txt"),
            "epistemics_fallback": str(prompts_dir / "epistemics_fallback_v1.txt"),
            "verifier": str(prompts_dir / "verifier_v1.txt"),
        },
        "source_context": {
            "mode_field": "interaction_mode",
            "mode_weights": {
                "journaling": 1.0,
                "introspection": 0.95,
                "normal_chat": 0.8,
                "roleplay": 0.4,
                "heated": 0.5,
                "unknown": 0.7,
            },
            "vad": {
                "enabled": False,
                "arousal_field": "affect.arousal",
                "arousal_weight": 0.35,
            },
            "heuristic_fallback": {
                "enabled": True,
                "profanity_penalty": 0.15,
                "caps_ratio_threshold": 0.3,
                "caps_penalty": 0.10,
                "exclaim_density_threshold": 0.1,
                "exclaim_penalty": 0.05,
            },
        },
        "context": {
            "strategy": "conversation_mode",
            "fallback": "experience_id",
        },
        "epistemics": {
            "llm_fallback_threshold": 0.6,
            "cue_conflict_resolution": "specificity_then_rightmost",
            "default_temporal_scope": "ongoing",
            "default_modality": "certain",
            "modality_confidence_caps": {
                "possible": 0.4,
                "likely": 0.6,
                "unsure": 0.2,
                "certain": 1.0,
            },
            "temporal_specificity": {
                "past": 6,
                "transitional": 5,
                "habitual": 4,
                "ongoing": 3,
                "state": 2,
                "unknown": 1,
            },
            "cues": {
                "negation": ["not", "don't", "doesn't", "never", "no longer", "cannot", "can't", "won't"],
                "modality": {
                    "possible": ["might", "maybe", "perhaps", "possibly"],
                    "likely": ["i think", "i suspect", "probably"],
                    "unsure": ["unsure", "not sure", "uncertain"],
                },
                "past": ["used to", "formerly", "previously", "back then"],
                "transitional": ["lately", "recently", "these days", "increasingly"],
                "habitual_strong": ["always", "never", "every time"],
                "habitual_soft": ["usually", "generally", "typically", "tend to"],
                "ongoing": ["still", "continue to", "keep"],
                "state": ["right now", "at the moment", "currently", "today"],
            },
            "degree": {
                "strong": ["extremely", "very", "really"],
                "moderate": ["quite", "fairly"],
                "weak": ["somewhat", "slightly", "a bit"],
            },
            "degree_values": {
                "strong": 0.9,
                "moderate": 0.6,
                "weak": 0.3,
                "default": 0.5,
            },
        },
        "streams": {
            "types": ["identity", "state", "meta", "relational"],
            "mapping": {
                "FEELING_STATE": {
                    "state": {"primary": "state", "secondary": "identity", "confidence": 0.65},
                    "habitual": {"primary": "identity", "secondary": "state", "confidence": 0.80},
                    "ongoing": {"primary": "identity", "secondary": "state", "confidence": 0.75},
                    "transitional": {"primary": "state", "secondary": "identity", "confidence": 0.70},
                    "default": {"primary": "state", "secondary": "identity", "confidence": 0.60},
                },
                "TRAIT": {"default": {"primary": "identity", "secondary": None, "confidence": 0.85}},
                "PREFERENCE": {"default": {"primary": "identity", "secondary": None, "confidence": 0.80}},
                "VALUE": {"default": {"primary": "identity", "secondary": None, "confidence": 0.90}},
                "CAPABILITY_LIMIT": {"default": {"primary": "identity", "secondary": None, "confidence": 0.80}},
                "META_BELIEF": {"default": {"primary": "meta", "secondary": "identity", "confidence": 0.80}},
                "RELATIONAL": {"default": {"primary": "relational", "secondary": "identity", "confidence": 0.75}},
                "default": {"default": {"primary": "identity", "secondary": None, "confidence": 0.50}},
            },
        },
        "embeddings": {
            "enabled": True,
            "model": "test-model",
            "dimension": 384,
            "linear_scan_max_nodes": 50000,
        },
        "resolution": {
            "top_k": 10,
            "match_threshold": 0.91,
            "no_match_threshold": 0.75,
            "verifier": {"enabled": True, "trigger_band": [0.75, 0.91]},
            "tentative_link": {
                "auto_accept_threshold": 0.85,
                "auto_reject_threshold": 0.15,
                "confidence_params": {"a": 1.2, "b": 0.9, "c": 0.06},
                "age_definition": "days_since_last_support_else_created",
            },
            "tension": {"enabled": True, "embedding_threshold": 0.88, "top_k_conflict_check": 20},
        },
        "concurrency": {
            "strategy": "unique_canonical_hash_retry",
            "max_retries": 3,
            "retry_delay_ms": 100,
        },
        "scoring": {
            "half_life_days": {"identity": 60, "state": 7, "meta": 30, "relational": 30},
            "support": {"k_n": 10.0},
            "spread": {"midpoint_days": 14.0, "temperature": 4.0},
            "diversity": {"midpoint_contexts": 5.0, "temperature": 1.5},
            "conflict_penalty": {"enabled": True, "recent_window_days": 30, "weight": 0.35},
            "status_thresholds": {"developing": 0.3, "core": 0.6},
        },
        "migration": {
            "promote_state_to_identity": {
                "enabled": True,
                "strategy": "absolute",
                "min_spread": 0.70,
                "min_diversity": 0.60,
                "min_activation": 0.35,
            },
            "ratchet": {"enabled": True, "allow_demotion": False, "demotion_triggers": ["explicit_retraction"]},
        },
        "logging": {
            "eval_events": {"enabled": True, "path": str(tmp_path / "eval_events"), "format": "jsonl"},
        },
    }

    return ConfigObject(config_dict)


# ============================================================================
# EXPERIENCE FIXTURES
# ============================================================================

@dataclass
class MockExperience:
    """Mock experience object for testing."""
    id: str
    type: str
    content: str
    interaction_mode: str
    conversation_id: Optional[str]
    affect: Dict[str, float]
    created_at: datetime
    session_id: Optional[str] = None

    def get_text(self) -> str:
        return self.content


@pytest.fixture
def sample_experience():
    """Standard test experience."""
    return MockExperience(
        id=str(uuid4()),
        type="self_definition",
        content="I am thoughtful and I tend to overthink things.",
        interaction_mode="normal_chat",
        conversation_id="conv_123",
        affect={"arousal": 0.3, "valence": 0.5},
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def journaling_experience():
    """High-weight journaling experience."""
    return MockExperience(
        id=str(uuid4()),
        type="self_definition",
        content="I've always been curious about how things work.",
        interaction_mode="journaling",
        conversation_id="conv_456",
        affect={"arousal": 0.2, "valence": 0.6},
        created_at=datetime.now(timezone.utc),
    )


# ============================================================================
# THREAD-SAFE FAKE EMBEDDER
# ============================================================================

class FakeEmbedder:
    """Deterministic embedder using LOCAL RNG (thread-safe)."""

    def __init__(self, dimension: int = 384, record_calls: bool = True):
        self.dimension = dimension
        self.enabled = True
        self._record_calls = record_calls
        self.embed_calls = []
        self.linear_scan_max = 50000

    def embed(self, text: str) -> np.ndarray:
        if self._record_calls:
            self.embed_calls.append(text)

        hash_bytes = hashlib.sha256(text.lower().encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], "big")

        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed(t) for t in texts]

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def text_similarity(self, a: str, b: str) -> float:
        """Simple text similarity for fallback."""
        a_set = set(a.lower().split())
        b_set = set(b.lower().split())
        if not a_set or not b_set:
            return 0.0
        intersection = len(a_set & b_set)
        union = len(a_set | b_set)
        return intersection / union if union > 0 else 0.0

    def serialize(self, embedding: np.ndarray) -> bytes:
        return embedding.tobytes()

    def deserialize(self, data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.float32)

    def should_use_linear_scan(self, count: int) -> bool:
        return count <= self.linear_scan_max

    def reset_call_log(self):
        self.embed_calls = []


@pytest.fixture
def fake_embedder():
    """Fixture for deterministic embedder."""
    return FakeEmbedder(dimension=384)


# ============================================================================
# MOCK LLM CLIENT
# ============================================================================

class MockLLMClient:
    """LLM mock with call tracking for assertions."""

    def __init__(self):
        self.responses: Dict[str, str] = {}
        self.call_log: List[str] = []

    def set_response(self, prompt_contains: str, response: str):
        """Set response for prompts containing the given text."""
        self.responses[prompt_contains] = response

    def complete(self, prompt: str) -> str:
        """Mock completion that returns pre-set responses."""
        self.call_log.append(prompt)
        for key, response in self.responses.items():
            if key in prompt:
                return response
        raise ValueError(f"No mock response for prompt containing: {list(self.responses.keys())}")

    def was_called(self) -> bool:
        return len(self.call_log) > 0

    def was_called_with(self, text: str) -> bool:
        return any(text in call for call in self.call_log)

    def call_count(self) -> int:
        return len(self.call_log)

    def reset(self):
        self.call_log = []


@pytest.fixture
def mock_llm():
    """Fixture for mock LLM client."""
    return MockLLMClient()
