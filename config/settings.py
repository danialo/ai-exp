"""Configuration settings for AI Experience Memory System.

Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment."""

    # Project root directory
    PROJECT_ROOT = Path(__file__).parent.parent

    # Database paths
    RAW_STORE_DB_PATH: str = os.getenv(
        "RAW_STORE_DB_PATH",
        str(PROJECT_ROOT / "data" / "raw_store.db"),
    )
    VECTOR_INDEX_PATH: str = os.getenv(
        "VECTOR_INDEX_PATH",
        str(PROJECT_ROOT / "data" / "vector_index"),
    )

    # Embedding configuration
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    VENICEAI_API_KEY: str | None = os.getenv("VENICEAI_API_KEY")

    # Web search and browsing configuration
    SERP_API_KEY: str | None = os.getenv("SERPAPI_API_KEY")
    BROWSER_HEADLESS: bool = os.getenv("BROWSER_HEADLESS", "true").lower() == "true"
    BROWSER_TIMEOUT_MS: int = int(os.getenv("BROWSER_TIMEOUT_MS", "30000"))  # 30 seconds
    BROWSER_SCREENSHOTS_ENABLED: bool = os.getenv("BROWSER_SCREENSHOTS_ENABLED", "false").lower() == "true"
    BROWSER_SCREENSHOTS_PATH: str = os.getenv(
        "BROWSER_SCREENSHOTS_PATH",
        str(PROJECT_ROOT / "persona_space" / "logs" / "screenshots"),
    )
    MAX_SEARCHES_PER_CONVERSATION: int = int(os.getenv("MAX_SEARCHES_PER_CONVERSATION", "5"))
    MAX_URL_FETCHES_PER_CONVERSATION: int = int(os.getenv("MAX_URL_FETCHES_PER_CONVERSATION", "3"))
    WEB_CONTENT_MAX_LENGTH: int = int(os.getenv("WEB_CONTENT_MAX_LENGTH", "10000"))  # chars for interpretation

    # LLM configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "venice")  # "openai" or "venice"
    LLM_BASE_URL: str | None = os.getenv("LLM_BASE_URL", "https://api.venice.ai/api/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "500"))
    LLM_TOP_K: int | None = int(os.getenv("LLM_TOP_K")) if os.getenv("LLM_TOP_K") else None

    # Available models for UI selection
    # Only models that support function calling (tools) for persona autonomy
    AVAILABLE_MODELS = {
        "openai": {
            "gpt-4o": {
                "name": "GPT-4o",
                "base_url": None,
                "supports_logit_bias": True,
                "is_reasoning_model": False,
            },
            "gpt-4o-mini": {
                "name": "GPT-4o Mini",
                "base_url": None,
                "supports_logit_bias": True,
                "is_reasoning_model": False,
            },
            "gpt-5": {
                "name": "GPT-5",
                "base_url": None,
                "supports_logit_bias": False,
                "is_reasoning_model": True,
            },
            "gpt-5-mini": {
                "name": "GPT-5 Mini",
                "base_url": None,
                "supports_logit_bias": False,
                "is_reasoning_model": True,
            },
            "o1-preview": {
                "name": "O1 Preview",
                "base_url": None,
                "supports_logit_bias": False,
                "is_reasoning_model": True,
            },
            "o1-mini": {
                "name": "O1 Mini",
                "base_url": None,
                "supports_logit_bias": False,
                "is_reasoning_model": True,
            },
        },
    }

    # Persona mode configuration
    PERSONA_MODE_ENABLED: bool = os.getenv("PERSONA_MODE_ENABLED", "false").lower() == "true"
    PERSONA_SPACE_PATH: str = os.getenv("PERSONA_SPACE_PATH", str(PROJECT_ROOT / "persona_space"))
    PERSONA_TEMPERATURE: float = float(os.getenv("PERSONA_TEMPERATURE", "1.0"))  # High for creative, non-canned responses
    PERSONA_TOP_K: int = int(os.getenv("PERSONA_TOP_K", "100"))  # Increased for emergent behavior
    PERSONA_TOP_P: float = float(os.getenv("PERSONA_TOP_P", "0.92"))  # Nucleus sampling
    PERSONA_PRESENCE_PENALTY: float = float(os.getenv("PERSONA_PRESENCE_PENALTY", "0.6"))  # Reduce repetition across responses
    PERSONA_FREQUENCY_PENALTY: float = float(os.getenv("PERSONA_FREQUENCY_PENALTY", "0.3"))  # Reduce within-response repetition

    # Anti-meta-talk configuration
    ANTI_METATALK_ENABLED: bool = os.getenv("ANTI_METATALK_ENABLED", "true").lower() == "true"
    LOGIT_BIAS_STRENGTH: float = float(os.getenv("LOGIT_BIAS_STRENGTH", "-100"))  # Strong penalty for meta-talk tokens
    AUTO_REWRITE_METATALK: bool = os.getenv("AUTO_REWRITE_METATALK", "true").lower() == "true"

    # Retrieval parameters
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    RECENCY_WEIGHT: float = float(os.getenv("RECENCY_WEIGHT", "0.2"))
    SEMANTIC_WEIGHT: float = float(os.getenv("SEMANTIC_WEIGHT", "0.8"))

    # Session and consolidation configuration
    SESSION_TIMEOUT_MINUTES: int = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
    SHORT_TERM_WEIGHT: float = float(os.getenv("SHORT_TERM_WEIGHT", "0.7"))
    LONG_TERM_WEIGHT: float = float(os.getenv("LONG_TERM_WEIGHT", "0.3"))
    CONSOLIDATION_ENABLED: bool = os.getenv("CONSOLIDATION_ENABLED", "true").lower() == "true"
    DECAY_RECALC_INTERVAL: int = int(os.getenv("DECAY_RECALC_INTERVAL", "86400"))  # 1 day

    # Vector index paths for dual-index system
    SHORT_TERM_INDEX_PATH: str = os.getenv(
        "SHORT_TERM_INDEX_PATH",
        str(PROJECT_ROOT / "data" / "vector_index_short_term"),
    )
    LONG_TERM_INDEX_PATH: str = os.getenv(
        "LONG_TERM_INDEX_PATH",
        str(PROJECT_ROOT / "data" / "vector_index_long_term"),
    )
    SELF_INDEX_PATH: str = os.getenv(
        "SELF_INDEX_PATH",
        str(PROJECT_ROOT / "data" / "vector_index_self"),
    )
    BELIEFS_INDEX_PATH: str = os.getenv(
        "BELIEFS_INDEX_PATH",
        str(PROJECT_ROOT / "data" / "vector_index_beliefs"),
    )

    # Belief-memory retrieval configuration
    BELIEF_MEMORY_WEIGHT: float = float(os.getenv("BELIEF_MEMORY_WEIGHT", "0.7"))  # Weight for beliefs in self-queries
    MEMORY_WEIGHT: float = float(os.getenv("MEMORY_WEIGHT", "0.3"))  # Weight for memories in self-queries

    # Self-concept configuration
    SELF_EXTRACTION_FREQUENCY: int = int(os.getenv("SELF_EXTRACTION_FREQUENCY", "10"))  # After N narratives
    CORE_TRAIT_THRESHOLD: int = int(os.getenv("CORE_TRAIT_THRESHOLD", "5"))  # Narratives for core trait
    SURFACE_TRAIT_THRESHOLD: int = int(os.getenv("SURFACE_TRAIT_THRESHOLD", "2"))  # Narratives for surface trait
    SURFACE_DECAY_DAYS: int = int(os.getenv("SURFACE_DECAY_DAYS", "7"))  # Days before surface traits decay
    CORE_TRAIT_LIMIT: int = int(os.getenv("CORE_TRAIT_LIMIT", "5"))  # Max core traits in prompt
    SURFACE_TRAIT_LIMIT: int = int(os.getenv("SURFACE_TRAIT_LIMIT", "3"))  # Max surface traits in prompt

    # Redis configuration (for awareness loop)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str | None = os.getenv("REDIS_PASSWORD")

    # Awareness loop configuration
    AWARENESS_ENABLED: bool = os.getenv("AWARENESS_ENABLED", "false").lower() == "true"
    AWARENESS_TICK_RATE_FAST: float = float(os.getenv("AWARENESS_TICK_RATE_FAST", "2.0"))
    AWARENESS_TICK_RATE_SLOW: float = float(os.getenv("AWARENESS_TICK_RATE_SLOW", "0.1"))
    AWARENESS_INTROSPECTION_INTERVAL: int = int(os.getenv("AWARENESS_INTROSPECTION_INTERVAL", "180"))
    AWARENESS_INTROSPECTION_JITTER: int = int(os.getenv("AWARENESS_INTROSPECTION_JITTER", "5"))
    AWARENESS_SNAPSHOT_INTERVAL: int = int(os.getenv("AWARENESS_SNAPSHOT_INTERVAL", "60"))
    AWARENESS_BUFFER_SIZE: int = int(os.getenv("AWARENESS_BUFFER_SIZE", "512"))
    AWARENESS_QUEUE_MAXSIZE: int = int(os.getenv("AWARENESS_QUEUE_MAXSIZE", "2048"))
    AWARENESS_NOTES_MAX: int = int(os.getenv("AWARENESS_NOTES_MAX", "100"))
    AWARENESS_EMBEDDING_DIM: int = int(os.getenv("AWARENESS_EMBEDDING_DIM", "64"))
    AWARENESS_EMBEDDING_CACHE_TTL: int = int(os.getenv("AWARENESS_EMBEDDING_CACHE_TTL", "300"))
    AWARENESS_WATCHDOG_THRESHOLD_MS: float = float(os.getenv("AWARENESS_WATCHDOG_THRESHOLD_MS", "250"))
    AWARENESS_WATCHDOG_STRIKES: int = int(os.getenv("AWARENESS_WATCHDOG_STRIKES", "3"))
    AWARENESS_INTROSPECTION_BUDGET_PER_MIN: int = int(os.getenv("AWARENESS_INTROSPECTION_BUDGET_PER_MIN", "3000"))
    AWARENESS_DATA_DIR: str = os.getenv("AWARENESS_DATA_DIR", str(PROJECT_ROOT / "data"))

    # Identity anchor configuration (dual-anchor system)
    IDENTITY_ANCHOR_BETA_WEEK_CAP: float = float(os.getenv("IDENTITY_ANCHOR_BETA_WEEK_CAP", "0.01"))
    IDENTITY_LEDGER_DIR: str = os.getenv("IDENTITY_LEDGER_DIR", str(PROJECT_ROOT / "data" / "identity"))

    # Autonomous belief gardener configuration
    BELIEF_GARDENER_ENABLED: bool = os.getenv("BELIEF_GARDENER_ENABLED", "false").lower() == "true"
    BELIEF_GARDENER_SCAN_INTERVAL: int = int(os.getenv("BELIEF_GARDENER_SCAN_INTERVAL", "60"))  # minutes
    BELIEF_GARDENER_MIN_EVIDENCE_TENTATIVE: int = int(os.getenv("BELIEF_GARDENER_MIN_EVIDENCE_TENTATIVE", "3"))
    BELIEF_GARDENER_MIN_EVIDENCE_ASSERTED: int = int(os.getenv("BELIEF_GARDENER_MIN_EVIDENCE_ASSERTED", "5"))
    BELIEF_GARDENER_DAILY_BUDGET_FORMATIONS: int = int(os.getenv("BELIEF_GARDENER_DAILY_BUDGET_FORMATIONS", "3"))
    BELIEF_GARDENER_DAILY_BUDGET_PROMOTIONS: int = int(os.getenv("BELIEF_GARDENER_DAILY_BUDGET_PROMOTIONS", "5"))
    BELIEF_GARDENER_DAILY_BUDGET_DEPRECATIONS: int = int(os.getenv("BELIEF_GARDENER_DAILY_BUDGET_DEPRECATIONS", "3"))
    BELIEF_GARDENER_LOOKBACK_DAYS: int = int(os.getenv("BELIEF_GARDENER_LOOKBACK_DAYS", "30"))

    # Affect blending weights (user, memory, self)
    @staticmethod
    def get_affect_weights() -> Tuple[float, float, float]:
        """Parse affect weights from environment.

        Returns:
            Tuple of (user_weight, memory_weight, self_weight)
        """
        weights_str = os.getenv("AFFECT_WEIGHTS", "0.5,0.3,0.2")
        weights = [float(w.strip()) for w in weights_str.split(",")]
        if len(weights) != 3:
            raise ValueError(f"AFFECT_WEIGHTS must have 3 values, got {len(weights)}")
        if not abs(sum(weights) - 1.0) < 0.01:
            raise ValueError(f"AFFECT_WEIGHTS must sum to 1.0, got {sum(weights)}")
        return tuple(weights)

    @classmethod
    def ensure_data_directories(cls) -> None:
        """Ensure data directories exist."""
        Path(cls.RAW_STORE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(cls.VECTOR_INDEX_PATH).mkdir(parents=True, exist_ok=True)
        Path(cls.SHORT_TERM_INDEX_PATH).mkdir(parents=True, exist_ok=True)
        Path(cls.LONG_TERM_INDEX_PATH).mkdir(parents=True, exist_ok=True)
        Path(cls.SELF_INDEX_PATH).mkdir(parents=True, exist_ok=True)
        Path(cls.BELIEFS_INDEX_PATH).mkdir(parents=True, exist_ok=True)
        if cls.PERSONA_MODE_ENABLED:
            Path(cls.PERSONA_SPACE_PATH).mkdir(parents=True, exist_ok=True)


# Singleton settings instance
settings = Settings()
