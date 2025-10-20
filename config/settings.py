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

    # LLM configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "venice")  # "openai" or "venice"
    LLM_BASE_URL: str | None = os.getenv("LLM_BASE_URL", "https://api.venice.ai/api/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "500"))

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


# Singleton settings instance
settings = Settings()
