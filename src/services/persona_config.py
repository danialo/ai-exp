"""
Persona Configuration Loader - Allows the persona to control its own LLM parameters.

The persona can modify meta/llm_config.json to adjust its generation behavior.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class PersonaLLMConfig:
    """Configuration for LLM generation parameters controlled by the persona."""

    # Generation parameters
    temperature: float = 0.9
    top_k: Optional[int] = None
    top_p: float = 0.92
    presence_penalty: float = 0.6
    frequency_penalty: float = 0.3
    max_tokens: Optional[int] = None

    # Meta-configuration
    enable_anti_metatalk: bool = True
    auto_rewrite_metatalk: bool = True

    # Retrieval parameters
    retrieve_memories: bool = True
    memory_top_k: int = 5


class PersonaConfigLoader:
    """Loads and validates persona LLM configuration from persona_space."""

    def __init__(self, persona_space_path: str):
        """
        Initialize config loader.

        Args:
            persona_space_path: Path to persona_space directory
        """
        self.persona_space = Path(persona_space_path)
        self.config_path = self.persona_space / "meta" / "llm_config.json"

    def load_config(self) -> PersonaLLMConfig:
        """
        Load configuration from file with validation and defaults.

        Returns:
            PersonaLLMConfig with validated values
        """
        # Start with defaults
        config = PersonaLLMConfig()

        # Try to load from file
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)

                # Update config with file values
                for key, value in data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                    else:
                        logger.warning(f"Unknown config key in llm_config.json: {key}")

                # Validate loaded config
                self._validate_config(config)

                logger.info(f"Loaded persona LLM config from {self.config_path}")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse llm_config.json: {e}. Using defaults.")
            except Exception as e:
                logger.error(f"Error loading llm_config.json: {e}. Using defaults.")
        else:
            logger.info("No llm_config.json found, using defaults")

        return config

    def _validate_config(self, config: PersonaLLMConfig) -> None:
        """
        Validate config values and log warnings for extreme settings.

        Args:
            config: Config to validate

        Raises:
            ValueError: If config values are out of valid ranges
        """
        # Validate temperature
        if config.temperature < 0 or config.temperature > 2:
            raise ValueError(f"temperature must be 0-2, got {config.temperature}")

        if config.temperature > 1.5:
            logger.warning(f"High temperature {config.temperature} - responses may be very random")
        elif config.temperature < 0.3:
            logger.warning(f"Low temperature {config.temperature} - responses may be repetitive")

        # Validate top_p
        if config.top_p < 0 or config.top_p > 1:
            raise ValueError(f"top_p must be 0-1, got {config.top_p}")

        # Validate penalties
        if config.presence_penalty < -2 or config.presence_penalty > 2:
            raise ValueError(f"presence_penalty must be -2 to 2, got {config.presence_penalty}")

        if config.frequency_penalty < -2 or config.frequency_penalty > 2:
            raise ValueError(f"frequency_penalty must be -2 to 2, got {config.frequency_penalty}")

        # Validate max_tokens
        if config.max_tokens is not None and config.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive or null, got {config.max_tokens}")

        # Validate memory_top_k
        if config.memory_top_k < 1:
            raise ValueError(f"memory_top_k must be positive, got {config.memory_top_k}")

        if config.memory_top_k > 20:
            logger.warning(f"High memory_top_k {config.memory_top_k} - may include irrelevant memories")

    def create_default_config(self) -> None:
        """Create default config file with explanatory comments."""
        default_config = {
            "_note": "This file controls your LLM generation parameters. Modify to adjust your behavior.",
            "_docs": {
                "temperature": "0-2. Higher = more random/creative. Lower = more focused/deterministic. Default: 0.9",
                "top_k": "Integer or null. Limits vocab to top K tokens. Higher = more diversity. Default: null (disabled)",
                "top_p": "0-1. Nucleus sampling. Lower = more focused. Higher = more diverse. Default: 0.92",
                "presence_penalty": "-2 to 2. Positive = avoid repeating topics. Negative = encourage repetition. Default: 0.6",
                "frequency_penalty": "-2 to 2. Positive = avoid repeating words. Negative = encourage repetition. Default: 0.3",
                "max_tokens": "Integer or null. Maximum response length. Null = use model default. Default: null",
                "enable_anti_metatalk": "Boolean. Enable token suppression for meta-talk phrases. Default: true",
                "auto_rewrite_metatalk": "Boolean. Automatically rewrite responses containing meta-talk. Default: true",
                "retrieve_memories": "Boolean. Retrieve relevant memories for context. Default: true",
                "memory_top_k": "Integer. Number of memories to retrieve. Default: 5"
            },
            "temperature": 0.9,
            "top_k": None,
            "top_p": 0.92,
            "presence_penalty": 0.6,
            "frequency_penalty": 0.3,
            "max_tokens": None,
            "enable_anti_metatalk": True,
            "auto_rewrite_metatalk": True,
            "retrieve_memories": True,
            "memory_top_k": 5
        }

        # Ensure meta directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write config
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

        logger.info(f"Created default llm_config.json at {self.config_path}")


def create_persona_config_loader(persona_space_path: str) -> PersonaConfigLoader:
    """
    Factory function to create a PersonaConfigLoader.

    Args:
        persona_space_path: Path to persona_space directory

    Returns:
        PersonaConfigLoader instance
    """
    return PersonaConfigLoader(persona_space_path)
