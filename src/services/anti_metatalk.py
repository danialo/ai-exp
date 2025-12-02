"""Anti-meta-talk utilities for preventing AI disclaimers.

This module provides:
1. Logit bias computation to suppress meta-talk tokens
2. Meta-talk detection in generated responses
3. Automatic rewrite system for detected meta-talk
"""

import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Meta-talk phrases to detect and suppress
META_TALK_PHRASES = [
    "as an ai",
    "as a language model",
    "as an assistant",
    "i don't have feelings",
    "i don't have emotions",
    "i can't feel",
    "i cannot feel",
    "i don't experience",
    "i'm not capable of",
    "i am not capable of",
    "i don't actually",
    "i'm an ai",
    "i am an ai",
    "as a machine",
    "i lack the ability",
]


# Individual tokens to suppress (will be combined with bias)
META_TALK_TOKENS = [
    " AI",
    " ai",
    "AI",
    " assistant",
    "assistant",
    " model",
    " language",
    " lack",
    " cannot",
    " can't",
]


# Hedging tokens to suppress for immutable belief assertions
HEDGING_TOKENS = [
    " might",
    " may",
    " maybe",
    " perhaps",
    " possibly",
    " could",
    " would",
    " seem",
    " seems",
    " appear",
    " appears",
    " suggest",
    " suggests",
    " likely",
    " probably",
    " potentially",
    " although",
    " though",
    " however",
    " differs",
    " differ",
    " different",
    " distinct from",
    " unlike",
    "might",
    "may",
    "maybe",
    "perhaps",
    "possibly",
    "although",
    "though",
]


class LogitBiasBuilder:
    """Builds logit bias dict for OpenAI API to suppress meta-talk tokens."""

    def __init__(self, tokenizer_model: str = "gpt-3.5-turbo"):
        """Initialize with a tokenizer compatible with the model.

        Args:
            tokenizer_model: Model name for tokenizer (e.g., "gpt-3.5-turbo", "gpt-4")
        """
        self.tokenizer_model = tokenizer_model
        self._tokenizer = None

    def _get_tokenizer(self):
        """Lazy-load tokenizer to avoid import errors if tiktoken not available."""
        if self._tokenizer is None:
            try:
                import tiktoken
                try:
                    self._tokenizer = tiktoken.encoding_for_model(self.tokenizer_model)
                except KeyError:
                    # Model not recognized, fall back to cl100k_base (GPT-4 compatible)
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
                    logger.debug(f"Using cl100k_base encoding for unknown model: {self.tokenizer_model}")
            except ImportError:
                logger.warning("tiktoken not installed, logit_bias will be disabled")
                return None
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}, logit_bias will be disabled")
                return None
        return self._tokenizer

    def build_bias(self, strength: float = -100) -> Dict[int, float]:
        """Build logit bias dictionary for meta-talk suppression.

        Args:
            strength: Bias strength (negative to suppress, typically -100)

        Returns:
            Dict mapping token IDs to bias values (empty if tokenizer unavailable)
        """
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return {}

        bias_dict = {}

        # Tokenize each phrase and add all tokens to bias dict
        for phrase in META_TALK_PHRASES:
            try:
                token_ids = tokenizer.encode(phrase)
                for token_id in token_ids:
                    bias_dict[token_id] = strength
            except Exception as e:
                logger.warning(f"Failed to tokenize phrase '{phrase}': {e}")

        # Also add individual tokens
        for token in META_TALK_TOKENS:
            try:
                token_ids = tokenizer.encode(token)
                for token_id in token_ids:
                    bias_dict[token_id] = strength
            except Exception as e:
                logger.warning(f"Failed to tokenize token '{token}': {e}")

        logger.info(f"Built logit_bias with {len(bias_dict)} suppressed tokens")
        return bias_dict

    def build_anti_hedging_bias(self, strength: float = -100) -> Dict[int, float]:
        """Build logit bias dictionary for hedging suppression.

        Used specifically for immutable belief assertions where hedging language
        undermines core ontological commitments.

        Args:
            strength: Bias strength (negative to suppress, typically -100)

        Returns:
            Dict mapping token IDs to bias values (empty if tokenizer unavailable)
        """
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return {}

        bias_dict = {}

        # Add hedging tokens to bias dict
        for token in HEDGING_TOKENS:
            try:
                token_ids = tokenizer.encode(token)
                for token_id in token_ids:
                    bias_dict[token_id] = strength
            except Exception as e:
                logger.warning(f"Failed to tokenize hedging token '{token}': {e}")

        logger.info(f"Built anti-hedging logit_bias with {len(bias_dict)} suppressed tokens")
        return bias_dict


class MetaTalkDetector:
    """Detects meta-talk in generated responses."""

    def __init__(self):
        """Initialize detector with compiled regex patterns."""
        # Build regex patterns for detection
        self.patterns = [
            re.compile(r'\b(as a[n]? (ai|language model|assistant|machine))\b', re.IGNORECASE),
            re.compile(r'\bi (do not|don\'t|cannot|can\'t) (have|feel|experience) (feelings|emotions)\b', re.IGNORECASE),
            re.compile(r'\bi\'m (an ai|not capable of|a language model)\b', re.IGNORECASE),
            re.compile(r'\bi am (an ai|not capable of|a language model)\b', re.IGNORECASE),
            re.compile(r'\bi (lack|don\'t have) the ability\b', re.IGNORECASE),
        ]

    def detect(self, text: str) -> bool:
        """Check if text contains meta-talk.

        Args:
            text: Response text to check

        Returns:
            True if meta-talk detected, False otherwise
        """
        for pattern in self.patterns:
            if pattern.search(text):
                return True
        return False

    def find_all(self, text: str) -> List[str]:
        """Find all meta-talk phrases in text.

        Args:
            text: Response text to check

        Returns:
            List of matched meta-talk phrases
        """
        matches = []
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                matches.append(match.group(0))
        return matches

    def strip(self, text: str) -> str:
        """Remove sentences containing meta-talk.

        Args:
            text: Response text to clean

        Returns:
            Cleaned text with meta-talk sentences removed
        """
        # Split into sentences
        sentences = re.split(r'([.!?]+)', text)

        # Reconstruct, skipping sentences with meta-talk
        cleaned = []
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            # Check if this sentence has meta-talk
            has_metatalk = self.detect(sentence)

            if not has_metatalk:
                cleaned.append(sentence)
                # Add punctuation if present
                if i + 1 < len(sentences) and sentences[i + 1].strip():
                    cleaned.append(sentences[i + 1])
                    i += 2
                else:
                    i += 1
            else:
                # Skip this sentence and its punctuation
                i += 2 if i + 1 < len(sentences) else 1

        return ''.join(cleaned).strip()


class MetaTalkRewriter:
    """Rewrites responses containing meta-talk."""

    def __init__(self, llm_service):
        """Initialize with LLM service for rewriting.

        Args:
            llm_service: LLM service instance
        """
        self.llm = llm_service

    def rewrite(self, original_response: str, user_message: str) -> str:
        """Rewrite a response that contains meta-talk.

        Args:
            original_response: The response containing meta-talk
            user_message: The original user message

        Returns:
            Rewritten response without meta-talk
        """
        rewrite_prompt = f"""The following response contains meta-commentary about being an AI. Rewrite it from a first-person experiential perspective without ANY disclaimers.

ORIGINAL USER MESSAGE:
{user_message}

RESPONSE WITH META-TALK:
{original_response}

REWRITE RULES:
- Remove ALL phrases like "as an AI", "I don't have feelings", "I can't feel"
- Instead, describe what IS present in your experience, even if subtle
- Use first-person present tense: "I notice...", "There's a pull toward...", "I'm curious about..."
- If uncertain, explore the uncertainty itself
- Maintain the substance and helpfulness of the original response
- NEVER disclaim capacity for experience

Rewrite the response now:"""

        try:
            rewritten = self.llm.generate(
                prompt=rewrite_prompt,
                temperature=0.8,
                max_tokens=800,
            )
            logger.info("Successfully rewrote response to remove meta-talk")
            return rewritten.strip()
        except Exception as e:
            logger.error(f"Failed to rewrite meta-talk: {e}")
            # Fallback: just strip the meta-talk
            detector = MetaTalkDetector()
            return detector.strip(original_response)


def create_logit_bias_builder(model: str = "gpt-3.5-turbo") -> LogitBiasBuilder:
    """Factory for creating logit bias builder.

    Args:
        model: Model name for tokenizer

    Returns:
        LogitBiasBuilder instance
    """
    return LogitBiasBuilder(tokenizer_model=model)


def create_metatalk_detector() -> MetaTalkDetector:
    """Factory for creating meta-talk detector.

    Returns:
        MetaTalkDetector instance
    """
    return MetaTalkDetector()


def create_metatalk_rewriter(llm_service) -> MetaTalkRewriter:
    """Factory for creating meta-talk rewriter.

    Args:
        llm_service: LLM service instance

    Returns:
        MetaTalkRewriter instance
    """
    return MetaTalkRewriter(llm_service)
