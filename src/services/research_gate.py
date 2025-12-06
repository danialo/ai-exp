"""
Research Gate - Deterministic decision layer for research tool execution.

This module decides WHETHER research is needed BEFORE the model generates prose.
The model's job is synthesis, not decision-making about when to fetch data.

Architecture:
    User Message → requires_research() → [Research tools run] → Model synthesizes

    NOT:
    User Message → Model decides → Maybe calls tools → Maybe announces instead

The gate uses a two-stage approach:
1. Fast heuristics (no LLM call) - handles 95% of cases
2. Lightweight classifier (cheap LLM call) - for ambiguous cases only
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ResearchDecision(Enum):
    """Result of research gate decision."""
    NO_RESEARCH = "no_research"           # Definitely don't need research
    RESEARCH_REQUIRED = "research_required"  # Definitely need research
    AMBIGUOUS = "ambiguous"                # Need classifier to decide


@dataclass
class GateResult:
    """Result of research gate evaluation."""
    needs_research: bool
    decision: ResearchDecision
    reason: str
    classifier_used: bool = False
    confidence: float = 1.0


class ResearchGate:
    """
    Deterministic research decision gate.

    Makes the decision BEFORE the model generates prose, eliminating
    the possibility of "I'll proceed with research" announcements.
    """

    # === EXPLICIT RESEARCH TRIGGERS ===
    # These ALWAYS trigger research (unless meta-blocked)
    EXPLICIT_RESEARCH_TRIGGERS = [
        r"\bresearch\b",
        r"\binvestigate\b",
        r"\bdig\s+into\b",
        r"\blook\s+into\b",
        r"\bfind\s+out\s+about\b",
        r"\bfull\s+investigation\b",
        r"\bcomprehensive\s+analysis\b",
        r"\bdeep\s+dive\b",
        r"\bin-?depth\s+analysis\b",
    ]

    # === CURRENT EVENTS / FACTUAL TRIGGERS ===
    # These suggest research MAY be needed (depends on context)
    CURRENT_EVENTS_PATTERNS = [
        r"\bwhat('s|\s+is)\s+happening\s+(with|to|in)\b",
        r"\bwhat\s+happened\s+(with|to|in)\b",
        r"\blatest\s+(news|developments?|updates?)\s+(on|about|with)\b",
        r"\bcurrent\s+(events?|situation|state)\s+(of|in|with|regarding)\b",
        r"\brecent\s+(news|developments?|events?)\s+(on|about|with|in)\b",
        r"\bupdate\s+(on|about|me\s+on)\b",
        r"\bwhat('s|\s+is)\s+going\s+on\s+with\b",
        r"\bwhat('s|\s+is)\s+the\s+deal\s+with\b",
        r"\btell\s+me\s+about\s+the\s+(latest|recent|current)\b",
        r"\byesterday('s)?\b",  # Time-sensitive
        r"\btoday('s)?\b",
        r"\bthis\s+(week|month|year)\b",
        r"\blast\s+(week|month|year)\b",
    ]

    # === EXPLICIT NO-RESEARCH PATTERNS ===
    # These should NEVER trigger research
    NO_RESEARCH_PATTERNS = [
        # Questions about capabilities/how-to
        r"\bhow\s+do\s+(I|you)\b",
        r"\bhow\s+can\s+(I|you)\b",
        r"\bcan\s+you\s+(help|assist|show|explain|teach)\b",
        r"\bhelp\s+me\s+(understand|write|code|create|build|fix|debug)\b",
        r"\bexplain\s+(how|what|why|the)\b",

        # Math/calculations
        r"\bcalculate\b",
        r"\bcompute\b",
        r"\bwhat\s+is\s+\d+\s*[+\-*/]\s*\d+\b",

        # Code-related
        r"\bwrite\s+(a|me|some|the)?\s*(code|function|script|program)\b",
        r"\bfix\s+(this|my|the)\s*(code|bug|error|issue)\b",
        r"\bdebug\b",
        r"\brefactor\b",

        # Definition questions (use knowledge, not search)
        r"\bwhat\s+is\s+(a|an|the)\s+\w+\b",  # "what is a function"
        r"\bdefine\s+\b",
        r"\bwhat\s+does\s+\w+\s+mean\b",

        # Conversational
        r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|thanks|thank\s+you)\b",
        r"\bhow\s+are\s+you\b",

        # Opinion/advice (no research needed)
        r"\bwhat\s+do\s+you\s+think\b",
        r"\bwhat('s|\s+is)\s+your\s+opinion\b",
        r"\bdo\s+you\s+recommend\b",
    ]

    # === META PHRASES ===
    # Talking ABOUT research capabilities, not requesting research
    META_PHRASES = [
        "research system",
        "research subsystem",
        "research ability",
        "research capabilities",
        "research tools",
        "research engine",
        "research htn",
        "research module",
        "your research",
        "how does your research",
        "can you research",  # Asking about capability, not requesting
        "do you have research",
    ]

    def __init__(self, llm_service=None, use_classifier: bool = True):
        """
        Initialize the research gate.

        Args:
            llm_service: Optional LLM service for classifier fallback
            use_classifier: Whether to use LLM classifier for ambiguous cases
        """
        self.llm_service = llm_service
        self.use_classifier = use_classifier and llm_service is not None

        # Compile regex patterns for performance
        self._explicit_triggers = [re.compile(p, re.IGNORECASE) for p in self.EXPLICIT_RESEARCH_TRIGGERS]
        self._current_events = [re.compile(p, re.IGNORECASE) for p in self.CURRENT_EVENTS_PATTERNS]
        self._no_research = [re.compile(p, re.IGNORECASE) for p in self.NO_RESEARCH_PATTERNS]

    def requires_research(self, message: str, context: Optional[Dict[str, Any]] = None) -> GateResult:
        """
        Determine if the message requires research.

        This is the main entry point. It runs fast heuristics first,
        then falls back to the classifier only for ambiguous cases.

        Args:
            message: User message to evaluate
            context: Optional context (conversation history, etc.)

        Returns:
            GateResult with decision and reasoning
        """
        text = message.lower().strip()

        # Stage 1: Fast heuristics
        heuristic_result = self._evaluate_heuristics(text, message)

        if heuristic_result.decision != ResearchDecision.AMBIGUOUS:
            logger.info(f"Research gate (heuristics): {heuristic_result.decision.value} - {heuristic_result.reason}")
            return heuristic_result

        # Stage 2: Classifier for ambiguous cases
        if self.use_classifier:
            classifier_result = self._run_classifier(message, context)
            logger.info(f"Research gate (classifier): {classifier_result.decision.value} - {classifier_result.reason}")
            return classifier_result

        # No classifier available, default to no research for ambiguous cases
        # (conservative - let the model use search_web tool if needed)
        return GateResult(
            needs_research=False,
            decision=ResearchDecision.NO_RESEARCH,
            reason="Ambiguous query, no classifier available - defaulting to no research",
            confidence=0.5
        )

    def _evaluate_heuristics(self, text_lower: str, original_text: str) -> GateResult:
        """
        Fast heuristic evaluation (no LLM call).

        Returns:
            GateResult with decision (may be AMBIGUOUS)
        """
        # Check 1: Meta phrases (talking ABOUT research)
        if any(phrase in text_lower for phrase in self.META_PHRASES):
            return GateResult(
                needs_research=False,
                decision=ResearchDecision.NO_RESEARCH,
                reason="Meta question about research capabilities",
                confidence=0.95
            )

        # Check 2: Explicit NO-research patterns
        for pattern in self._no_research:
            if pattern.search(text_lower):
                return GateResult(
                    needs_research=False,
                    decision=ResearchDecision.NO_RESEARCH,
                    reason=f"Matched no-research pattern: {pattern.pattern}",
                    confidence=0.9
                )

        # Check 3: Explicit research triggers
        for pattern in self._explicit_triggers:
            if pattern.search(text_lower):
                return GateResult(
                    needs_research=True,
                    decision=ResearchDecision.RESEARCH_REQUIRED,
                    reason=f"Matched explicit research trigger: {pattern.pattern}",
                    confidence=0.95
                )

        # Check 4: Current events patterns (strong signal but not definitive)
        current_events_matches = [p for p in self._current_events if p.search(text_lower)]
        if current_events_matches:
            # Current events + named entities = likely research
            # Check for proper nouns / named entities (simple heuristic)
            has_proper_noun = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', original_text))

            if has_proper_noun:
                return GateResult(
                    needs_research=True,
                    decision=ResearchDecision.RESEARCH_REQUIRED,
                    reason=f"Current events query with named entity: {current_events_matches[0].pattern}",
                    confidence=0.85
                )
            else:
                # Current events but no named entity - ambiguous
                return GateResult(
                    needs_research=False,
                    decision=ResearchDecision.AMBIGUOUS,
                    reason=f"Current events pattern without clear named entity",
                    confidence=0.5
                )

        # Check 5: Length and complexity heuristics
        word_count = len(text_lower.split())
        has_question_mark = '?' in original_text

        # Very short messages are unlikely to need research
        if word_count <= 3 and not has_question_mark:
            return GateResult(
                needs_research=False,
                decision=ResearchDecision.NO_RESEARCH,
                reason="Very short message without question",
                confidence=0.7
            )

        # Default: ambiguous
        return GateResult(
            needs_research=False,
            decision=ResearchDecision.AMBIGUOUS,
            reason="No strong signals in either direction",
            confidence=0.5
        )

    def _run_classifier(self, message: str, context: Optional[Dict[str, Any]] = None) -> GateResult:
        """
        Run lightweight LLM classifier for ambiguous cases.

        Uses a minimal prompt to classify research intent.
        This should be a fast, cheap call (low tokens).
        """
        if not self.llm_service:
            return GateResult(
                needs_research=False,
                decision=ResearchDecision.NO_RESEARCH,
                reason="No LLM service for classification",
                confidence=0.5,
                classifier_used=False
            )

        # Minimal classification prompt
        classifier_prompt = """You are a classifier. Determine if the following user message requires web research to answer properly.

Research is needed when:
- The question asks about recent/current events, news, or developments
- The answer depends on up-to-date factual information
- Multiple sources would need to be consulted

Research is NOT needed when:
- The question is about general concepts, definitions, or how-to
- The question is conversational, personal, or opinion-seeking
- The question can be answered from general knowledge

Message: "{message}"

Respond with ONLY one word: YES or NO"""

        try:
            result = self.llm_service.generate(
                prompt=classifier_prompt.format(message=message),
                temperature=0.0,  # Deterministic
                max_tokens=5,     # Just need YES/NO
            )

            response = result.strip().upper()
            needs_research = response.startswith("YES")

            return GateResult(
                needs_research=needs_research,
                decision=ResearchDecision.RESEARCH_REQUIRED if needs_research else ResearchDecision.NO_RESEARCH,
                reason=f"Classifier decision: {response}",
                confidence=0.8,
                classifier_used=True
            )

        except Exception as e:
            logger.warning(f"Classifier failed: {e}, defaulting to no research")
            return GateResult(
                needs_research=False,
                decision=ResearchDecision.NO_RESEARCH,
                reason=f"Classifier error: {e}",
                confidence=0.3,
                classifier_used=True
            )

    def extract_topic(self, message: str) -> str:
        """
        Extract the research topic from a message.

        Used to formulate the research question when research is needed.

        Args:
            message: User message

        Returns:
            Extracted topic string suitable for research
        """
        # Remove common question prefixes
        topic = message.strip()

        prefixes_to_remove = [
            r"^(hey\s+astra[,!:]?\s*)",
            r"^(can\s+you|could\s+you|would\s+you|please)\s*",
            r"^(research|investigate|look\s+into|find\s+out\s+about)\s*",
            r"^(what('s|\s+is)\s+(happening|going\s+on)\s+(with|in|to))\s*",
            r"^(what\s+happened\s+(with|to))\s*",
            r"^(tell\s+me\s+about)\s*",
            r"^(update\s+(me\s+)?on)\s*",
        ]

        for prefix in prefixes_to_remove:
            topic = re.sub(prefix, "", topic, flags=re.IGNORECASE).strip()

        # Clean up punctuation at the end
        topic = topic.rstrip('?!.,')

        return topic if topic else message


# Module-level convenience function
_default_gate: Optional[ResearchGate] = None

def get_research_gate(llm_service=None) -> ResearchGate:
    """Get or create the default research gate instance."""
    global _default_gate
    if _default_gate is None or (llm_service is not None and _default_gate.llm_service is None):
        _default_gate = ResearchGate(llm_service=llm_service)
    return _default_gate


def requires_research(message: str, context: Optional[Dict[str, Any]] = None, llm_service=None) -> bool:
    """
    Convenience function to check if research is required.

    Args:
        message: User message
        context: Optional context
        llm_service: Optional LLM service for classifier

    Returns:
        True if research is required, False otherwise
    """
    gate = get_research_gate(llm_service)
    result = gate.requires_research(message, context)
    return result.needs_research
