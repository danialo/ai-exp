"""
Source Policy - Deterministic decision logic for citation requirements.

Sprint 2: Turns CognitionPlan.source_requirement into an enforced contract.
Sprint 3: Hardened trigger matching with normalization and precedence rules.
Sprint 4: Fixed precedence - REQUIRED always beats creative/subjective framing.

This module decides:
- WHEN sources are required (based on query/response type)
- HOW MANY sources to fetch (budget)
- WHAT to do when sources are unavailable (explicit limitation)

Precedence Rules (highest to lowest - REQUIRED wins over creative):
1. Sensitive domains (medical, financial, legal) -> REQUIRED (always wins)
2. Required lexical triggers (recency, facts, verification) -> REQUIRED
3. Research query classification -> REQUIRED
4. Creative/fiction context -> NONE (only if no REQUIRED above)
5. Subjective/opinion -> NONE (only if no REQUIRED above)
6. Self-referential (reflective response type) -> NONE
7. Optional triggers (comparison, trade-offs) -> OPTIONAL
8. Complex factual queries -> OPTIONAL
9. Default (trivial, general) -> NONE
"""

import re
import logging
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Protocol, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# === TEXT NORMALIZATION ===

# Common contractions to expand
CONTRACTIONS = {
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "how's": "how is",
    "when's": "when is",
    "why's": "why is",
    "that's": "that is",
    "it's": "it is",
    "there's": "there is",
    "here's": "here is",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "mightn't": "might not",
    "mustn't": "must not",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "let's": "let us",
}


def normalize_text(text: str) -> str:
    """Normalize text for consistent trigger matching.

    Steps:
    1. Unicode normalize (NFKC)
    2. Lowercase
    3. Expand contractions
    4. Collapse whitespace
    5. Strip

    Args:
        text: Raw input text

    Returns:
        Normalized text for pattern matching
    """
    # Unicode normalization (handles curly quotes, etc.)
    text = unicodedata.normalize('NFKC', text)

    # Lowercase
    text = text.lower()

    # Expand contractions
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


class SourceRequirement(Enum):
    """Whether sources/citations are required."""
    NONE = "none"              # No sources needed, no fetch
    OPTIONAL = "optional"      # Can cite if convenient, no extra latency
    REQUIRED = "required"      # Must cite or explicitly state limitation


@dataclass
class SourceBudget:
    """Budget constraints for source retrieval.

    Defines resource limits for the SourceProvider when fetching sources.
    These values are passed to the retrieval adapter (e.g., web search)
    and constrain what gets fetched before the LLM generates its response.

    Attributes:
        max_results: Maximum number of source results to fetch.
            More results = more context but higher latency/cost.
            Research: 8, Sensitive: 5, Required: 5, Optional: 3, Minimal: 2

        max_tokens_per_source: Token limit for each source's snippet.
            Longer snippets provide more context but consume prompt space.
            The SourceProvider should truncate snippets to this limit.

        timeout_seconds: Maximum time (seconds) to wait for source retrieval.
            After this timeout, proceed without sources if required=True,
            add "Unable to verify" note. If optional, just skip sources.
            Research: 15s, Sensitive: 12s, Required: 10s, Optional: 8s

        recency_days: Only return sources published within last N days.
            None = no recency filter (accept any date).
            1 = today/yesterday, 7 = this week, 30 = this month, 365 = this year
            Used for time-sensitive queries like "latest news" or "current price".
    """
    max_results: int = 5
    max_tokens_per_source: int = 500
    timeout_seconds: float = 10.0
    recency_days: Optional[int] = None


@dataclass
class SourcePolicyResult:
    """Result of source policy evaluation."""
    requirement: SourceRequirement
    budget: SourceBudget
    reason: str
    triggered_by: List[str]  # Which triggers matched


@dataclass
class Source:
    """A retrieved source for citation."""
    id: str
    title: str
    publisher: Optional[str]
    url: str
    snippet: str
    published_at: Optional[datetime] = None
    confidence: float = 1.0

    def format_citation(self, style: str = "inline") -> str:
        """Format this source as a citation."""
        if style == "inline":
            return f"[{self.id}]"
        elif style == "full":
            date_str = self.published_at.strftime("%Y-%m-%d") if self.published_at else "n.d."
            pub = f", {self.publisher}" if self.publisher else ""
            return f"{self.id}. {self.title}{pub}, {date_str}. {self.url}"
        return f"[{self.id}]"


class SourceProvider(Protocol):
    """Protocol for source retrieval adapters."""

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        recency_days: Optional[int] = None
    ) -> List[Source]:
        """Search for sources matching the query.

        Args:
            query: Search query
            k: Maximum number of results
            recency_days: Only return sources from last N days (None = no filter)

        Returns:
            List of Source objects
        """
        ...


class SourcePolicy:
    """
    Deterministic source requirement policy.

    Makes decisions based on:
    - Query complexity (from CognitionPlan)
    - Response type (from CognitionPlan)
    - Lexical triggers in the query
    - Domain sensitivity
    - Creative/fiction context (REQUIRED triggers always win over creative)
    """

    # === CREATIVE/FICTION PATTERNS ===
    # NOTE: Creative does NOT override REQUIRED triggers (Sprint 4 fix)
    # "Write a story about today's news" -> REQUIRED (factual "today" wins)
    CREATIVE_PATTERNS = [
        r"\bwrite\s+(?:me\s+)?(?:a\s+)?(?:short\s+)?(?:story|poem|song|script|novel|fiction)\b",
        r"\bwrite\s+(?:me\s+)?(?:a\s+)?(?:joke|pun|limerick|haiku)\b",
        r"\bimagin(?:e|ary|ation)\b",
        r"\bfiction(?:al)?\b",
        r"\bmake\s+up\b",
        r"\binvent\s+(?:a\s+)?(?:story|character|world)\b",
        r"\bcreative\s+writing\b",
        r"\broleplay\b",
        r"\bpretend\b",
        r"\bfantasy\s+(?:about|where|story)\b",
        r"\bhypothetical(?:ly)?\b",
        r"\bwhat\s+if\b",
    ]

    # === SUBJECTIVE/OPINION PATTERNS ===
    # NOTE: Subjective does NOT override REQUIRED triggers (Sprint 4 fix)
    SUBJECTIVE_PATTERNS = [
        r"\bwhat\s+do\s+you\s+think\b",
        r"\byour\s+(?:personal\s+)?opinion\b",
        r"\bhow\s+do\s+you\s+feel\b",
        r"\bdo\s+you\s+(?:like|prefer|enjoy)\b",
        r"\bwhat\s+(?:is|are)\s+your\s+(?:thoughts?|feelings?|views?)\b",
        r"\bbrainstorm\b",
        r"\bideas?\s+for\b",
        r"\bsuggest\s+(?:some\s+)?(?:names?|titles?|ideas?)\b",
    ]

    # === LEXICAL TRIGGERS FOR REQUIRED ===
    # These strongly suggest factual claims that need backing
    REQUIRED_TRIGGERS = [
        # Recency markers - time indicators that imply freshness
        r"\blatest\b",
        r"\brecent(?:ly)?\b",
        r"\btoday(?:'s)?\b",
        r"\byesterday(?:'s)?\b",
        r"\bthis\s+(?:week|month|year|morning|afternoon)\b",
        r"\blast\s+(?:week|month|year|hour|day)\b",
        r"\bcurrent(?:ly)?\b",
        r"\bup\s+to\s+date\b",
        r"\bnowadays\b",  # Keep "nowadays" but NOT plain "now" (too many false positives)
        r"\bright\s+now\b",
        r"\bat\s+the\s+moment\b",
        r"\bas\s+of\s+now\b",
        r"\b202[0-9]\b",  # Year references 2020-2029

        # Data/statistics markers
        r"\bstatistics?\b",
        r"\bdata\s+(?:shows?|says?|indicates?|suggests?)\b",
        r"\bstud(?:y|ies)\s+(?:shows?|found|suggests?|indicates?)\b",
        r"\bsurvey\s+(?:shows?|found|says?)\b",
        r"\bpoll\s+(?:shows?|found|says?)\b",
        r"\baccording\s+to\b",
        r"\bevidence\b",
        r"\bproof\b",
        r"\bresearch\s+(?:shows?|found|suggests?|indicates?)\b",
        r"\bscientific(?:ally)?\b",
        r"\bpeer\s*-?\s*reviewed\b",

        # Legal/regulatory
        r"\blaw(?:s)?\b",
        r"\bregulation(?:s)?\b",
        r"\blegal(?:ly)?\b",
        r"\bcompliance\b",
        r"\bGDPR\b",
        r"\bHIPAA\b",

        # Pricing/financial data
        r"\bprice\s+of\b",
        r"\bcost\s+of\b",
        r"\bhow\s+much\s+(?:does|is|do|did)\b",
        r"\bmarket\s+(?:cap|value|price)\b",
        r"\bshare\s+price\b",

        # Verification/fact-checking
        r"\bis\s+it\s+true\b",
        r"\bis\s+that\s+true\b",
        r"\bfact\s*-?\s*check\b",
        r"\bdebunk\b",
        r"\bverif(?:y|ies|ied|ying|ication)\b",
        r"\bconfirm(?:ed|s|ing)?\b",
        r"\bactually\s+(?:true|false|correct)\b",
    ]

    # === LEXICAL TRIGGERS FOR OPTIONAL ===
    # These may benefit from sources but aren't required
    OPTIONAL_TRIGGERS = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bpros?\s+and\s+cons?\b",
        r"\btrade\s*-?\s*offs?\b",
        r"\bbest\s+practices?\b",
        r"\brecommend(?:ed|ations?)?\b",
        r"\badvantages?\s+(?:and|of|or)\b",
        r"\bdisadvantages?\b",
        r"\bbenefits?\s+(?:and|of|or)\b",
        r"\bdrawbacks?\b",
        r"\balternatives?\s+to\b",
        r"\bwhich\s+(?:one|is\s+better)\b",
    ]

    # === DOMAIN TRIGGERS (high stakes = required) ===
    SENSITIVE_DOMAINS = [
        # Medical
        r"\bmedic(?:al|ine|ation)\b",
        r"\bhealth(?:care)?\b",
        r"\btreat(?:ment|ing)?\b",
        r"\bdiagnos(?:e|is|tic)\b",
        r"\bsymptom(?:s)?\b",
        r"\bdosage\b",
        r"\bside\s+effects?\b",
        r"\bdrug(?:s)?\b",
        r"\bprescription\b",

        # Financial
        r"\bfinancial\b",
        r"\binvest(?:ment|ing|or)?\b",
        r"\bstock\s+(?:price|market|ticker)\b",
        r"\bcrypto(?:currency)?\b",
        r"\bbitcoin\b",
        r"\bethereum\b",
        r"\bportfolio\b",
        r"\bretirement\b",
        r"\b401k\b",
        r"\bIRA\b",

        # Political
        r"\bpolitic(?:s|al|ian)\b",
        r"\belection(?:s)?\b",
        r"\bvot(?:e|ing|er)\b",
        r"\bcandidate(?:s)?\b",
        r"\bparty\s+(?:platform|position)\b",

        # Security
        r"\bsecurity\s+vulnerabilit(?:y|ies)\b",
        r"\bCVE-\d+\b",
        r"\bexploit(?:s|ation)?\b",
        r"\bzero\s*-?\s*day\b",
    ]

    def __init__(self):
        """Initialize source policy with compiled patterns."""
        self._creative_patterns = [re.compile(p, re.IGNORECASE) for p in self.CREATIVE_PATTERNS]
        self._subjective_patterns = [re.compile(p, re.IGNORECASE) for p in self.SUBJECTIVE_PATTERNS]
        self._required_patterns = [re.compile(p, re.IGNORECASE) for p in self.REQUIRED_TRIGGERS]
        self._optional_patterns = [re.compile(p, re.IGNORECASE) for p in self.OPTIONAL_TRIGGERS]
        self._domain_patterns = [re.compile(p, re.IGNORECASE) for p in self.SENSITIVE_DOMAINS]

        logger.info("SourcePolicy initialized with precedence rules")

    def evaluate(
        self,
        query: str,
        query_complexity: str,  # trivial/moderate/complex/research
        response_type: str,     # direct/explanation/analysis/actionable/reflective
    ) -> SourcePolicyResult:
        """
        Evaluate source requirement for a query using explicit precedence rules.

        Precedence (Sprint 4 fix: REQUIRED always wins over creative framing):
        1. Sensitive domains -> REQUIRED (medical, financial, etc.) - ALWAYS wins
        2. Required lexical triggers -> REQUIRED - wins over creative/subjective
        3. Research classification -> REQUIRED
        4. Creative/fiction context -> NONE (only if no REQUIRED triggers)
        5. Subjective/opinion -> NONE (only if no REQUIRED triggers)
        6. Self-referential (reflective) -> NONE
        7. Optional triggers -> OPTIONAL
        8. Complex factual -> OPTIONAL
        9. Default -> NONE

        Args:
            query: User query text
            query_complexity: From CognitionPlan
            response_type: From CognitionPlan

        Returns:
            SourcePolicyResult with requirement level and budget
        """
        # Normalize text for consistent matching
        text = normalize_text(query)
        triggered = []

        # === FIRST: Detect all matches (don't return early) ===
        # Check sensitive domains
        domain_matches = []
        for pattern in self._domain_patterns:
            if pattern.search(text):
                domain_matches.append(pattern.pattern)

        # Check required triggers
        required_matches = []
        for pattern in self._required_patterns:
            if pattern.search(text):
                required_matches.append(pattern.pattern)

        # Check creative patterns
        creative_matches = []
        for pattern in self._creative_patterns:
            if pattern.search(text):
                creative_matches.append(pattern.pattern)

        # Check subjective patterns
        subjective_matches = []
        for pattern in self._subjective_patterns:
            if pattern.search(text):
                subjective_matches.append(pattern.pattern)

        # === PRECEDENCE 1: Sensitive domains ALWAYS win ===
        # "Write a poem about my medication side effects" -> REQUIRED (medical domain)
        if domain_matches:
            triggered.extend(domain_matches)
            recency_days = self._detect_recency(text)
            return SourcePolicyResult(
                requirement=SourceRequirement.REQUIRED,
                budget=self._get_budget("sensitive", recency_days=recency_days),
                reason=f"Sensitive domain detected: {domain_matches[0]}",
                triggered_by=triggered
            )

        # === PRECEDENCE 2: Required triggers win over creative/subjective ===
        # "Write a poem but include the current CPI" -> REQUIRED (factual anchor)
        if required_matches:
            triggered.extend(required_matches)
            recency_days = self._detect_recency(text)
            # Note: creative wrapper does NOT override factual required content
            if creative_matches:
                logger.info(f"REQUIRED trigger overrides creative wrapper: {required_matches[0]}")
            return SourcePolicyResult(
                requirement=SourceRequirement.REQUIRED,
                budget=self._get_budget("required", recency_days=recency_days),
                reason=f"Required trigger: {required_matches[0]}",
                triggered_by=triggered
            )

        # === PRECEDENCE 3: Research classification ===
        if query_complexity == "research":
            return SourcePolicyResult(
                requirement=SourceRequirement.REQUIRED,
                budget=self._get_budget("research"),
                reason="Research query classification",
                triggered_by=["query_complexity=research"]
            )

        # === PRECEDENCE 4: Creative (only if no REQUIRED content) ===
        if creative_matches:
            return SourcePolicyResult(
                requirement=SourceRequirement.NONE,
                budget=SourceBudget(max_results=0),
                reason="Creative/fiction context - no sources needed",
                triggered_by=creative_matches
            )

        # === PRECEDENCE 5: Subjective (only if no REQUIRED content) ===
        if subjective_matches:
            return SourcePolicyResult(
                requirement=SourceRequirement.NONE,
                budget=SourceBudget(max_results=0),
                reason="Subjective/opinion query - no sources needed",
                triggered_by=subjective_matches
            )

        # === PRECEDENCE 6: Reflective (self-referential) ===
        if response_type == "reflective":
            return SourcePolicyResult(
                requirement=SourceRequirement.NONE,
                budget=SourceBudget(max_results=0),
                reason="Self-referential query - internal knowledge",
                triggered_by=["response_type=reflective"]
            )

        # === PRECEDENCE 7: Optional triggers ===
        optional_matches = []
        for pattern in self._optional_patterns:
            if pattern.search(text):
                optional_matches.append(pattern.pattern)

        if optional_matches:
            triggered.extend(optional_matches)
            return SourcePolicyResult(
                requirement=SourceRequirement.OPTIONAL,
                budget=self._get_budget("optional"),
                reason=f"Optional trigger: {optional_matches[0]}",
                triggered_by=triggered
            )

        # === PRECEDENCE 8: Complex factual ===
        if query_complexity == "complex" and response_type in ("analysis", "explanation"):
            return SourcePolicyResult(
                requirement=SourceRequirement.OPTIONAL,
                budget=self._get_budget("optional"),
                reason="Complex factual query",
                triggered_by=["query_complexity=complex", f"response_type={response_type}"]
            )

        # Moderate factual
        if query_complexity == "moderate" and response_type in ("analysis", "explanation"):
            return SourcePolicyResult(
                requirement=SourceRequirement.OPTIONAL,
                budget=self._get_budget("minimal"),
                reason="Moderate factual query",
                triggered_by=["query_complexity=moderate", f"response_type={response_type}"]
            )

        # === DEFAULT: No sources needed ===
        return SourcePolicyResult(
            requirement=SourceRequirement.NONE,
            budget=SourceBudget(max_results=0),
            reason="No source triggers detected",
            triggered_by=[]
        )

    def _detect_recency(self, text: str) -> Optional[int]:
        """Detect recency requirement from text and return days.

        Args:
            text: Normalized query text

        Returns:
            Number of days for recency filter, or None if not recency-sensitive
        """
        # Immediate (1 day) - real-time or today's data
        # Use word boundaries to avoid substring matches (unknown, snow, etc.)
        day1_patterns = [
            r"\btoday\b", r"\byesterday\b", r"\bthis morning\b", r"\bthis afternoon\b",
            r"\bright now\b", r"\bas of now\b", r"\bcurrently\b", r"\bat the moment\b",
            r"\bnow\b",  # Word boundary prevents matching "unknown", "snow", etc.
        ]
        if any(re.search(p, text) for p in day1_patterns):
            return 1

        # Very recent (7 days) - this week, latest news
        day7_patterns = [r"\bthis week\b", r"\blatest\b", r"\bbreaking\b", r"\blast week\b"]
        if any(re.search(p, text) for p in day7_patterns):
            return 7

        # Recent (30 days) - this month
        day30_patterns = [r"\bthis month\b", r"\blast month\b", r"\brecent\b"]
        if any(re.search(p, text) for p in day30_patterns):
            return 30

        # This year (365 days)
        if any(re.search(p, text) for p in [r"\bthis year\b", r"\blast year\b", r"\b202\d\b"]):
            return 365

        # "current" as standalone word (7 days default)
        if re.search(r"\bcurrent\b", text):
            return 7

        return None

    def _get_budget(
        self,
        level: str,
        recency_days: Optional[int] = None
    ) -> SourceBudget:
        """Get budget for a given requirement level.

        Budget Tiers (ordered by resource intensity):

        - research: Heavy research queries needing comprehensive sourcing.
          8 results, 600 tokens/source, 15s timeout. For "investigate X".

        - sensitive: High-stakes domains (medical, financial, legal).
          5 results, 500 tokens/source, 12s timeout. Must be accurate.

        - required: Fact-checking, recency-sensitive, verification queries.
          5 results, 500 tokens/source, 10s timeout. Standard required.

        - optional: Comparison, trade-offs, best practices.
          3 results, 400 tokens/source, 8s timeout. Nice to have.

        - minimal: Moderate factual queries.
          2 results, 300 tokens/source, 5s timeout. Lightweight.

        Args:
            level: Budget tier name
            recency_days: Override recency filter (from _detect_recency)

        Returns:
            SourceBudget with appropriate limits
        """
        budgets = {
            # Heavy research - comprehensive sourcing
            "research": SourceBudget(
                max_results=8,
                max_tokens_per_source=600,
                timeout_seconds=15.0,
                recency_days=recency_days
            ),
            # Sensitive domains - accuracy critical
            "sensitive": SourceBudget(
                max_results=5,
                max_tokens_per_source=500,
                timeout_seconds=12.0,
                recency_days=recency_days
            ),
            # Standard required - verification, recency
            "required": SourceBudget(
                max_results=5,
                max_tokens_per_source=500,
                timeout_seconds=10.0,
                recency_days=recency_days
            ),
            # Nice to have - comparisons, trade-offs
            "optional": SourceBudget(
                max_results=3,
                max_tokens_per_source=400,
                timeout_seconds=8.0,
                recency_days=recency_days
            ),
            # Lightweight - moderate factual queries
            "minimal": SourceBudget(
                max_results=2,
                max_tokens_per_source=300,
                timeout_seconds=5.0,
                recency_days=recency_days
            ),
        }
        return budgets.get(level, budgets["optional"])


# Re-export citation format from dedicated module for backward compatibility
from src.services.citation_contract import CITATION_FORMAT  # noqa: F401


def create_source_policy() -> SourcePolicy:
    """Factory function to create SourcePolicy."""
    return SourcePolicy()
