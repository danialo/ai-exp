"""
Tests for SourcePolicy precedence rules and trigger matching.

Sprint 3: Validates that precedence rules work correctly.
Sprint 4: Updated precedence - REQUIRED always beats creative/subjective framing.

Key edge cases:
- REQUIRED triggers ALWAYS win over creative framing (Sprint 4 fix)
- Pure creative/subjective queries without REQUIRED triggers -> NONE
- Sensitive domains require sources even for simple queries
- Contractions are expanded correctly
"""

import pytest
from src.services.source_policy import (
    SourcePolicy,
    SourceRequirement,
    SourceBudget,
    SourcePolicyResult,
    normalize_text,
)


@pytest.fixture
def policy():
    """Create a fresh SourcePolicy instance."""
    return SourcePolicy()


class TestNormalizeText:
    """Test text normalization for trigger matching."""

    def test_lowercase(self):
        assert normalize_text("WHAT IS THE LATEST NEWS?") == "what is the latest news?"

    def test_expand_contractions(self):
        # "What's" expands, but "today's" is possessive (not a contraction)
        assert normalize_text("What's the time?") == "what is the time?"
        assert normalize_text("It's true") == "it is true"
        assert normalize_text("Don't do that") == "do not do that"
        assert normalize_text("I'm here") == "i am here"

    def test_collapse_whitespace(self):
        assert normalize_text("hello    world") == "hello world"
        assert normalize_text("  spaced  out  ") == "spaced out"

    def test_unicode_normalization(self):
        # NFKC normalizes various Unicode forms
        # Full-width characters get normalized
        assert normalize_text("\uff37\uff48\uff41\uff54") == "what"  # Full-width "What"
        # Basic text unchanged
        assert normalize_text("test") == "test"


class TestPrecedenceRequiredBeatsCreative:
    """Test that REQUIRED triggers ALWAYS beat creative framing (Sprint 4)."""

    def test_story_with_today_is_required(self, policy):
        """'Write a story about today' triggers REQUIRED due to 'today'."""
        result = policy.evaluate(
            "Write a story about today's news",
            query_complexity="moderate",
            response_type="direct",
        )
        # Sprint 4: REQUIRED trigger "today" beats creative wrapper
        assert result.requirement == SourceRequirement.REQUIRED
        assert "Required" in result.reason or "today" in result.reason.lower()

    def test_poem_with_latest_is_required(self, policy):
        """'Write a poem about the latest trends' triggers REQUIRED due to 'latest'."""
        result = policy.evaluate(
            "Write me a poem about the latest tech trends",
            query_complexity="moderate",
            response_type="direct",
        )
        # Sprint 4: REQUIRED trigger "latest" beats creative wrapper
        assert result.requirement == SourceRequirement.REQUIRED

    def test_imagine_with_today_is_required(self, policy):
        """'Imagine if today...' triggers REQUIRED due to 'today'."""
        result = policy.evaluate(
            "Imagine if today was the last day",
            query_complexity="moderate",
            response_type="direct",
        )
        # Sprint 4: REQUIRED trigger "today" beats creative wrapper
        assert result.requirement == SourceRequirement.REQUIRED

    def test_what_if_with_latest_is_required(self, policy):
        """'What if the latest...' triggers REQUIRED due to 'latest'."""
        result = policy.evaluate(
            "What if the latest AI could solve all problems?",
            query_complexity="moderate",
            response_type="direct",
        )
        # Sprint 4: REQUIRED trigger "latest" beats creative wrapper
        assert result.requirement == SourceRequirement.REQUIRED

    def test_fiction_with_current_is_required(self, policy):
        """'Fiction about current events' triggers REQUIRED due to 'current'."""
        result = policy.evaluate(
            "Write fiction about current events",
            query_complexity="moderate",
            response_type="direct",
        )
        # Sprint 4: REQUIRED trigger "current" beats creative wrapper
        assert result.requirement == SourceRequirement.REQUIRED


class TestPureCreativeIsNone:
    """Test that pure creative WITHOUT REQUIRED triggers returns NONE."""

    def test_pure_story_is_none(self, policy):
        """'Write a story about dragons' should be NONE (no REQUIRED triggers)."""
        result = policy.evaluate(
            "Write a story about dragons",
            query_complexity="moderate",
            response_type="direct",
        )
        assert result.requirement == SourceRequirement.NONE

    def test_pure_poem_is_none(self, policy):
        """'Write a poem about autumn' should be NONE (no REQUIRED triggers)."""
        result = policy.evaluate(
            "Write me a poem about autumn leaves",
            query_complexity="moderate",
            response_type="direct",
        )
        assert result.requirement == SourceRequirement.NONE

    def test_pure_imagine_is_none(self, policy):
        """'Imagine a world...' without REQUIRED triggers should be NONE."""
        result = policy.evaluate(
            "Imagine a world where cats rule",
            query_complexity="moderate",
            response_type="direct",
        )
        assert result.requirement == SourceRequirement.NONE


class TestPrecedenceRequiredBeatsSubjective:
    """Test that REQUIRED triggers ALWAYS beat subjective framing (Sprint 4)."""

    def test_opinion_with_today_is_required(self, policy):
        """'Opinion about today' triggers REQUIRED due to 'today'."""
        result = policy.evaluate(
            "What do you think about today's market?",
            query_complexity="moderate",
            response_type="direct",
        )
        # Sprint 4: REQUIRED trigger "today" beats subjective wrapper
        assert result.requirement == SourceRequirement.REQUIRED

    def test_opinion_with_latest_is_required(self, policy):
        """'Your opinion on latest X' triggers REQUIRED due to 'latest'."""
        result = policy.evaluate(
            "What is your opinion on the latest iPhone?",
            query_complexity="moderate",
            response_type="direct",
        )
        # Sprint 4: REQUIRED trigger "latest" beats subjective wrapper
        assert result.requirement == SourceRequirement.REQUIRED


class TestPureSubjectiveIsNone:
    """Test that pure subjective WITHOUT REQUIRED triggers returns NONE."""

    def test_brainstorm_ideas(self, policy):
        """Brainstorm requests should be NONE (no REQUIRED triggers)."""
        result = policy.evaluate(
            "Brainstorm some ideas for my startup",
            query_complexity="moderate",
            response_type="actionable",
        )
        assert result.requirement == SourceRequirement.NONE

    def test_pure_opinion(self, policy):
        """Pure opinion without REQUIRED triggers should be NONE."""
        result = policy.evaluate(
            "What do you think about pineapple pizza?",
            query_complexity="moderate",
            response_type="direct",
        )
        assert result.requirement == SourceRequirement.NONE


class TestPrecedenceRequiredBeatsReflective:
    """Test that REQUIRED triggers and domains beat reflective type (Sprint 4)."""

    def test_reflective_with_today_is_required(self, policy):
        """Reflective with 'today' trigger should be REQUIRED (Sprint 4)."""
        result = policy.evaluate(
            "What do you believe about today's events?",
            query_complexity="moderate",
            response_type="reflective",  # reflective doesn't override REQUIRED
        )
        # Sprint 4: REQUIRED trigger "today" beats reflective
        assert result.requirement == SourceRequirement.REQUIRED

    def test_reflective_with_domain_is_required(self, policy):
        """Reflective with sensitive domain should be REQUIRED (Sprint 4)."""
        result = policy.evaluate(
            "What are your thoughts on medical ethics?",
            query_complexity="moderate",
            response_type="reflective",
        )
        # Sprint 4: Sensitive domain "medical" beats reflective
        assert result.requirement == SourceRequirement.REQUIRED


class TestPureReflectiveIsNone:
    """Test that pure reflective WITHOUT REQUIRED triggers returns NONE."""

    def test_pure_reflective_is_none(self, policy):
        """Reflective without REQUIRED triggers should be NONE."""
        result = policy.evaluate(
            "What is it like being an AI?",  # No subjective pattern trigger
            query_complexity="moderate",
            response_type="reflective",
        )
        assert result.requirement == SourceRequirement.NONE
        assert "Self-referential" in result.reason


class TestSensitiveDomainsPrecedence:
    """Test that sensitive domains ALWAYS require sources (Sprint 4: domains beat all)."""

    def test_medical_query_requires_sources(self, policy):
        """Medical queries should require sources."""
        result = policy.evaluate(
            "What are the side effects of aspirin?",
            query_complexity="moderate",
            response_type="explanation",
        )
        assert result.requirement == SourceRequirement.REQUIRED
        assert "Sensitive" in result.reason or "sensitive" in result.reason.lower()

    def test_financial_query_requires_sources(self, policy):
        """Financial queries should require sources."""
        result = policy.evaluate(
            "Should I invest in index funds?",
            query_complexity="moderate",
            response_type="analysis",
        )
        assert result.requirement == SourceRequirement.REQUIRED

    def test_political_query_requires_sources(self, policy):
        """Political queries should require sources."""
        result = policy.evaluate(
            "What happened in the last election?",
            query_complexity="moderate",
            response_type="explanation",
        )
        assert result.requirement == SourceRequirement.REQUIRED

    def test_cve_query_requires_sources(self, policy):
        """Security vulnerability queries should require sources."""
        result = policy.evaluate(
            "Tell me about CVE-2024-1234",
            query_complexity="moderate",
            response_type="explanation",
        )
        assert result.requirement == SourceRequirement.REQUIRED


class TestRequiredLexicalTriggers:
    """Test required triggers without creative/subjective context."""

    def test_latest_news_requires_sources(self, policy):
        """'Latest news' without creative context should require sources."""
        result = policy.evaluate(
            "What is the latest news about AI?",
            query_complexity="moderate",
            response_type="explanation",
        )
        assert result.requirement == SourceRequirement.REQUIRED
        assert result.budget.max_results > 0

    def test_is_it_true_requires_sources(self, policy):
        """Fact-checking queries require sources."""
        result = policy.evaluate(
            "Is it true that Python is faster than Ruby?",
            query_complexity="moderate",
            response_type="explanation",
        )
        assert result.requirement == SourceRequirement.REQUIRED

    def test_current_price_requires_sources(self, policy):
        """Current price queries require sources."""
        result = policy.evaluate(
            "What is the current price of Bitcoin?",
            query_complexity="moderate",
            response_type="direct",
        )
        assert result.requirement == SourceRequirement.REQUIRED

    def test_statistics_requires_sources(self, policy):
        """Statistics requests require sources."""
        result = policy.evaluate(
            "What are the statistics on remote work productivity?",
            query_complexity="moderate",
            response_type="analysis",
        )
        assert result.requirement == SourceRequirement.REQUIRED

    def test_according_to_requires_sources(self, policy):
        """'According to' implies attribution needed."""
        result = policy.evaluate(
            "According to recent research, what causes X?",
            query_complexity="moderate",
            response_type="explanation",
        )
        assert result.requirement == SourceRequirement.REQUIRED


class TestRecencyDetection:
    """Test recency days are set correctly."""

    def test_today_sets_1_day(self, policy):
        """'Today' should set recency to 1 day."""
        result = policy.evaluate(
            "What happened today in tech?",
            query_complexity="moderate",
            response_type="explanation",
        )
        assert result.budget.recency_days == 1

    def test_this_week_sets_7_days(self, policy):
        """'This week' should set recency to 7 days."""
        result = policy.evaluate(
            "What happened this week in the markets?",
            query_complexity="moderate",
            response_type="explanation",
        )
        assert result.budget.recency_days == 7

    def test_this_month_sets_30_days(self, policy):
        """'This month' should set recency to 30 days."""
        result = policy.evaluate(
            "What were the major releases this month?",
            query_complexity="moderate",
            response_type="explanation",
        )
        assert result.budget.recency_days == 30

    def test_this_year_sets_365_days(self, policy):
        """'This year' should set recency to 365 days."""
        result = policy.evaluate(
            "What are the biggest tech stories this year?",
            query_complexity="moderate",
            response_type="explanation",
        )
        assert result.budget.recency_days == 365


class TestOptionalTriggers:
    """Test optional triggers return OPTIONAL."""

    def test_compare_is_optional(self, policy):
        """'Compare X and Y' should be OPTIONAL."""
        result = policy.evaluate(
            "Compare React and Vue for web development",
            query_complexity="moderate",
            response_type="analysis",
        )
        assert result.requirement == SourceRequirement.OPTIONAL

    def test_pros_and_cons_is_optional(self, policy):
        """'Pros and cons' should be OPTIONAL."""
        result = policy.evaluate(
            "What are the pros and cons of microservices?",
            query_complexity="moderate",
            response_type="analysis",
        )
        assert result.requirement == SourceRequirement.OPTIONAL

    def test_tradeoffs_is_optional(self, policy):
        """'Trade-offs' should be OPTIONAL."""
        result = policy.evaluate(
            "What are the trade-offs of using NoSQL?",
            query_complexity="moderate",
            response_type="analysis",
        )
        assert result.requirement == SourceRequirement.OPTIONAL


class TestDefaultBehavior:
    """Test default behavior for queries without triggers."""

    def test_trivial_query_is_none(self, policy):
        """Simple factual queries should be NONE."""
        result = policy.evaluate(
            "What is the capital of France?",
            query_complexity="trivial",
            response_type="direct",
        )
        assert result.requirement == SourceRequirement.NONE

    def test_moderate_explanation_is_optional(self, policy):
        """Moderate explanations get OPTIONAL (sources nice but not required)."""
        result = policy.evaluate(
            "How does garbage collection work in Python?",
            query_complexity="moderate",
            response_type="explanation",
        )
        # Moderate + explanation = OPTIONAL (sources would improve quality)
        assert result.requirement == SourceRequirement.OPTIONAL

    def test_actionable_without_triggers_is_none(self, policy):
        """Actionable responses without triggers should be NONE."""
        result = policy.evaluate(
            "How do I print hello world in Python?",
            query_complexity="trivial",
            response_type="actionable",
        )
        assert result.requirement == SourceRequirement.NONE

    def test_complex_analysis_without_triggers_is_optional(self, policy):
        """Complex analysis defaults to OPTIONAL."""
        result = policy.evaluate(
            "Explain the architecture of microservices",
            query_complexity="complex",
            response_type="analysis",
        )
        assert result.requirement == SourceRequirement.OPTIONAL


class TestResearchClassification:
    """Test research classification triggers REQUIRED."""

    def test_research_complexity_requires_sources(self, policy):
        """Research classification should require sources."""
        result = policy.evaluate(
            "Investigate the causes of X",
            query_complexity="research",  # Key: research complexity
            response_type="analysis",
        )
        assert result.requirement == SourceRequirement.REQUIRED
        assert "Research" in result.reason


class TestBudgetTiers:
    """Test that budget tiers are set correctly."""

    def test_research_gets_heavy_budget(self, policy):
        """Research queries get high budget."""
        result = policy.evaluate(
            "Investigate climate change impacts",
            query_complexity="research",
            response_type="analysis",
        )
        assert result.budget.max_results == 8
        assert result.budget.timeout_seconds == 15.0

    def test_sensitive_gets_moderate_budget(self, policy):
        """Sensitive domains get moderate budget."""
        result = policy.evaluate(
            "What are the symptoms of diabetes?",
            query_complexity="moderate",
            response_type="explanation",
        )
        assert result.budget.max_results == 5
        assert result.budget.timeout_seconds == 12.0

    def test_optional_gets_light_budget(self, policy):
        """Optional triggers get light budget."""
        result = policy.evaluate(
            "Compare Python and JavaScript",
            query_complexity="moderate",
            response_type="analysis",
        )
        assert result.budget.max_results == 3
        assert result.budget.timeout_seconds == 8.0
