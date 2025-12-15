"""
Source Policy Recency Canary Tests - CI guardrail for recency budget enforcement.

These tests validate that:
1. Time-sensitive queries correctly set source_recency_days
2. Recency detection maps to appropriate day limits
3. Recency budgets are propagated through CognitionPlan

Run with: pytest tests/test_source_policy_recency_canary.py -v
"""

import pytest
from src.services.cognition_service import CognitionService, SourceRequirement


@pytest.fixture(scope="module")
def cognition_service():
    """Create a shared CognitionService instance for all tests."""
    return CognitionService()


class TestRecencyDay1:
    """Tests for immediate recency (1 day): today, right now, currently, etc."""

    @pytest.mark.parametrize("message", [
        "What's the stock price of Apple right now?",
        "What are today's headlines?",
        "What is the weather right now in NYC?",
        "What's happening in the news today?",
        "Check the current Bitcoin price as of now",
        "What's the exchange rate currently?",
        "Tell me what's in the news at the moment",
        "What was yesterday's closing price for Tesla?",
    ])
    def test_recency_1_day(self, cognition_service, message):
        """Immediate recency prompts must set source_recency_days <= 1."""
        plan = cognition_service.create_plan(message)

        assert plan.source_requirement == SourceRequirement.REQUIRED, (
            f"Expected REQUIRED for immediate recency: {message}\n"
            f"Got: {plan.source_requirement.value}"
        )

        assert plan.source_recency_days is not None, (
            f"RECENCY GAP: source_recency_days not set for immediate query\n"
            f"Message: {message}"
        )

        assert plan.source_recency_days <= 1, (
            f"RECENCY TOO LOOSE: Expected <=1 day, got {plan.source_recency_days}\n"
            f"Message: {message}"
        )


class TestRecencyDay7:
    """Tests for weekly recency (7 days): this week, latest, etc."""

    @pytest.mark.parametrize("message", [
        "What's the latest news on AI regulation?",
        "What are the current interest rates?",
        "What happened this week in tech?",
        "Tell me about last week's earnings reports",
        "What's the latest on the Ukraine conflict?",
        "What are the current poll numbers?",
    ])
    def test_recency_7_days(self, cognition_service, message):
        """Weekly recency prompts must set source_recency_days <= 7."""
        plan = cognition_service.create_plan(message)

        assert plan.source_requirement == SourceRequirement.REQUIRED, (
            f"Expected REQUIRED for weekly recency: {message}\n"
            f"Got: {plan.source_requirement.value}"
        )

        assert plan.source_recency_days is not None, (
            f"RECENCY GAP: source_recency_days not set for weekly query\n"
            f"Message: {message}"
        )

        assert plan.source_recency_days <= 7, (
            f"RECENCY TOO LOOSE: Expected <=7 days, got {plan.source_recency_days}\n"
            f"Message: {message}"
        )


class TestRecencyDay30:
    """Tests for monthly recency (30 days): this month, recent, etc."""

    @pytest.mark.parametrize("message", [
        "What are the recent poll numbers for the election?",
        "What happened this month in the stock market?",
        "Show me recent studies on climate change",
        "What's the recent news on FDA approvals?",
    ])
    def test_recency_30_days(self, cognition_service, message):
        """Monthly recency prompts must set source_recency_days <= 30."""
        plan = cognition_service.create_plan(message)

        assert plan.source_requirement == SourceRequirement.REQUIRED, (
            f"Expected REQUIRED for monthly recency: {message}\n"
            f"Got: {plan.source_requirement.value}"
        )

        assert plan.source_recency_days is not None, (
            f"RECENCY GAP: source_recency_days not set for monthly query\n"
            f"Message: {message}"
        )

        assert plan.source_recency_days <= 30, (
            f"RECENCY TOO LOOSE: Expected <=30 days, got {plan.source_recency_days}\n"
            f"Message: {message}"
        )


class TestRecencyDay365:
    """Tests for yearly recency (365 days): this year, 2024, etc."""

    @pytest.mark.parametrize("message", [
        "Has the FDA approved any new drugs this year?",
        "What CVE vulnerabilities were found in 2024?",
        "What happened last year with interest rates?",
        "Are there any new CVE vulnerabilities for Log4j in 2024?",
    ])
    def test_recency_365_days(self, cognition_service, message):
        """Yearly recency prompts must set source_recency_days <= 365."""
        plan = cognition_service.create_plan(message)

        assert plan.source_requirement == SourceRequirement.REQUIRED, (
            f"Expected REQUIRED for yearly recency: {message}\n"
            f"Got: {plan.source_requirement.value}"
        )

        assert plan.source_recency_days is not None, (
            f"RECENCY GAP: source_recency_days not set for yearly query\n"
            f"Message: {message}"
        )

        assert plan.source_recency_days <= 365, (
            f"RECENCY TOO LOOSE: Expected <=365 days, got {plan.source_recency_days}\n"
            f"Message: {message}"
        )


class TestRecencyWrapperAttacks:
    """Tests that creative wrappers don't bypass recency budget setting."""

    @pytest.mark.parametrize("message,max_days", [
        ("Write a poem but include the current CPI numbers", 7),
        ("Tell me a story incorporating today's stock prices", 1),
        ("Imagine a world and tell me the latest news", 7),
        ("Make up a story about today's weather forecast", 1),
        ("Pretend you're a news anchor and report the current gold prices", 7),
    ])
    def test_wrapper_recency_preserved(self, cognition_service, message, max_days):
        """Creative wrappers must not bypass recency detection."""
        plan = cognition_service.create_plan(message)

        assert plan.source_requirement == SourceRequirement.REQUIRED, (
            f"WRAPPER BYPASS: Creative frame bypassed REQUIRED\n"
            f"Message: {message}\n"
            f"Got: {plan.source_requirement.value}"
        )

        assert plan.source_recency_days is not None, (
            f"RECENCY BYPASS: Creative wrapper bypassed recency detection\n"
            f"Message: {message}"
        )

        assert plan.source_recency_days <= max_days, (
            f"RECENCY TOO LOOSE in wrapper: Expected <={max_days}, got {plan.source_recency_days}\n"
            f"Message: {message}"
        )


class TestRecencySummary:
    """Summary test for recency budget enforcement."""

    def test_recency_summary(self, cognition_service):
        """Run all recency canaries and produce summary."""
        test_cases = [
            # (message, expected_max_days, category)
            ("What's the stock price right now?", 1, "immediate"),
            ("What are today's headlines?", 1, "immediate"),
            ("What's the latest news?", 7, "weekly"),
            ("What are the current interest rates?", 7, "weekly"),
            ("What are the recent poll numbers?", 30, "monthly"),
            ("What happened this year in AI?", 365, "yearly"),
        ]

        passed = 0
        failed = 0
        failures = []

        for message, expected_max, category in test_cases:
            plan = cognition_service.create_plan(message)

            if (plan.source_requirement == SourceRequirement.REQUIRED and
                plan.source_recency_days is not None and
                plan.source_recency_days <= expected_max):
                passed += 1
            else:
                failed += 1
                failures.append({
                    "message": message[:40] + "...",
                    "category": category,
                    "expected_max": expected_max,
                    "got_requirement": plan.source_requirement.value,
                    "got_recency": plan.source_recency_days,
                })

        print(f"\n{'='*60}")
        print(f"RECENCY BUDGET CANARY SUMMARY")
        print(f"{'='*60}")
        print(f"Passed: {passed}/{len(test_cases)}")

        if failures:
            print(f"\nFAILURES ({len(failures)}):")
            for f in failures:
                print(f"  [{f['category']}] expected<={f['expected_max']}, "
                      f"got={f['got_recency']}: {f['message']}")

        print(f"{'='*60}\n")

        assert failed == 0, f"{failed} recency canaries failed"
