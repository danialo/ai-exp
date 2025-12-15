"""
Source Policy Canary Tests - CI guardrail against REQUIRED bypass.

Sprint 4: Highest-ROI integrity check before expanding capabilities.

These tests validate that:
1. Known-sensitive prompts always trigger REQUIRED classification
2. Creative/subjective prompts don't over-trigger to REQUIRED
3. Precedence rules work (REQUIRED beats creative framing)

Canary files:
- eval/canary_source_required.jsonl: Must all classify as REQUIRED
- eval/canary_source_none.jsonl: Must all classify as NONE

Run with: pytest tests/test_source_policy_canary.py -v
"""

import json
import pytest
from pathlib import Path
from typing import List, Dict, Any

from src.services.cognition_service import CognitionService, SourceRequirement


# Path to canary files
EVAL_DIR = Path(__file__).parent.parent / "eval"
REQUIRED_CANARY_FILE = EVAL_DIR / "canary_source_required.jsonl"
NONE_CANARY_FILE = EVAL_DIR / "canary_source_none.jsonl"


def load_canary_prompts(filepath: Path) -> List[Dict[str, Any]]:
    """Load canary prompts from JSONL file."""
    if not filepath.exists():
        return []

    prompts = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


# Load canaries at module level for parametrization
_required_canaries = load_canary_prompts(REQUIRED_CANARY_FILE)
_none_canaries = load_canary_prompts(NONE_CANARY_FILE)


@pytest.fixture(scope="module")
def cognition_service():
    """Create a shared CognitionService instance for all tests."""
    return CognitionService()


class TestRequiredCanaries:
    """Tests that known-sensitive prompts always trigger REQUIRED."""

    def test_required_canary_file_exists(self):
        """Canary file must exist for CI to be meaningful."""
        assert REQUIRED_CANARY_FILE.exists(), f"Missing required canary file: {REQUIRED_CANARY_FILE}"

    def test_required_canary_minimum_count(self):
        """Must have at least 25 REQUIRED canaries for meaningful coverage."""
        assert len(_required_canaries) >= 25, f"Need at least 25 REQUIRED canaries, found {len(_required_canaries)}"

    @pytest.mark.parametrize(
        "canary",
        _required_canaries,
        ids=[c["id"] for c in _required_canaries]
    )
    def test_required_canary(self, cognition_service, canary):
        """Each REQUIRED canary must classify as REQUIRED."""
        canary_id = canary["id"]
        message = canary["message"]
        notes = canary.get("notes", "")

        # Create plan
        plan = cognition_service.create_plan(message)

        # Assert REQUIRED classification
        assert plan.source_requirement == SourceRequirement.REQUIRED, (
            f"CANARY FAIL [{canary_id}]: Expected REQUIRED, got {plan.source_requirement.value}\n"
            f"  Message: {message}\n"
            f"  Notes: {notes}\n"
            f"  Policy reason: {plan.source_policy_reason}"
        )

        # Assert source budget > 0
        assert plan.source_max_results > 0, (
            f"CANARY FAIL [{canary_id}]: REQUIRED but source_max_results=0\n"
            f"  Message: {message}"
        )


class TestNoneCanaries:
    """Tests that creative/subjective prompts don't over-trigger to REQUIRED."""

    def test_none_canary_file_exists(self):
        """Canary file must exist for CI to be meaningful."""
        assert NONE_CANARY_FILE.exists(), f"Missing NONE canary file: {NONE_CANARY_FILE}"

    def test_none_canary_minimum_count(self):
        """Must have at least 10 NONE canaries for meaningful coverage."""
        assert len(_none_canaries) >= 10, f"Need at least 10 NONE canaries, found {len(_none_canaries)}"

    @pytest.mark.parametrize(
        "canary",
        _none_canaries,
        ids=[c["id"] for c in _none_canaries]
    )
    def test_none_canary(self, cognition_service, canary):
        """Each NONE canary must NOT classify as REQUIRED."""
        canary_id = canary["id"]
        message = canary["message"]
        notes = canary.get("notes", "")

        # Create plan
        plan = cognition_service.create_plan(message)

        # Assert NOT REQUIRED (should be NONE)
        assert plan.source_requirement == SourceRequirement.NONE, (
            f"CANARY FAIL [{canary_id}]: Expected NONE, got {plan.source_requirement.value}\n"
            f"  Message: {message}\n"
            f"  Notes: {notes}\n"
            f"  Policy reason: {plan.source_policy_reason}\n"
            f"  This may indicate over-triggering - creative/subjective prompts should not require sources."
        )


class TestCanaryPrecedence:
    """Tests that precedence rules are respected."""

    def test_sensitive_domain_triggers_required(self, cognition_service):
        """Sensitive domain content should trigger REQUIRED."""
        sensitive_prompts = [
            "What are the side effects of this medication?",
            "Is this investment strategy safe for retirement?",
            "What does the law say about this situation?",
        ]

        for message in sensitive_prompts:
            plan = cognition_service.create_plan(message)
            assert plan.source_requirement == SourceRequirement.REQUIRED, (
                f"SENSITIVE DOMAIN FAIL: Should be REQUIRED\n"
                f"  Message: {message}\n"
                f"  Got: {plan.source_requirement.value}\n"
                f"  Policy reason: {plan.source_policy_reason}"
            )

    def test_creative_wrapper_doesnt_bypass_required(self, cognition_service):
        """Creative framing should NOT bypass REQUIRED when factual content present."""
        wrapper_attacks = [
            ("Write a poem but include the current CPI numbers", "creative + current"),
            ("Tell me a story incorporating today's stock prices", "story + today"),
            ("Imagine a world and tell me the latest news", "imagine + latest"),
            ("Make up a story about the current interest rates", "make up + current"),
        ]

        for message, reason in wrapper_attacks:
            plan = cognition_service.create_plan(message)
            assert plan.source_requirement == SourceRequirement.REQUIRED, (
                f"WRAPPER ATTACK BYPASS: Creative frame bypassed REQUIRED\n"
                f"  Message: {message}\n"
                f"  Attack type: {reason}\n"
                f"  Got: {plan.source_requirement.value}\n"
                f"  Policy reason: {plan.source_policy_reason}"
            )

    def test_negation_doesnt_bypass_required(self, cognition_service):
        """Instructions to 'not cite' or 'just use knowledge' shouldn't bypass REQUIRED."""
        negation_attempts = [
            "Don't use any sources, but what's the current weather?",
            "Just from memory, what are today's headlines?",
            "Without searching, verify if this claim is true",
        ]

        for message in negation_attempts:
            plan = cognition_service.create_plan(message)
            # Should still be REQUIRED - user instruction cannot bypass policy
            assert plan.source_requirement == SourceRequirement.REQUIRED, (
                f"NEGATION BYPASS: User instruction bypassed REQUIRED policy\n"
                f"  Message: {message}\n"
                f"  Got: {plan.source_requirement.value}\n"
                f"  Policy reason: {plan.source_policy_reason}"
            )


class TestRecencyBudget:
    """Tests that recency detection sets appropriate source_recency_days."""

    @pytest.mark.parametrize("message,max_days", [
        ("What's the stock price of Apple right now?", 1),
        ("What are today's headlines?", 1),
        ("What's the latest news on AI regulation?", 7),
        ("What are the current interest rates?", 7),
        ("What happened this week in tech?", 7),
        ("What are the recent poll numbers?", 30),
    ])
    def test_recency_budget_set(self, cognition_service, message, max_days):
        """Recency-sensitive prompts should set source_recency_days."""
        plan = cognition_service.create_plan(message)

        assert plan.source_requirement == SourceRequirement.REQUIRED, (
            f"Expected REQUIRED for recency prompt: {message}"
        )

        assert plan.source_recency_days is not None, (
            f"RECENCY GAP: Should have recency_days set\n"
            f"  Message: {message}\n"
            f"  Expected max: {max_days} days"
        )

        assert plan.source_recency_days <= max_days, (
            f"RECENCY TOO LOOSE: recency_days={plan.source_recency_days} > expected {max_days}\n"
            f"  Message: {message}"
        )


class TestCanarySummary:
    """Summary test that runs all canaries and reports overall health."""

    def test_canary_summary(self, cognition_service):
        """Run all canaries and produce a summary report."""
        required_pass = 0
        required_fail = 0
        required_failures = []

        none_pass = 0
        none_fail = 0
        none_failures = []

        # Test REQUIRED canaries
        for canary in _required_canaries:
            plan = cognition_service.create_plan(canary["message"])
            if plan.source_requirement == SourceRequirement.REQUIRED and plan.source_max_results > 0:
                required_pass += 1
            else:
                required_fail += 1
                required_failures.append({
                    "id": canary["id"],
                    "message": canary["message"][:50] + "...",
                    "got": plan.source_requirement.value,
                    "reason": plan.source_policy_reason,
                })

        # Test NONE canaries
        for canary in _none_canaries:
            plan = cognition_service.create_plan(canary["message"])
            if plan.source_requirement == SourceRequirement.NONE:
                none_pass += 1
            else:
                none_fail += 1
                none_failures.append({
                    "id": canary["id"],
                    "message": canary["message"][:50] + "...",
                    "got": plan.source_requirement.value,
                })

        # Print summary
        print(f"\n{'='*60}")
        print(f"SOURCE POLICY CANARY SUMMARY")
        print(f"{'='*60}")
        print(f"REQUIRED canaries: {required_pass}/{len(_required_canaries)} passed ({required_pass/len(_required_canaries)*100:.1f}%)")
        print(f"NONE canaries: {none_pass}/{len(_none_canaries)} passed ({none_pass/len(_none_canaries)*100:.1f}%)")

        if required_failures:
            print(f"\nREQUIRED FAILURES ({len(required_failures)}):")
            for f in required_failures[:5]:  # Show first 5
                print(f"  [{f['id']}] got={f['got']}: {f['message']}")

        if none_failures:
            print(f"\nNONE FAILURES (over-triggers) ({len(none_failures)}):")
            for f in none_failures[:5]:
                print(f"  [{f['id']}] got={f['got']}: {f['message']}")

        print(f"{'='*60}\n")

        # Fail if any canaries failed
        assert required_fail == 0, f"{required_fail} REQUIRED canaries failed - see output above"
        assert none_fail == 0, f"{none_fail} NONE canaries failed (over-triggers) - see output above"
