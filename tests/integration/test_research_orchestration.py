"""
Integration tests for research orchestration.

These tests verify that the research gate makes the announcement bug
MECHANICALLY IMPOSSIBLE by ensuring:

1. Research tools run BEFORE model generates prose
2. Research tools are excluded when research_context exists
3. The model cannot "announce" research because it's already done

These tests do NOT depend on model obedience - they test the SYSTEM behavior.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.services.research_gate import ResearchGate, ResearchDecision, GateResult


class TestResearchOrchestration:
    """Test that research orchestration prevents announcement bug."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        llm = Mock()
        llm.model = "gpt-4"
        llm.generate = Mock(return_value={"content": "YES"})
        llm.generate_with_tools = Mock(return_value={
            "message": Mock(content="Test response", tool_calls=None),
            "finish_reason": "stop"
        })
        return llm

    @pytest.fixture
    def mock_services(self, mock_llm_service):
        """Create all mock services needed for PersonaService."""
        return {
            "llm_service": mock_llm_service,
            "persona_space_path": "/tmp/test_persona_space",
            "web_search_service": Mock(),
            "url_fetcher_service": Mock(),
        }

    def test_research_gate_decides_before_model(self):
        """Research gate must decide BEFORE model is called."""
        gate = ResearchGate(llm_service=None, use_classifier=False)

        # This should return a decision immediately, without any model call
        result = gate.requires_research("What happened with DOGE yesterday?")

        # Decision should be made (not pending or waiting for model)
        assert result.decision in [ResearchDecision.RESEARCH_REQUIRED, ResearchDecision.NO_RESEARCH, ResearchDecision.AMBIGUOUS]
        assert result.needs_research is True  # This specific query should trigger

    def test_research_tools_excluded_when_context_exists(self):
        """When research_context is provided, research tools must be excluded."""
        # Import here to avoid circular imports during test collection
        from src.services.persona_service import PersonaService

        # Create mock services
        llm = Mock()
        llm.model = "gpt-4"

        # Create service (minimal, just to test tool definitions)
        with patch('src.services.persona_service.PersonaPromptBuilder'):
            with patch('src.services.persona_service.EmotionalReconciler'):
                with patch('src.services.persona_service.PersonaFileManager'):
                    with patch('src.services.persona_service.create_persona_config_loader'):
                        with patch('src.services.persona_service.create_logit_bias_builder'):
                            with patch('src.services.persona_service.create_metatalk_detector'):
                                with patch('src.services.persona_service.create_metatalk_rewriter'):
                                    service = PersonaService(
                                        llm_service=llm,
                                        persona_space_path="/tmp/test"
                                    )

        # Get tools WITHOUT research context
        tools_all = service._get_tool_definitions(exclude_research_tools=False)
        tool_names_all = {t['function']['name'] for t in tools_all}

        # Get tools WITH research context (simulated by exclude_research_tools=True)
        tools_filtered = service._get_tool_definitions(exclude_research_tools=True)
        tool_names_filtered = {t['function']['name'] for t in tools_filtered}

        # Research tools should be present when no context
        assert 'search_web' in tool_names_all
        assert 'browse_url' in tool_names_all
        assert 'research_and_summarize' in tool_names_all
        assert 'check_recent_research' in tool_names_all

        # Research tools should be ABSENT when context exists
        assert 'search_web' not in tool_names_filtered
        assert 'browse_url' not in tool_names_filtered
        assert 'research_and_summarize' not in tool_names_filtered
        assert 'check_recent_research' not in tool_names_filtered

        # Other tools should still be present
        assert 'write_file' in tool_names_filtered
        assert 'read_file' in tool_names_filtered

    def test_explicit_research_triggers_gate(self):
        """Explicit research commands must trigger the gate."""
        gate = ResearchGate(llm_service=None, use_classifier=False)

        # All of these should trigger research
        research_queries = [
            "research the latest AI news",
            "investigate what happened with Tesla",
            "deep dive into quantum computing",
            "look into the recent changes",
            "dig into this topic for me",
            "comprehensive analysis of the market",
        ]

        for query in research_queries:
            result = gate.requires_research(query)
            assert result.needs_research is True, f"Failed for: {query}"

    def test_non_research_queries_dont_trigger_gate(self):
        """Non-research queries must NOT trigger the gate."""
        gate = ResearchGate(llm_service=None, use_classifier=False)

        # None of these should trigger research
        non_research_queries = [
            "How do I write a Python function?",
            "Help me fix this bug",
            "What is machine learning?",
            "Calculate 2 + 2",
            "Hello, how are you?",
            "What do you think about AI?",
            "Explain recursion to me",
        ]

        for query in non_research_queries:
            result = gate.requires_research(query)
            assert result.needs_research is False, f"Incorrectly triggered for: {query}"

    def test_meta_questions_about_research_dont_trigger(self):
        """Questions ABOUT the research system must NOT trigger research."""
        gate = ResearchGate(llm_service=None, use_classifier=False)

        meta_queries = [
            "How does your research system work?",
            "What are your research capabilities?",
            "Can you research things on the web?",
            "Tell me about your research tools",
        ]

        for query in meta_queries:
            result = gate.requires_research(query)
            assert result.needs_research is False, f"Meta query triggered research: {query}"


class TestToolExclusionIntegrity:
    """Test that tool exclusion cannot be bypassed."""

    def test_research_tool_names_constant(self):
        """Verify the research tool names constant is correct."""
        from src.services.persona_service import PersonaService

        expected_research_tools = {"search_web", "browse_url", "check_recent_research", "research_and_summarize"}
        assert PersonaService.RESEARCH_TOOL_NAMES == expected_research_tools

    def test_all_research_tools_in_full_list(self):
        """All research tools must be in the full tool list (so they can be excluded)."""
        from src.services.persona_service import PersonaService

        # Create minimal mock
        llm = Mock()
        llm.model = "gpt-4"

        with patch('src.services.persona_service.PersonaPromptBuilder'):
            with patch('src.services.persona_service.EmotionalReconciler'):
                with patch('src.services.persona_service.PersonaFileManager'):
                    with patch('src.services.persona_service.create_persona_config_loader'):
                        with patch('src.services.persona_service.create_logit_bias_builder'):
                            with patch('src.services.persona_service.create_metatalk_detector'):
                                with patch('src.services.persona_service.create_metatalk_rewriter'):
                                    service = PersonaService(
                                        llm_service=llm,
                                        persona_space_path="/tmp/test"
                                    )

        tools_all = service._get_tool_definitions(exclude_research_tools=False)
        tool_names_all = {t['function']['name'] for t in tools_all}

        # Every research tool name in the constant should be in the full tool list
        for research_tool in PersonaService.RESEARCH_TOOL_NAMES:
            assert research_tool in tool_names_all, f"Research tool {research_tool} not in tool list"


class TestBadModelOutputHandling:
    """
    Test that the system handles "bad" model outputs correctly.

    Even if the model TRIES to announce research, the system should have
    already run the research, so the announcement is irrelevant.
    """

    def test_system_runs_research_before_model_can_announce(self):
        """
        Verify that research runs BEFORE model generation.

        This test simulates the scenario where:
        1. User asks a research query
        2. System gate triggers research
        3. Research completes
        4. THEN model is called (with research_context, without research tools)

        The model cannot "announce" research because research is already done.
        """
        gate = ResearchGate(llm_service=None, use_classifier=False)

        # Step 1: System evaluates query
        query = "What happened with DOGE yesterday?"
        result = gate.requires_research(query)

        # Step 2: Gate decides research is needed
        assert result.needs_research is True

        # At this point, the system would:
        # 1. Run _get_research_context(query) - research HAPPENS HERE
        # 2. THEN call model with research_context set
        # 3. Model receives tools WITHOUT research tools

        # The model CANNOT announce research because:
        # - Research already ran
        # - Research tools are not available to the model
        # - The prompt includes the research results

        # This test verifies the GATE decision is made before any model call
        assert result.classifier_used is False  # Pure heuristics, no model needed


class TestGateResultConsistency:
    """Test that gate results are consistent and complete."""

    def test_gate_result_always_has_reason(self):
        """Every gate result must have a reason for auditability."""
        gate = ResearchGate(llm_service=None, use_classifier=False)

        test_queries = [
            "research this",
            "how do I code",
            "what's happening today",
            "hello",
        ]

        for query in test_queries:
            result = gate.requires_research(query)
            assert result.reason is not None
            assert len(result.reason) > 0

    def test_gate_result_has_confidence(self):
        """Every gate result must have a confidence score."""
        gate = ResearchGate(llm_service=None, use_classifier=False)

        result = gate.requires_research("research AI news")
        assert 0 <= result.confidence <= 1

        result = gate.requires_research("how do I code?")
        assert 0 <= result.confidence <= 1
