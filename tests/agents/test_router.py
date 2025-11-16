"""Unit tests for AgentRouter."""

import pytest
from unittest.mock import AsyncMock, Mock

from src.agents.router import AgentRouter, AgentType


@pytest.fixture
def mock_astra_agent():
    """Create mock Astra agent."""
    astra = Mock()
    astra.generate_response = AsyncMock(return_value=("Hello!", None))
    return astra


@pytest.fixture
def mock_coder_agent():
    """Create mock CoderAgent."""
    coder = Mock()
    coder.process = AsyncMock(return_value={
        "plan": ["step 1"],
        "artifacts": [],
        "checks": {},
        "assumptions": []
    })
    coder.get_capabilities = Mock(return_value={
        "name": "CoderAgent",
        "description": "Code generation"
    })
    return coder


@pytest.fixture
def router(mock_astra_agent, mock_coder_agent):
    """Create AgentRouter instance."""
    return AgentRouter(
        astra_agent=mock_astra_agent,
        coder_agent=mock_coder_agent
    )


class TestRouting:
    """Test routing logic."""

    def test_route_with_explicit_override(self, router):
        """Test explicit agent override takes precedence."""
        result = router.route(
            user_message="just chatting",
            agent_type_override=AgentType.CODER
        )
        assert result == AgentType.CODER

    def test_route_with_execute_goal_tool(self, router):
        """Test routing when execute_goal tool requested."""
        result = router.route(
            user_message="hello",
            tools_requested=["execute_goal"]
        )
        assert result == AgentType.CODER

    def test_route_with_implement_keyword(self, router):
        """Test routing with 'implement' keyword."""
        result = router.route(
            user_message="implement a new feature for parsing JSON"
        )
        assert result == AgentType.CODER

    def test_route_with_code_keyword(self, router):
        """Test routing with 'code' keyword."""
        result = router.route(
            user_message="write code to handle user input"
        )
        assert result == AgentType.CODER

    def test_route_with_function_keyword(self, router):
        """Test routing with 'function' keyword."""
        result = router.route(
            user_message="create a function that calculates fibonacci"
        )
        assert result == AgentType.CODER

    def test_route_with_refactor_keyword(self, router):
        """Test routing with 'refactor' keyword."""
        result = router.route(
            user_message="refactor the parser module"
        )
        assert result == AgentType.CODER

    def test_route_with_debug_keyword(self, router):
        """Test routing with 'debug' keyword."""
        result = router.route(
            user_message="debug this issue in the code"
        )
        assert result == AgentType.CODER

    def test_route_general_chat_to_astra(self, router):
        """Test routing general conversation to Astra."""
        result = router.route(
            user_message="How are you feeling today?"
        )
        assert result == AgentType.ASTRA_CHAT

    def test_route_question_to_astra(self, router):
        """Test routing questions to Astra."""
        result = router.route(
            user_message="What do you think about artificial intelligence?"
        )
        assert result == AgentType.ASTRA_CHAT

    def test_route_belief_query_to_astra(self, router):
        """Test routing belief queries to Astra."""
        result = router.route(
            user_message="What are your beliefs about consciousness?"
        )
        assert result == AgentType.ASTRA_CHAT


class TestProcessMethod:
    """Test the main process() method."""

    @pytest.mark.asyncio
    async def test_process_with_coder(self, router, mock_coder_agent):
        """Test processing code request."""
        result = await router.process(
            user_message="implement a calculator"
        )

        # Verify CoderAgent was called
        assert mock_coder_agent.process.called

        # Verify metadata added
        assert result["_agent_type"] == "coder"
        assert result["_routing_info"]["routed_to"] == "CoderAgent"

    @pytest.mark.asyncio
    async def test_process_with_astra(self, router, mock_astra_agent):
        """Test processing chat request."""
        result = await router.process(
            user_message="How are you?"
        )

        # Verify Astra was called
        assert mock_astra_agent.generate_response.called

        # Verify response format
        assert result["response"] == "Hello!"
        assert result["_agent_type"] == "astra_chat"
        assert result["_routing_info"]["routed_to"] == "Astra"

    @pytest.mark.asyncio
    async def test_process_with_explicit_agent_override(self, router, mock_coder_agent):
        """Test explicit agent selection via string override."""
        result = await router.process(
            user_message="just chatting",
            agent_type_override="coder"
        )

        # Should route to coder despite chat message
        assert mock_coder_agent.process.called
        assert result["_agent_type"] == "coder"

    @pytest.mark.asyncio
    async def test_process_with_invalid_agent_override(self, router, mock_astra_agent):
        """Test handling of invalid agent override."""
        # Should fallback to normal routing
        result = await router.process(
            user_message="hello",
            agent_type_override="invalid_agent"
        )

        # Should default to Astra
        assert mock_astra_agent.generate_response.called

    @pytest.mark.asyncio
    async def test_process_passes_kwargs_to_coder(self, router, mock_coder_agent):
        """Test that kwargs are passed to CoderAgent."""
        await router.process(
            user_message="implement feature",
            goal_text="fix_bug",
            existing_files=["src/test.py"],
            constraints=["no network"]
        )

        # Verify request was built correctly
        call_args = mock_coder_agent.process.call_args
        request = call_args[0][0]

        assert request["goal_text"] == "fix_bug"
        assert "src/test.py" in request["context"]["existing_files"]
        assert "no network" in request["context"]["constraints"]

    @pytest.mark.asyncio
    async def test_process_passes_kwargs_to_astra(self, router, mock_astra_agent):
        """Test that kwargs are passed to Astra."""
        await router.process(
            user_message="hello",
            retrieve_memories=False,
            top_k=10,
            conversation_history=[{"role": "user", "content": "hi"}]
        )

        # Verify kwargs were passed
        call_kwargs = mock_astra_agent.generate_response.call_args[1]

        assert call_kwargs["retrieve_memories"] is False
        assert call_kwargs["top_k"] == 10
        assert len(call_kwargs["conversation_history"]) == 1


class TestCoderRequestBuilding:
    """Test building CoderAgent requests."""

    @pytest.mark.asyncio
    async def test_coder_request_default_values(self, router, mock_coder_agent):
        """Test CoderAgent request with default values."""
        await router.process(user_message="implement feature")

        request = mock_coder_agent.process.call_args[0][0]

        assert request["goal_text"] == "implement_feature"
        assert request["context"]["requirements"] == "implement feature"
        assert "no network" in request["context"]["constraints"]
        assert "pure stdlib" in request["context"]["constraints"]
        assert request["timeout_ms"] == 120000

    @pytest.mark.asyncio
    async def test_coder_request_custom_values(self, router, mock_coder_agent):
        """Test CoderAgent request with custom values."""
        await router.process(
            user_message="build calculator",
            goal_text="implement_feature",
            existing_files=["src/calc.py"],
            constraints=["max 300 lines"],
            timeout_ms=60000
        )

        request = mock_coder_agent.process.call_args[0][0]

        assert request["goal_text"] == "implement_feature"
        assert request["context"]["requirements"] == "build calculator"
        assert "src/calc.py" in request["context"]["existing_files"]
        assert "max 300 lines" in request["context"]["constraints"]
        assert request["timeout_ms"] == 60000


class TestCapabilities:
    """Test agent capabilities reporting."""

    def test_get_agent_capabilities(self, router):
        """Test getting capabilities of all agents."""
        caps = router.get_agent_capabilities()

        assert "astra" in caps
        assert "coder" in caps

        assert caps["astra"]["name"] == "Astra"
        assert caps["astra"]["type"] == "chat_personality"

        assert caps["coder"]["name"] == "CoderAgent"
        assert caps["coder"]["description"] == "Code generation"


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_process_unknown_agent_type(self, router):
        """Test handling of unknown agent type."""
        # This shouldn't happen in practice, but test defense
        with pytest.raises(ValueError, match="not yet implemented"):
            await router.process(
                user_message="test",
                agent_type_override="planner"  # Not implemented yet
            )

    def test_route_with_mixed_keywords(self, router):
        """Test routing with both code and chat keywords."""
        # Code keywords should take precedence
        result = router.route(
            user_message="I'd like to implement a feature. What do you think?"
        )
        assert result == AgentType.CODER

    def test_route_case_insensitive(self, router):
        """Test that routing is case-insensitive."""
        result1 = router.route(user_message="IMPLEMENT A FEATURE")
        result2 = router.route(user_message="implement a feature")
        result3 = router.route(user_message="ImPlEmEnT a feature")

        assert result1 == result2 == result3 == AgentType.CODER

    @pytest.mark.asyncio
    async def test_astra_returns_dict_not_tuple(self, router, mock_astra_agent):
        """Test handling when Astra returns dict instead of tuple."""
        # Change mock to return dict
        mock_astra_agent.generate_response.return_value = {"some": "result"}

        result = await router.process(user_message="hello")

        assert result["response"] == {"some": "result"}
        assert result["_agent_type"] == "astra_chat"
