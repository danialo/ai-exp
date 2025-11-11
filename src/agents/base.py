"""Base agent interface for multi-agent system."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAgent(ABC):
    """Base interface for all specialized agents.

    Each agent is a specialist focused on a specific domain:
    - CoderAgent: Code generation with safety guarantees
    - ResearchAgent: Web search and information gathering
    - PlannerAgent: HTN decomposition and task planning
    - AstraAgent: Personality, beliefs, general conversation
    """

    @abstractmethod
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process request and return structured output.

        Args:
            request: Agent-specific request format

        Returns:
            Agent-specific response format

        Raises:
            ValueError: Invalid request format or processing error
        """
        pass

    @abstractmethod
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate output matches expected schema.

        Args:
            output: Response to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities and constraints.

        Returns:
            Dict with keys:
            - name: Agent name
            - description: What this agent does
            - max_input_tokens: Approximate input token limit
            - max_output_tokens: Approximate output token limit
            - cost_per_request: Estimated cost in USD
            - avg_latency_ms: Average response time
        """
        pass
