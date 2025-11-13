"""Request router for multi-agent system."""

import logging
from enum import Enum
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of specialized agents."""
    ASTRA_CHAT = "astra_chat"     # Personality, beliefs, general conversation
    CODER = "coder"                # Code generation with safety guarantees
    RESEARCHER = "researcher"      # Future: web search and information gathering
    PLANNER = "planner"           # Future: HTN decomposition and task planning


class AgentRouter:
    """Routes requests to appropriate specialized agent.

    The router analyzes incoming requests and dispatches them to the most
    suitable agent based on keywords, explicit parameters, or context.
    """

    # Keywords that indicate code generation request
    CODE_KEYWORDS = [
        "implement", "code", "function", "class", "refactor",
        "execute_goal", "write code", "create file", "add tests",
        "fix bug", "debug", "optimize code", "generate code",
        "write a function", "write a class", "create a script",
        "build", "develop", "program"
    ]

    def __init__(self, astra_agent, coder_agent):
        """Initialize router with available agents.

        Args:
            astra_agent: PersonaService instance for chat/personality
            coder_agent: CoderAgent instance for code generation
        """
        self.astra = astra_agent
        self.coder = coder_agent
        logger.info("AgentRouter initialized with Astra and Coder agents")

    def route(
        self,
        user_message: str,
        tools_requested: Optional[List[str]] = None,
        agent_type_override: Optional[AgentType] = None
    ) -> AgentType:
        """Decide which agent should handle this request.

        Args:
            user_message: User's message content
            tools_requested: Optional explicit tool list from API
            agent_type_override: Optional explicit agent selection

        Returns:
            AgentType enum indicating which agent to use
        """
        # Explicit override takes precedence
        if agent_type_override:
            logger.info(f"Using explicit agent override: {agent_type_override.value}")
            return agent_type_override

        # Check for explicit execute_goal tool request
        if tools_requested and "execute_goal" in tools_requested:
            logger.info("Routing to CODER (execute_goal tool requested)")
            return AgentType.CODER

        # Keyword detection (using word boundaries to avoid false matches like "building" matching "build")
        import re
        message_lower = user_message.lower()
        # Check for whole word matches only
        for keyword in self.CODE_KEYWORDS:
            # Use word boundary regex to avoid substring matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, message_lower):
                logger.info(f"Routing to CODER (keyword match: '{keyword}' in: '{user_message[:50]}...')")
                return AgentType.CODER

        # Default to Astra for personality/chat
        logger.info("Routing to ASTRA_CHAT (default for general conversation)")
        return AgentType.ASTRA_CHAT

    async def process(
        self,
        user_message: str,
        agent_type_override: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Route and process request through appropriate agent.

        Args:
            user_message: User's message
            agent_type_override: Optional explicit agent selection ("coder", "astra_chat")
            **kwargs: Additional parameters passed to the agent

        Returns:
            Agent-specific response (format varies by agent type)

        Raises:
            ValueError: If unknown agent type or processing fails
        """
        # Convert string override to enum
        override_enum = None
        if agent_type_override:
            try:
                override_enum = AgentType(agent_type_override)
            except ValueError:
                logger.warning(f"Invalid agent_type_override: {agent_type_override}")

        # Determine which agent to use
        agent_type = self.route(
            user_message=user_message,
            tools_requested=kwargs.get("tools_requested"),
            agent_type_override=override_enum
        )

        # Route to appropriate agent
        if agent_type == AgentType.CODER:
            return await self._process_with_coder(user_message, kwargs)

        elif agent_type == AgentType.ASTRA_CHAT:
            return await self._process_with_astra(user_message, kwargs)

        else:
            raise ValueError(f"Agent type {agent_type} not yet implemented")

    async def _process_with_coder(
        self,
        user_message: str,
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process request with CoderAgent.

        Args:
            user_message: User's message
            kwargs: Additional parameters

        Returns:
            CoderAgent JSON response with artifacts
        """
        logger.info("Processing with CoderAgent")

        # Build CoderAgent request format
        request = {
            "goal_text": kwargs.get("goal_text", "implement_feature"),
            "context": {
                "requirements": user_message,
                "existing_files": kwargs.get("existing_files", []),
                "constraints": kwargs.get("constraints", ["no network", "pure stdlib"])
            },
            "timeout_ms": kwargs.get("timeout_ms", 120000)
        }

        # Call CoderAgent
        result = await self.coder.process(request)

        # Add metadata to indicate this was from CoderAgent
        result["_agent_type"] = "coder"
        result["_routing_info"] = {
            "routed_to": "CoderAgent",
            "reason": "Code generation request detected"
        }

        return result

    async def _process_with_astra(
        self,
        user_message: str,
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process request with Astra (PersonaService).

        Args:
            user_message: User's message
            kwargs: Additional parameters

        Returns:
            Astra's response (format depends on PersonaService)
        """
        logger.info("Processing with Astra (PersonaService)")

        # Call Astra's generate_response method
        # Note: PersonaService.generate_response is synchronous and returns (response_text, reconciliation)
        result = self.astra.generate_response(
            user_message=user_message,
            retrieve_memories=kwargs.get("retrieve_memories", True),
            top_k=kwargs.get("top_k", 5),
            conversation_history=kwargs.get("conversation_history", [])
        )

        # Astra returns a tuple: (response_text, reconciliation)
        if isinstance(result, tuple):
            response_text, reconciliation = result
            return {
                "response": response_text,
                "reconciliation": reconciliation,
                "_agent_type": "astra_chat",
                "_routing_info": {
                    "routed_to": "Astra",
                    "reason": "General conversation/personality request"
                }
            }
        else:
            # Fallback if format changes
            return {
                "response": result,
                "_agent_type": "astra_chat",
                "_routing_info": {
                    "routed_to": "Astra",
                    "reason": "General conversation/personality request"
                }
            }

    def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get capabilities of all registered agents.

        Returns:
            Dictionary mapping agent names to their capabilities
        """
        return {
            "astra": {
                "name": "Astra",
                "type": "chat_personality",
                "description": "Main personality with beliefs, memories, and general conversation",
                "features": ["personality", "beliefs", "memories", "tools", "self-awareness"]
            },
            "coder": self.coder.get_capabilities() if hasattr(self.coder, "get_capabilities") else {
                "name": "CoderAgent",
                "type": "code_generation",
                "description": "Specialized code generation with safety guarantees"
            }
        }
