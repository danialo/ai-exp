"""LLM service for generating responses with context from memories."""

from typing import Protocol, List
from datetime import datetime

from openai import OpenAI


class Memory(Protocol):
    """Protocol for memory objects."""

    prompt_text: str
    response_text: str
    created_at: datetime
    similarity_score: float
    recency_score: float


class LLMService:
    """Service for generating LLM responses with memory context."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500,
        base_url: str | None = None,
        self_aware_prompt_builder=None,
    ):
        """Initialize LLM service.

        Args:
            api_key: API key for the LLM provider
            model: Model name to use
            temperature: Temperature for generation (0-2)
            max_tokens: Maximum tokens in response
            base_url: Optional base URL for API endpoint (e.g., Venice AI)
            self_aware_prompt_builder: Optional builder for self-aware prompts
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.self_aware_prompt_builder = self_aware_prompt_builder

    def generate_response(
        self,
        prompt: str,
        memories: List[Memory] | None = None,
        system_prompt: str | None = None,
        include_self_awareness: bool = True,
    ) -> str:
        """Generate a response using the LLM with optional memory context.

        Args:
            prompt: User's current prompt
            memories: Retrieved relevant memories for context
            system_prompt: Optional system prompt to guide behavior
            include_self_awareness: Whether to include self-concept in system prompt

        Returns:
            Generated response text
        """
        # Build messages
        messages = []

        # Build self-aware system prompt if available
        if system_prompt is None:
            if self.self_aware_prompt_builder and include_self_awareness:
                system_prompt = self.self_aware_prompt_builder.build_self_aware_system_prompt()
            else:
                system_prompt = self._build_default_system_prompt()

        messages.append({"role": "system", "content": system_prompt})

        # Add memory context if available
        if memories:
            context = self._format_memories(memories)
            messages.append({
                "role": "system",
                "content": f"Here is relevant context from past conversations:\n\n{context}"
            })

        # Add user prompt
        messages.append({"role": "user", "content": prompt})

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content

    def _build_default_system_prompt(self) -> str:
        """Build default system prompt.

        Returns:
            Default system prompt
        """
        return (
            "You are a helpful AI assistant with memory of past conversations. "
            "When provided with context from previous interactions, use that information "
            "to answer the user's current question. Pay special attention to references "
            "like 'that', 'it', 'the one', etc. that likely refer to topics from recent "
            "conversations. Use the conversation history to resolve these references and "
            "provide contextually aware responses."
        )

    def _format_memories(self, memories: List[Memory]) -> str:
        """Format memories for injection into prompt.

        Args:
            memories: List of memory objects

        Returns:
            Formatted memory context string
        """
        if not memories:
            return ""

        lines = ["Recent conversation history (most relevant first):"]
        lines.append("")

        for i, mem in enumerate(memories, 1):
            timestamp = mem.created_at.strftime("%Y-%m-%d %H:%M")
            lines.append(f"[{i}] Conversation from {timestamp} (relevance: {mem.similarity_score:.2f}):")
            lines.append(f"    User: {mem.prompt_text}")
            lines.append(f"    Assistant: {mem.response_text}")
            lines.append("")

        return "\n".join(lines)


def create_llm_service(
    api_key: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 500,
    base_url: str | None = None,
    self_aware_prompt_builder=None,
) -> LLMService:
    """Factory function to create an LLM service.

    Args:
        api_key: API key for the LLM provider
        model: Model name to use
        temperature: Temperature for generation
        max_tokens: Maximum tokens in response
        base_url: Optional base URL for API endpoint
        self_aware_prompt_builder: Optional self-aware prompt builder

    Returns:
        LLMService instance
    """
    return LLMService(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
        self_aware_prompt_builder=self_aware_prompt_builder,
    )
