"""LLM service for generating responses with context from memories."""

import logging
from typing import Protocol, List, Dict, Any, Optional
from datetime import datetime

from openai import OpenAI

logger = logging.getLogger(__name__)


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
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
    ):
        """Initialize LLM service.

        Args:
            api_key: API key for the LLM provider
            model: Model name to use
            temperature: Temperature for generation (0-2)
            max_tokens: Maximum tokens in response
            base_url: Optional base URL for API endpoint (e.g., Venice AI)
            self_aware_prompt_builder: Optional builder for self-aware prompts
            top_k: Top-k sampling for creativity (if supported by API)
            top_p: Nucleus sampling parameter (0-1)
            presence_penalty: Penalty for token presence (-2 to 2)
            frequency_penalty: Penalty for token frequency (-2 to 2)
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.self_aware_prompt_builder = self_aware_prompt_builder
        self.top_k = top_k
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

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
            system_prompt: Optional system prompt to guide behavior (will be augmented with self-concept)
            include_self_awareness: Whether to include self-concept in system prompt

        Returns:
            Generated response text
        """
        # Build messages
        messages = []

        # Build self-aware system prompt if available
        if self.self_aware_prompt_builder and include_self_awareness:
            # Augment provided system prompt with self-awareness
            base_prompt = system_prompt if system_prompt else None
            system_prompt = self.self_aware_prompt_builder.build_self_aware_system_prompt(
                base_prompt=base_prompt
            )
        elif system_prompt is None:
            # No self-awareness builder, use default
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
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Add top_k if specified and supported
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k

        response = self.client.chat.completions.create(**kwargs)

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

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_k: int | None = None,
    ) -> str:
        """Simple generation without memory context or system prompt customization.

        Args:
            prompt: The prompt to generate from
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            top_k: Override default top_k

        Returns:
            Generated response text
        """
        # Detect reasoning models (GPT-5, O1, O3 series)
        is_reasoning_model = any(x in self.model.lower() for x in ["gpt-5", "o1", "o3"])
        max_tokens_param = "max_completion_tokens" if is_reasoning_model else "max_tokens"

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Only add max_tokens if explicitly set
        final_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        if final_max_tokens is not None:
            kwargs[max_tokens_param] = final_max_tokens

        # Only add temperature for non-reasoning models
        if not is_reasoning_model:
            kwargs["temperature"] = temperature if temperature is not None else self.temperature

            # Add top_k if specified
            if top_k is not None or self.top_k is not None:
                kwargs["top_k"] = top_k if top_k is not None else self.top_k

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: Dict[int, float] | None = None,
    ) -> Dict[str, Any]:
        """Generate response with OpenAI function calling / tools.

        Args:
            messages: List of message dicts (role, content)
            tools: List of tool definitions in OpenAI format
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            top_p: Override default top_p
            presence_penalty: Override default presence_penalty
            frequency_penalty: Override default frequency_penalty
            logit_bias: Token ID to bias mapping for suppressing specific tokens

        Returns:
            Dict with 'message' (full message object) and 'finish_reason'
        """
        # Some models (GPT-5, O1 series) use max_completion_tokens instead of max_tokens
        # and don't support temperature/top_p/penalties
        is_reasoning_model = any(x in self.model.lower() for x in ["gpt-5", "o1", "o3"])
        max_tokens_param = "max_completion_tokens" if is_reasoning_model else "max_tokens"

        kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
        }

        # Only add max_tokens if explicitly set
        final_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        if final_max_tokens is not None:
            kwargs[max_tokens_param] = final_max_tokens

        # Only add temperature/sampling params for non-reasoning models
        if not is_reasoning_model:
            kwargs["temperature"] = temperature if temperature is not None else self.temperature

            # Add optional parameters if specified
            if top_p is not None or self.top_p is not None:
                kwargs["top_p"] = top_p if top_p is not None else self.top_p

            if presence_penalty is not None or self.presence_penalty is not None:
                kwargs["presence_penalty"] = presence_penalty if presence_penalty is not None else self.presence_penalty

            if frequency_penalty is not None or self.frequency_penalty is not None:
                kwargs["frequency_penalty"] = frequency_penalty if frequency_penalty is not None else self.frequency_penalty

            if logit_bias is not None and len(logit_bias) > 0:
                kwargs["logit_bias"] = logit_bias

        # VERBOSE DIAGNOSTIC LOGGING
        tool_names = [t["function"]["name"] for t in tools] if tools else []
        has_execute_goal = "execute_goal" in tool_names
        last_msg = messages[-1].get("content", "")[:150] if messages else ""

        print("\n" + "="*80)
        print("🔧 LLM API CALL")
        print("="*80)
        print(f"Model: {self.model}")
        print(f"Provider: {self.client.base_url}")
        print(f"Tools count: {len(tools) if tools else 0}")
        print(f"Tool names: {tool_names}")
        print(f"execute_goal present: {has_execute_goal}")
        print(f"Messages count: {len(messages)}")
        print(f"Last user message: {last_msg}")
        print(f"Temperature: {kwargs.get('temperature', 'default')}")
        print(f"Max tokens: {kwargs.get('max_tokens', 'default')}")
        print("="*80 + "\n")

        logger.info(f"LLM API CALL: model={self.model}, tools={len(tools) if tools else 0}, execute_goal={has_execute_goal}")

        try:
            response = self.client.chat.completions.create(**kwargs)

            # LOG RESPONSE
            finish_reason = response.choices[0].finish_reason
            message = response.choices[0].message
            has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls is not None and len(message.tool_calls) > 0

            print("\n" + "="*80)
            print("🔧 LLM API RESPONSE")
            print("="*80)
            print(f"Finish reason: {finish_reason}")
            print(f"Has tool calls: {has_tool_calls}")
            if has_tool_calls:
                print(f"Tool calls count: {len(message.tool_calls)}")
                for tc in message.tool_calls:
                    print(f"  - {tc.function.name}()")
            else:
                print(f"Text response: {message.content[:200] if message.content else 'None'}...")
            print("="*80 + "\n")

            logger.info(f"LLM RESPONSE: finish={finish_reason}, tool_calls={has_tool_calls}")
        except Exception as e:
            logger.error(f"LLM API error for model {self.model}: {e}")
            logger.error(f"Request kwargs: {kwargs}")
            raise

        choice = response.choices[0]

        # Debug logging for reasoning models
        if any(x in self.model.lower() for x in ["gpt-5", "o1", "o3"]):
            logger.info(f"Reasoning model response - content: {repr(choice.message.content)}, refusal: {choice.message.refusal}")
            logger.info(f"Full message object: {choice.message}")
            logger.info(f"Finish reason: {choice.finish_reason}")

        return {
            "message": choice.message,
            "finish_reason": choice.finish_reason,
        }


def create_llm_service(
    api_key: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 500,
    base_url: str | None = None,
    self_aware_prompt_builder=None,
    top_k: int | None = None,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
) -> LLMService:
    """Factory function to create an LLM service.

    Args:
        api_key: API key for the LLM provider
        model: Model name to use
        temperature: Temperature for generation
        max_tokens: Maximum tokens in response
        base_url: Optional base URL for API endpoint
        self_aware_prompt_builder: Optional self-aware prompt builder
        top_k: Top-k sampling for creativity
        top_p: Nucleus sampling parameter
        presence_penalty: Penalty for token presence
        frequency_penalty: Penalty for token frequency

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
        top_k=top_k,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )
