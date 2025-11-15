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

        # Add call budgeter for chunking large operations
        # Estimate context limit at 2x max_tokens (conservative for models with larger context)
        from src.services.call_budgeter import CallBudgeter
        self.call_budgeter = CallBudgeter(
            max_tokens_per_call=max_tokens * 4,  # Assume 4x for context window
            safety_margin=0.8
        )

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

    def _naive_token_estimate(self, text: str) -> int:
        """Crude but safe token estimator: ~4 chars per token."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _completion(self, system_prompt: str, user_prompt: str, max_tokens: int, response_format: Optional[Dict] = None) -> str:
        """Low-level completion wrapper."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,  # Lower temp for structured synthesis
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _parse_json_or_die(self, text: str) -> Dict[str, Any]:
        """Parse JSON response, with fallback structure on failure."""
        import json
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}. Raw text: {text[:500]}")
            # Return minimal valid structure
            return {
                "narrative_summary": "Synthesis failed - JSON parse error",
                "key_events": [],
                "contested_claims": [],
                "open_questions": [],
                "coverage_stats": {"error": str(e)}
            }

    # === Research Synthesis with Chunking ===

    def summarize_research_session(
        self,
        root_question: str,
        docs: List[Dict[str, Any]],
        tasks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Main entry point for research synthesis with automatic chunking.

        Args:
            root_question: The original research question
            docs: List of SourceDoc dicts with content, claims, url, etc.
            tasks: List of executed HTN tasks

        Returns:
            dict with: narrative_summary, key_events, contested_claims,
                       open_questions, coverage_stats
        """
        import json

        if not docs:
            return {
                "narrative_summary": "No sources found for this question",
                "key_events": [],
                "contested_claims": [],
                "open_questions": [],
                "coverage_stats": {"sources_investigated": 0}
            }

        system_prompt = (
            "You are Astra's research synthesis module. "
            "You receive multiple source documents, each with claims and summaries, "
            "plus a list of research tasks that were executed. "
            "Your job is to synthesize a structured view of what was found."
        )

        instructions = (
            f"Root research question:\n{root_question}\n\n"
            "You will be given a batch of source docs from this research session.\n"
            "For this batch, produce a structured summary with the following keys:\n"
            "narrative_summary, key_events, contested_claims, open_questions, coverage_stats.\n"
            "Do NOT speculate beyond what the docs support.\n"
        )

        # Build text payloads for each doc
        doc_payloads = []
        for d in docs:
            claims_text = ""
            for c in d.get("claims", []):
                if isinstance(c, dict):
                    claims_text += f"- {c.get('claim', c.get('text', str(c)))}\n"
                else:
                    claims_text += f"- {c}\n"

            payload = (
                f"ID: {d.get('id', 'unknown')}\n"
                f"Title: {d.get('title', 'N/A')}\n"
                f"URL: {d.get('url', 'N/A')}\n"
                f"Published: {d.get('published_at', 'N/A')}\n"
                f"Summary: {d.get('content_summary', d.get('content', 'N/A')[:500])}\n"
                f"Claims:\n{claims_text}\n"
            )
            doc_payloads.append(payload)

        # Decide single call vs chunked calls
        partials = self._chunked_summarize_docs(
            system_prompt=system_prompt,
            instructions=instructions,
            doc_payloads=doc_payloads,
            root_question=root_question,
        )

        if len(partials) == 1:
            return partials[0]

        return self._merge_partial_summaries(
            root_question=root_question,
            partial_summaries=partials,
        )

    def _chunked_summarize_docs(
        self,
        system_prompt: str,
        instructions: str,
        doc_payloads: List[str],
        root_question: str,
        desired_response_tokens: int = 512,
    ) -> List[Dict[str, Any]]:
        """Map phase: Use CallBudgeter to split docs into chunks if needed."""

        base_prompt_tokens = (
            self._naive_token_estimate(system_prompt) +
            self._naive_token_estimate(instructions)
        )

        # Precompute token estimates for docs
        items = []
        for idx, text in enumerate(doc_payloads):
            items.append({
                "index": idx,
                "token_estimate": self._naive_token_estimate(text),
            })

        plans = self.call_budgeter.plan_chunked_calls(
            items=items,
            estimate_tokens_for_item=lambda it: it["token_estimate"],
            base_prompt_tokens=base_prompt_tokens,
            desired_response_tokens=desired_response_tokens,
        )

        partial_summaries: List[Dict[str, Any]] = []

        logger.info(
            f"Research synthesis: chunks={len(plans)} docs={len(doc_payloads)} question={repr(root_question[:80])}"
        )

        for plan in plans:
            subset_payloads = [
                doc_payloads[it["index"]] for it in items
                if it["index"] in plan.item_indices
            ]

            # Build user prompt for this chunk
            docs_block = "\n\n".join(subset_payloads)
            user_prompt = (
                instructions +
                "\n\nHere is a batch of source documents from the research session:\n\n" +
                docs_block +
                "\n\nReturn JSON with the following keys only: "
                "narrative_summary, key_events, contested_claims, open_questions, coverage_stats."
            )

            raw = self._completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=plan.response_token_budget,
                response_format={"type": "json_object"}
            )

            summary_obj = self._parse_json_or_die(raw)
            # Annotate that this is a partial summary
            summary_obj.setdefault("coverage_stats", {})
            summary_obj["coverage_stats"]["docs_in_batch"] = len(subset_payloads)
            summary_obj["coverage_stats"]["chunk_index"] = plan.call_index
            partial_summaries.append(summary_obj)

        return partial_summaries

    def _merge_partial_summaries(
        self,
        root_question: str,
        partial_summaries: List[Dict[str, Any]],
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Reduce phase: Merge multiple partial summaries into final synthesis."""
        import json

        system_prompt = (
            "You are Astra's global research synthesis module. "
            "You receive multiple partial summaries, each already structured, "
            "and must merge them into a single coherent global summary."
        )

        # Turn partials into text block
        parts = []
        for i, s in enumerate(partial_summaries):
            parts.append(
                f"PART {i+1}:\n"
                f"narrative_summary:\n{s.get('narrative_summary', '')}\n\n"
                f"key_events:\n{json.dumps(s.get('key_events', []), indent=2)}\n\n"
                f"contested_claims:\n{json.dumps(s.get('contested_claims', []), indent=2)}\n\n"
                f"open_questions:\n{json.dumps(s.get('open_questions', []), indent=2)}\n\n"
                f"coverage_stats:\n{json.dumps(s.get('coverage_stats', {}), indent=2)}\n"
            )
        partials_block = "\n\n".join(parts)

        user_prompt = (
            f"Root research question:\n{root_question}\n\n"
            "You are given several partial summaries from different batches of source documents.\n"
            "Your job is to merge them into a single structured summary with the same schema:\n"
            "narrative_summary, key_events, contested_claims, open_questions, coverage_stats.\n\n"
            "Unify overlapping events, consolidate contested claims, remove duplicates, and compute global coverage "
            "stats (total docs, total claims, count of contested claims, etc).\n\n"
            f"Partial summaries:\n\n{partials_block}\n\n"
            "Return only JSON with the keys: narrative_summary, key_events, contested_claims, open_questions, "
            "coverage_stats."
        )

        raw = self._completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        merged = self._parse_json_or_die(raw)
        return merged


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
