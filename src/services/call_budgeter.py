"""Call budgeting and chunking for LLM operations that may exceed context limits.

Single responsibility: Given a pile of items and a context window, determine how many
LLM calls are needed and what goes in each one.

Does NOT make LLM calls itself - only creates plans.
"""

from dataclasses import dataclass
from typing import List, Callable, Any


@dataclass
class ChunkPlan:
    """Plan for a single LLM call in a chunked operation."""
    call_index: int
    item_indices: List[int]  # which items from original list to include
    prompt_token_estimate: int
    response_token_budget: int


class CallBudgeter:
    """Plans chunked LLM calls to stay within context limits.

    Use this when you need to process more items than fit in a single context window.
    The budgeter will create a plan for multiple calls, each within budget.

    Example:
        budgeter = CallBudgeter(max_tokens_per_call=8000, safety_margin=0.8)
        plans = budgeter.plan_chunked_calls(
            items=docs,
            estimate_tokens_for_item=lambda d: len(d["content"]) // 4,
            base_prompt_tokens=500,
            desired_response_tokens=512,
        )

        for plan in plans:
            subset = [docs[i] for i in plan.item_indices]
            response = llm.call(subset, max_tokens=plan.response_token_budget)
    """

    def __init__(self, max_tokens_per_call: int, safety_margin: float = 0.8):
        """Initialize call budgeter.

        Args:
            max_tokens_per_call: Hard context limit for the model
            safety_margin: Fraction of max to actually use (0.8 = 80% utilization)
                          Leaves headroom for tokenization variance and safety
        """
        self.max_tokens_per_call = max_tokens_per_call
        self.safety_margin = safety_margin
        self.max_effective = int(max_tokens_per_call * safety_margin)

    def plan_chunked_calls(
        self,
        items: List[Any],
        estimate_tokens_for_item: Callable[[Any], int],
        base_prompt_tokens: int,
        desired_response_tokens: int,
    ) -> List[ChunkPlan]:
        """Create a plan for chunked LLM calls.

        Uses greedy bin packing to fit items into calls while respecting token limits.

        Args:
            items: List of items to process (docs, claims, beliefs, etc.)
            estimate_tokens_for_item: Function that estimates tokens for one item
            base_prompt_tokens: Fixed cost of system prompt + instructions + headers
            desired_response_tokens: Target output budget per call

        Returns:
            List of ChunkPlan, one per LLM call needed

        Algorithm:
            1. For each item, try to add it to current chunk
            2. If adding would exceed limit, finalize current chunk and start new one
            3. If single item is too large, give it solo call with reduced response budget
            4. Return all chunk plans
        """
        if not items:
            return []

        plans = []
        current_indices = []
        current_tokens = base_prompt_tokens + desired_response_tokens
        call_index = 0

        for idx, item in enumerate(items):
            item_tokens = estimate_tokens_for_item(item)

            # Special case: single item too large even for solo call
            min_needed = base_prompt_tokens + item_tokens + desired_response_tokens
            if min_needed > self.max_effective:
                # Finalize current chunk if any
                if current_indices:
                    plans.append(ChunkPlan(
                        call_index=call_index,
                        item_indices=current_indices.copy(),
                        prompt_token_estimate=current_tokens,
                        response_token_budget=desired_response_tokens,
                    ))
                    call_index += 1

                # Force item into solo call with smaller response budget
                reduced_response_budget = max(256, self.max_effective - base_prompt_tokens - item_tokens)
                plans.append(ChunkPlan(
                    call_index=call_index,
                    item_indices=[idx],
                    prompt_token_estimate=base_prompt_tokens + item_tokens,
                    response_token_budget=reduced_response_budget,
                ))
                call_index += 1

                # Reset for next chunk
                current_indices = []
                current_tokens = base_prompt_tokens + desired_response_tokens
                continue

            # Try adding item to current chunk
            tentative_tokens = current_tokens + item_tokens

            if tentative_tokens > self.max_effective and current_indices:
                # Current chunk is full, finalize it
                plans.append(ChunkPlan(
                    call_index=call_index,
                    item_indices=current_indices.copy(),
                    prompt_token_estimate=current_tokens,
                    response_token_budget=desired_response_tokens,
                ))
                call_index += 1

                # Start new chunk with this item
                current_indices = [idx]
                current_tokens = base_prompt_tokens + desired_response_tokens + item_tokens
            else:
                # Add to current chunk
                current_indices.append(idx)
                current_tokens = tentative_tokens

        # Finalize last chunk if any
        if current_indices:
            plans.append(ChunkPlan(
                call_index=call_index,
                item_indices=current_indices.copy(),
                prompt_token_estimate=current_tokens,
                response_token_budget=desired_response_tokens,
            ))

        return plans

    def estimate_total_calls(
        self,
        items: List[Any],
        estimate_tokens_for_item: Callable[[Any], int],
        base_prompt_tokens: int,
        desired_response_tokens: int,
    ) -> int:
        """Quick estimate of how many calls will be needed without creating full plan.

        Useful for deciding whether to chunk at all, or for progress indicators.
        """
        plans = self.plan_chunked_calls(
            items=items,
            estimate_tokens_for_item=estimate_tokens_for_item,
            base_prompt_tokens=base_prompt_tokens,
            desired_response_tokens=desired_response_tokens,
        )
        return len(plans)


def estimate_tokens_from_text(text: str) -> int:
    """Rough heuristic for token estimation from text.

    Uses ~4 chars per token as rough estimate (conservative for English).
    For production, consider using tiktoken for exact counts.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    return max(1, len(text) // 4)


def estimate_tokens_from_dict(obj: dict, key: str = "content") -> int:
    """Estimate tokens for a dict with text content.

    Args:
        obj: Dictionary containing text
        key: Key to extract text from (default: "content")

    Returns:
        Estimated token count
    """
    text = obj.get(key, "")
    if isinstance(text, str):
        return estimate_tokens_from_text(text)
    return 0
