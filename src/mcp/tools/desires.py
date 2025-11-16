"""MCP tools for desire recording and management.

Tools:
- astra.desires.record: Record a new vague wish/desire
- astra.desires.list: List top desires by strength
- astra.desires.reinforce: Manually reinforce a desire (boost strength)
"""

import json
import logging
from typing import Any, Dict, Optional

from src.services.desire_store import DesireStore, create_desire_store

logger = logging.getLogger(__name__)


class DesireTools:
    """MCP tool handlers for desire management."""

    def __init__(self, desire_store: Optional[DesireStore] = None):
        """Initialize desire tools.

        Args:
            desire_store: DesireStore instance (creates default if None)
        """
        self.desire_store = desire_store or create_desire_store()

    def record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Record a new desire.

        Args:
            payload: {
                "text": str (required),
                "strength": float (default: 1.0, range: 0.0-1.0),
                "tags": list[str] (optional),
                "context": dict (optional)
            }

        Returns:
            {
                "success": bool,
                "desire_id": str,
                "desire": dict,
                "error": str (if failed)
            }
        """
        try:
            # Validate required fields
            if "text" not in payload:
                return {
                    "success": False,
                    "error": "Missing required field: text",
                }

            text = payload["text"]
            strength = payload.get("strength", 1.0)
            tags = payload.get("tags", [])
            context = payload.get("context", {})

            # Validate strength range
            if not 0.0 <= strength <= 1.0:
                return {
                    "success": False,
                    "error": f"Strength must be between 0.0 and 1.0, got {strength}",
                }

            # Record desire
            desire = self.desire_store.record(
                text=text,
                strength=strength,
                tags=tags,
                context=context,
            )

            logger.info(f"Recorded desire via MCP: {desire.id} (text: {text[:50]}...)")

            return {
                "success": True,
                "desire_id": desire.id,
                "desire": {
                    "id": desire.id,
                    "text": desire.text,
                    "strength": desire.strength,
                    "created_at": desire.created_at,
                    "last_reinforced_at": desire.last_reinforced_at,
                    "tags": desire.tags,
                    "context": desire.context,
                },
            }

        except Exception as e:
            logger.error(f"Unexpected error recording desire: {e}", exc_info=True)
            return {"success": False, "error": f"Internal error: {e}"}

    def list_desires(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """List desires sorted by strength.

        Args:
            payload: {
                "limit": int (default: 10),
                "min_strength": float (default: 0.0),
                "tag": str (optional, filter by tag)
            }

        Returns:
            {
                "success": bool,
                "desires": list[dict],
                "count": int
            }
        """
        try:
            payload = payload or {}
            limit = payload.get("limit", 10)
            min_strength = payload.get("min_strength", 0.0)
            tag = payload.get("tag")

            # Get desires (filtered by tag if specified)
            if tag:
                desires = self.desire_store.search_by_tag(tag)
                # Apply min_strength filter
                desires = [d for d in desires if d.strength >= min_strength]
                desires = desires[:limit]
            else:
                desires = self.desire_store.list_top(
                    limit=limit,
                    min_strength=min_strength,
                )

            return {
                "success": True,
                "desires": [
                    {
                        "id": d.id,
                        "text": d.text,
                        "strength": d.strength,
                        "created_at": d.created_at,
                        "tags": d.tags,
                    }
                    for d in desires
                ],
                "count": len(desires),
            }

        except Exception as e:
            logger.error(f"Unexpected error listing desires: {e}", exc_info=True)
            return {"success": False, "error": f"Internal error: {e}"}

    def reinforce(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Manually reinforce a desire to boost its strength.

        Args:
            payload: {
                "desire_id": str (required),
                "delta": float (default: 0.1)
            }

        Returns:
            {
                "success": bool,
                "desire_id": str,
                "new_strength": float,
                "error": str (if failed)
            }
        """
        try:
            if "desire_id" not in payload:
                return {
                    "success": False,
                    "error": "Missing required field: desire_id",
                }

            desire_id = payload["desire_id"]
            delta = payload.get("delta", 0.1)

            # Reinforce desire
            desire = self.desire_store.reinforce(desire_id, delta=delta)

            logger.info(f"Reinforced desire via MCP: {desire_id} -> {desire.strength}")

            return {
                "success": True,
                "desire_id": desire.id,
                "new_strength": desire.strength,
            }

        except ValueError as e:
            logger.warning(f"Desire reinforcement failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error reinforcing desire: {e}", exc_info=True)
            return {"success": False, "error": f"Internal error: {e}"}


def create_desire_tools(
    desire_store: Optional[DesireStore] = None,
) -> DesireTools:
    """Factory function to create DesireTools.

    Args:
        desire_store: Optional DesireStore instance

    Returns:
        Initialized DesireTools
    """
    return DesireTools(desire_store=desire_store)
