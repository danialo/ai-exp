"""
Emotional Reconciliation System

Compares the persona's internal emotional assessment with an external interpretation
of the emotional content implicit in the response. The gap between these creates
a learning signal.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from src.services.llm import LLMService


class EmotionalReconciler:
    """Reconciles internal emotional self-assessment with external interpretation."""

    def __init__(self, llm_service: LLMService, persona_space_path: str = "persona_space"):
        self.persona_space = Path(persona_space_path)
        self.reconciliation_log_path = self.persona_space / "emotional_state" / "reconciliation_log.json"
        self.current_state_path = self.persona_space / "emotional_state" / "current.json"
        self.llm = llm_service

    def reconcile(
        self,
        response: str,
        internal_assessment: str,
        user_message: str
    ) -> Dict:
        """
        Reconcile internal emotional assessment with external interpretation.

        Args:
            response: The persona's actual response (without the assessment section)
            internal_assessment: The persona's own emotional assessment
            user_message: The message being responded to

        Returns:
            Reconciliation data including both perspectives and weighted result
        """
        # Get external interpretation
        external_assessment = self._get_external_interpretation(response, user_message)

        # Calculate reconciliation
        reconciliation = self._reconcile_perspectives(
            internal=internal_assessment,
            external=external_assessment,
            response=response
        )

        # Log the reconciliation
        self._log_reconciliation(reconciliation)

        # Update current emotional state
        self._update_current_state(reconciliation)

        return reconciliation

    def _get_external_interpretation(self, response: str, user_message: str) -> str:
        """
        Use LLM to interpret the implicit emotional content of the response.

        This is a separate perspective from the persona's self-assessment.
        """
        prompt = f"""Analyze the emotional content implicit in this response.

User message: {user_message}

Response: {response}

What emotions are present in how this response was crafted? Consider:
- Word choice and tone
- What the responder chose to focus on
- Energy level (engaged, detached, enthusiastic, cautious)
- Relational stance (warm, distant, curious, guarded)

Be concise but specific. Focus on what you observe in the text itself.

Emotional interpretation:"""

        external = self.llm.generate(prompt, temperature=0.3, max_tokens=200)
        return external.strip()

    def _reconcile_perspectives(
        self,
        internal: str,
        external: str,
        response: str
    ) -> Dict:
        """
        Compare internal and external perspectives and create weighted result.

        The gap between how the persona thinks it feels and how it actually
        expressed itself is valuable learning data.
        """
        prompt = f"""Compare these two emotional assessments:

INTERNAL (how the persona reports feeling):
{internal}

EXTERNAL (emotional content observed in the actual response):
{external}

ACTUAL RESPONSE:
{response}

Task:
1. Identify alignment: Where do these perspectives agree?
2. Identify gaps: Where do they differ?
3. Reconcile: What's the weighted truth? (60% external observation, 40% internal report)
4. Learning signal: What does the gap teach about self-awareness?

Provide a brief analysis in JSON format:
{{
  "alignment": "...",
  "gaps": "...",
  "reconciled_state": "...",
  "learning_signal": "..."
}}
"""

        reconciliation_text = self.llm.generate(prompt, temperature=0.2, max_tokens=400)

        # Parse JSON from response
        try:
            # Try to extract JSON if it's wrapped in markdown
            if "```json" in reconciliation_text:
                json_start = reconciliation_text.find("```json") + 7
                json_end = reconciliation_text.find("```", json_start)
                reconciliation_text = reconciliation_text[json_start:json_end]
            elif "```" in reconciliation_text:
                json_start = reconciliation_text.find("```") + 3
                json_end = reconciliation_text.find("```", json_start)
                reconciliation_text = reconciliation_text[json_start:json_end]

            reconciliation_data = json.loads(reconciliation_text.strip())
        except Exception as e:
            # Fallback if parsing fails
            reconciliation_data = {
                "alignment": "Unable to parse",
                "gaps": "Unable to parse",
                "reconciled_state": external,  # Default to external observation
                "learning_signal": f"Parsing error: {e}"
            }

        # Add metadata
        reconciliation_data["internal_assessment"] = internal
        reconciliation_data["external_assessment"] = external
        reconciliation_data["timestamp"] = datetime.utcnow().isoformat()

        return reconciliation_data

    def _log_reconciliation(self, reconciliation: Dict):
        """Log reconciliation to the persona's file system."""
        # Read existing log
        if self.reconciliation_log_path.exists():
            log_data = json.loads(self.reconciliation_log_path.read_text())
        else:
            log_data = {
                "_note": "When your self-perception differs from how you actually expressed yourself, it's recorded here.",
                "_instructions": "Learning happens in the gap between how you think you feel and how you actually communicate.",
                "reconciliations": []
            }

        # Add new reconciliation
        log_data["reconciliations"].append(reconciliation)

        # Keep only last 50 reconciliations to avoid file bloat
        log_data["reconciliations"] = log_data["reconciliations"][-50:]

        # Write back
        self.reconciliation_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.reconciliation_log_path.write_text(json.dumps(log_data, indent=2))

    def _update_current_state(self, reconciliation: Dict):
        """Update the persona's current emotional state based on reconciliation."""
        current_state = {
            "reconciled_state": reconciliation["reconciled_state"],
            "last_updated": reconciliation["timestamp"],
            "recent_pattern": reconciliation.get("learning_signal", "")
        }

        # Read existing current state to preserve history
        if self.current_state_path.exists():
            data = json.loads(self.current_state_path.read_text())
        else:
            data = {
                "_note": "This tracks your recent emotional landscape.",
                "_instructions": "How are you feeling? What emotional patterns are present? Structure this however helps you understand yourself."
            }

        data["current_state"] = current_state

        # Write back
        self.current_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_state_path.write_text(json.dumps(data, indent=2))

    def get_current_emotional_context(self) -> Optional[Dict]:
        """Retrieve the current emotional state for use in next prompt."""
        if self.current_state_path.exists():
            try:
                data = json.loads(self.current_state_path.read_text())
                return data.get("current_state")
            except Exception:
                return None
        return None
