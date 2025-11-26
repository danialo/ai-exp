"""
Insight Extractor - Extracts generalizable patterns from consolidated narratives.

Part of Memory Consolidation Layer (Phase 4).

Process:
1. Collect recent narratives
2. Cluster by semantic similarity
3. LLM analysis for recurring patterns
4. Generate LEARNING_PATTERN experiences
5. Feed to Belief Gardener
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from src.memory.raw_store import RawStore
from src.memory.models import (
    ExperienceModel,
    ExperienceType,
    ContentModel,
    ProvenanceModel,
    Actor,
    CaptureMethod,
)

logger = logging.getLogger(__name__)


@dataclass
class InsightConfig:
    """Configuration for insight extraction."""
    lookback_days: int = 7
    min_narratives_for_insight: int = 3
    max_narratives_per_analysis: int = 20
    max_llm_tokens: int = 1500
    min_pattern_confidence: float = 0.5


@dataclass
class ExtractedInsight:
    """A pattern extracted from narratives."""
    pattern_text: str
    category: str  # ontological, relational, capability, behavioral
    confidence: float
    evidence_ids: List[str]  # Narrative IDs
    reasoning: str


class InsightExtractor:
    """
    Extracts generalizable patterns from consolidated narratives.

    Identifies recurring themes, behaviors, and patterns that may
    warrant belief formation.
    """

    def __init__(
        self,
        raw_store: RawStore,
        llm_service,
        embedding_provider=None,
        config: Optional[InsightConfig] = None,
    ):
        """Initialize insight extractor.

        Args:
            raw_store: Experience raw store
            llm_service: LLM service for pattern analysis
            embedding_provider: Optional embedding provider for clustering
            config: Extraction configuration
        """
        self.raw_store = raw_store
        self.llm = llm_service
        self.embedding_provider = embedding_provider
        self.config = config or InsightConfig()

        logger.info("InsightExtractor initialized")

    def get_recent_narratives(self) -> List[ExperienceModel]:
        """Get recent narrative experiences for analysis."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.lookback_days)

        narratives = self.raw_store.list_recent(
            limit=self.config.max_narratives_per_analysis,
            experience_type=ExperienceType.NARRATIVE,
            since=cutoff,
        )

        logger.info(f"Found {len(narratives)} narratives in last {self.config.lookback_days} days")
        return narratives

    async def extract_insights(self) -> List[ExtractedInsight]:
        """
        Extract insights from recent narratives.

        Returns:
            List of extracted insights
        """
        narratives = self.get_recent_narratives()

        if len(narratives) < self.config.min_narratives_for_insight:
            logger.info(f"Not enough narratives for insight extraction "
                       f"({len(narratives)} < {self.config.min_narratives_for_insight})")
            return []

        # Build context for LLM
        narrative_texts = []
        for i, narr in enumerate(narratives):
            text = narr.content.text if narr.content else ""
            structured = narr.content.structured if narr.content else {}

            # Include key interactions if available
            key_interactions = structured.get("key_interactions", [])
            interactions_str = "; ".join(key_interactions[:3]) if key_interactions else ""

            narrative_texts.append(
                f"[{i+1}] {text[:200]}..."
                + (f" Key moments: {interactions_str}" if interactions_str else "")
            )

        # LLM pattern extraction
        prompt = self._build_extraction_prompt(narrative_texts)

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.6,
                max_tokens=self.config.max_llm_tokens,
            )

            insights = self._parse_insights(response, narratives)

            # Filter by confidence
            filtered = [i for i in insights if i.confidence >= self.config.min_pattern_confidence]

            logger.info(f"Extracted {len(filtered)} insights from {len(narratives)} narratives")
            return filtered

        except Exception as e:
            logger.error(f"Insight extraction failed: {e}")
            return []

    def _build_extraction_prompt(self, narrative_texts: List[str]) -> str:
        """Build prompt for pattern extraction."""
        return f"""Analyze these conversation summaries and identify recurring patterns.

RECENT CONVERSATIONS:
{chr(10).join(narrative_texts)}

Look for:
1. ONTOLOGICAL patterns - recurring self-descriptions or identity statements
2. RELATIONAL patterns - how interactions typically unfold
3. CAPABILITY patterns - what tasks/topics come up repeatedly
4. BEHAVIORAL patterns - consistent response styles or approaches

For each pattern found, provide:
PATTERN: [The pattern in first-person if about self]
CATEGORY: [ontological/relational/capability/behavioral]
CONFIDENCE: [0.0-1.0 based on how consistent the pattern is]
REASONING: [Why this appears to be a genuine pattern]

Format each as:
---
PATTERN: [pattern text]
CATEGORY: [category]
CONFIDENCE: [0.0-1.0]
REASONING: [explanation]
---

Only include genuine patterns that appear across multiple conversations.
Return up to 5 most significant patterns."""

    def _parse_insights(
        self,
        response: str,
        narratives: List[ExperienceModel]
    ) -> List[ExtractedInsight]:
        """Parse LLM response into insights."""
        insights = []

        blocks = response.split("---")

        for block in blocks:
            block = block.strip()
            if not block or len(block) < 20:
                continue

            pattern = self._extract_field(block, "PATTERN")
            category = self._extract_field(block, "CATEGORY")
            confidence_str = self._extract_field(block, "CONFIDENCE")
            reasoning = self._extract_field(block, "REASONING")

            if not pattern:
                continue

            try:
                confidence = float(confidence_str) if confidence_str else 0.5
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5

            # Normalize category
            category = (category or "behavioral").lower()
            valid_categories = ["ontological", "relational", "capability", "behavioral"]
            if category not in valid_categories:
                category = "behavioral"

            # Use narrative IDs as evidence
            evidence_ids = [n.id for n in narratives[:5]]

            insights.append(ExtractedInsight(
                pattern_text=pattern,
                category=category,
                confidence=confidence,
                evidence_ids=evidence_ids,
                reasoning=reasoning or "",
            ))

        return insights

    def _extract_field(self, text: str, field_name: str) -> Optional[str]:
        """Extract a field value from text."""
        import re
        pattern = rf"{field_name}:\s*(.+?)(?:\n[A-Z]+:|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    async def extract_and_store(self) -> Dict[str, Any]:
        """
        Extract insights and store as LEARNING_PATTERN experiences.

        Returns:
            Summary of extraction results
        """
        insights = await self.extract_insights()

        results = {
            "insights_found": len(insights),
            "patterns_stored": 0,
            "pattern_ids": [],
        }

        for insight in insights:
            pattern_id = self._store_as_learning_pattern(insight)
            if pattern_id:
                results["patterns_stored"] += 1
                results["pattern_ids"].append(pattern_id)

        logger.info(f"Insight extraction complete: {results}")
        return results

    def _store_as_learning_pattern(self, insight: ExtractedInsight) -> Optional[str]:
        """Store insight as LEARNING_PATTERN experience."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        pattern_id = f"insight_{insight.category}_{timestamp}"

        structured = {
            "pattern_text": insight.pattern_text,
            "category": insight.category,
            "confidence": insight.confidence,
            "evidence_count": len(insight.evidence_ids),
            "reasoning": insight.reasoning,
        }

        experience = ExperienceModel(
            id=pattern_id,
            type=ExperienceType.LEARNING_PATTERN,
            created_at=datetime.now(timezone.utc),
            content=ContentModel(
                text=f"Pattern detected: {insight.pattern_text}",
                structured=structured,
            ),
            provenance=ProvenanceModel(
                sources=[],
                actor=Actor.AGENT,
                method=CaptureMethod.MODEL_INFER,
            ),
            parents=insight.evidence_ids,
        )

        try:
            self.raw_store.append_experience(experience)
            return pattern_id
        except Exception as e:
            logger.error(f"Failed to store learning pattern: {e}")
            return None


def create_insight_extractor(
    raw_store: RawStore,
    llm_service,
    embedding_provider=None,
    config: Optional[InsightConfig] = None,
) -> InsightExtractor:
    """Factory function to create InsightExtractor."""
    return InsightExtractor(
        raw_store=raw_store,
        llm_service=llm_service,
        embedding_provider=embedding_provider,
        config=config,
    )
