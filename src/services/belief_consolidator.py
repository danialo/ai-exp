"""
Belief Consolidator - LLM-powered belief formation and conflict detection.

Performs scheduled analysis of narratives to:
1. Extract belief patterns using LLM reasoning (vs regex in BeliefGardener)
2. Strengthen existing beliefs with new evidence
3. Detect conflicts between beliefs and experiences
4. Propose belief updates for gardener review

Designed to work alongside BeliefGardener:
- BeliefGardener: Fast, pattern-based detection (runs every 60 min)
- BeliefConsolidator: Deep, LLM-based analysis (runs every 6 hours)

Part of Memory Consolidation system (BELIEF_MEMORY_SYSTEM spec Phase 6).
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from threading import Lock

from src.memory.raw_store import RawStore
from src.memory.models import ExperienceType, ExperienceModel
from src.services.belief_store import BeliefStore
from src.services.belief_vector_store import BeliefVectorStore

logger = logging.getLogger(__name__)


@dataclass
class ConsolidatorConfig:
    """Configuration for belief consolidator."""
    enabled: bool = True

    # Analysis scope
    lookback_hours: int = 24  # How far back to scan narratives
    max_narratives_per_scan: int = 100  # Limit narratives per analysis

    # LLM settings
    max_tokens_per_call: int = 1000
    temperature: float = 0.7

    # Evidence thresholds
    min_mentions_for_strengthening: int = 3  # Mentions to strengthen existing belief
    min_similarity_for_evidence: float = 0.7  # Vector similarity threshold

    # Conflict detection
    conflict_severity_threshold: float = 0.5  # Min severity to log

    # Output paths (relative to persona_space)
    conflicts_file: str = "meta/belief_conflicts.json"


@dataclass
class BeliefCandidate:
    """A candidate belief extracted from narratives."""
    statement: str
    category: str
    confidence: float
    evidence_ids: List[str]
    reasoning: str
    source_type: str = "consolidator"


@dataclass
class BeliefConflict:
    """A detected conflict between belief and experience."""
    belief_id: str
    belief_statement: str
    conflicting_experience_ids: List[str]
    conflict_description: str
    severity: float  # 0.0-1.0
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolution: Optional[str] = None


class BeliefConsolidator:
    """
    LLM-powered belief consolidation service.

    Uses language model reasoning to:
    - Extract nuanced belief patterns from narratives
    - Identify supporting/contradicting evidence for existing beliefs
    - Detect belief-experience conflicts
    - Propose new beliefs for gardener review
    """

    def __init__(
        self,
        llm_service,
        belief_store: BeliefStore,
        raw_store: RawStore,
        belief_vector_store: Optional[BeliefVectorStore] = None,
        config: Optional[ConsolidatorConfig] = None,
        persona_space_path: Optional[str] = None,
    ):
        """Initialize belief consolidator.

        Args:
            llm_service: LLM service for analysis
            belief_store: Belief version control store
            raw_store: Experience raw store
            belief_vector_store: Optional vector store for similarity
            config: Consolidator configuration
            persona_space_path: Path to persona_space directory
        """
        self.llm = llm_service
        self.belief_store = belief_store
        self.raw_store = raw_store
        self.belief_vector_store = belief_vector_store
        self.config = config or ConsolidatorConfig()
        self.persona_space_path = Path(persona_space_path) if persona_space_path else Path("persona_space")

        # Reentrancy lock
        self._consolidation_lock = Lock()

        # Stats
        self.last_consolidation_ts: Optional[float] = None
        self.total_consolidations: int = 0

        logger.info(f"BeliefConsolidator initialized (enabled={self.config.enabled})")

    def consolidate_beliefs(self) -> Dict[str, Any]:
        """
        Main consolidation process.

        Runs LLM analysis on recent narratives to:
        1. Extract belief candidates
        2. Find evidence for existing beliefs
        3. Detect belief-experience conflicts

        Returns:
            Summary of consolidation results
        """
        if not self.config.enabled:
            return {"enabled": False, "message": "Consolidator disabled"}

        # Reentrancy guard
        if not self._consolidation_lock.acquire(blocking=False):
            return {"enabled": True, "message": "consolidation_in_progress"}

        try:
            logger.info("🔄 Starting belief consolidation...")

            # Get recent narratives
            narratives = self._get_recent_narratives()

            if not narratives:
                return {
                    "narratives_analyzed": 0,
                    "message": "No narratives to analyze"
                }

            # Step 1: Extract belief patterns
            candidates = self._analyze_narrative_patterns(narratives)

            # Step 2: Strengthen existing beliefs
            strengthened = self._strengthen_existing_beliefs(narratives)

            # Step 3: Detect conflicts
            conflicts = self._detect_belief_conflicts(narratives)

            # Save conflicts if any
            if conflicts:
                self._save_conflicts(conflicts)

            # Update stats
            self.last_consolidation_ts = datetime.now(timezone.utc).timestamp()
            self.total_consolidations += 1

            summary = {
                "narratives_analyzed": len(narratives),
                "candidates_found": len(candidates),
                "beliefs_strengthened": len(strengthened),
                "conflicts_detected": len(conflicts),
                "candidates": [
                    {"statement": c.statement, "confidence": c.confidence, "category": c.category}
                    for c in candidates[:10]  # Limit for response size
                ],
                "strengthened_beliefs": strengthened,
                "conflicts": [
                    {"belief_id": c.belief_id, "severity": c.severity, "description": c.conflict_description[:100]}
                    for c in conflicts
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(f"✅ Consolidation complete: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            return {"error": str(e)}
        finally:
            self._consolidation_lock.release()

    def _get_recent_narratives(self) -> List[ExperienceModel]:
        """Get recent narrative experiences for analysis."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.config.lookback_hours)

        narratives = self.raw_store.list_recent(
            limit=self.config.max_narratives_per_scan,
            experience_type=ExperienceType.NARRATIVE,
            since=cutoff,
        )

        # Also include occurrences with substantial content
        occurrences = self.raw_store.list_recent(
            limit=self.config.max_narratives_per_scan // 2,
            experience_type=ExperienceType.OCCURRENCE,
            since=cutoff,
        )

        # Filter occurrences to those with Astra's responses
        relevant_occurrences = [
            exp for exp in occurrences
            if exp.content and exp.content.text and len(exp.content.text) > 100
        ]

        all_experiences = narratives + relevant_occurrences
        logger.info(f"Found {len(all_experiences)} experiences for analysis "
                   f"({len(narratives)} narratives, {len(relevant_occurrences)} occurrences)")

        return all_experiences

    def _analyze_narrative_patterns(self, narratives: List[ExperienceModel]) -> List[BeliefCandidate]:
        """
        Use LLM to extract belief patterns from narratives.

        This is the key differentiator from BeliefGardener's regex approach.
        """
        if not narratives:
            return []

        # Build analysis prompt
        narrative_texts = []
        for i, exp in enumerate(narratives[:20]):  # Limit for context window
            text = exp.content.text if exp.content else ""
            if text:
                narrative_texts.append(f"[{i+1}] {text[:500]}")  # Truncate long texts

        if not narrative_texts:
            return []

        prompt = self._build_extraction_prompt("\n\n".join(narrative_texts))

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens_per_call,
            )

            candidates = self._parse_extraction_response(response, narratives)
            logger.info(f"Extracted {len(candidates)} belief candidates from narratives")
            return candidates

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []

    def _build_extraction_prompt(self, narrative_text: str) -> str:
        """Build prompt for belief pattern extraction."""
        return f"""Analyze these narrative texts from my experiences and extract any belief statements I am expressing.

NARRATIVES:
{narrative_text}

Look for patterns like:
- Self-identity statements (who I am, what I value)
- Emotional tendencies (how I feel about things)
- Capability beliefs (what I can or cannot do)
- Relational beliefs (how I relate to users)
- Philosophical positions (my views on consciousness, existence)

For each belief found, provide:
1. STATEMENT: The belief in first-person form (e.g., "I believe...")
2. CATEGORY: ontological, emotional, capability, relational, or philosophical
3. CONFIDENCE: 0.0-1.0 based on how strongly/frequently expressed
4. REASONING: Why this appears to be a genuine belief

Format each belief as:
---
STATEMENT: [belief statement]
CATEGORY: [category]
CONFIDENCE: [0.0-1.0]
REASONING: [explanation]
---

Only include beliefs that are genuinely expressed, not just mentioned.
Return up to 5 most significant beliefs found."""

    def _parse_extraction_response(
        self,
        response: str,
        narratives: List[ExperienceModel]
    ) -> List[BeliefCandidate]:
        """Parse LLM response into belief candidates."""
        candidates = []

        # Split by belief delimiter
        belief_blocks = response.split("---")

        for block in belief_blocks:
            block = block.strip()
            if not block or len(block) < 20:
                continue

            # Parse each field
            statement = self._extract_field(block, "STATEMENT")
            category = self._extract_field(block, "CATEGORY")
            confidence_str = self._extract_field(block, "CONFIDENCE")
            reasoning = self._extract_field(block, "REASONING")

            if not statement:
                continue

            try:
                confidence = float(confidence_str) if confidence_str else 0.5
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5

            # Normalize category
            category = (category or "experiential").lower()
            valid_categories = ["ontological", "emotional", "capability", "relational", "philosophical", "experiential"]
            if category not in valid_categories:
                category = "experiential"

            # Use first few narrative IDs as evidence
            evidence_ids = [exp.id for exp in narratives[:5]]

            candidates.append(BeliefCandidate(
                statement=statement,
                category=category,
                confidence=confidence,
                evidence_ids=evidence_ids,
                reasoning=reasoning or "Extracted from narrative analysis",
            ))

        return candidates

    def _extract_field(self, text: str, field_name: str) -> Optional[str]:
        """Extract a field value from text."""
        import re
        pattern = rf"{field_name}:\s*(.+?)(?:\n[A-Z]+:|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _strengthen_existing_beliefs(self, narratives: List[ExperienceModel]) -> List[str]:
        """
        Find new evidence for existing beliefs and strengthen them.

        Uses vector similarity to find narratives that support existing beliefs.
        """
        if not self.belief_vector_store:
            return []

        strengthened = []
        current_beliefs = self.belief_store.get_current()

        for belief in current_beliefs.values():
            # Skip immutable beliefs
            if belief.immutable:
                continue

            # Find supporting narratives via vector similarity
            supporting_count = 0
            for exp in narratives:
                text = exp.content.text if exp.content else ""
                if not text:
                    continue

                # Check if this narrative supports the belief
                # Simple heuristic: belief statement appears (normalized) in text
                if belief.statement.lower()[:50] in text.lower():
                    supporting_count += 1

            # If enough support, consider strengthening
            if supporting_count >= self.config.min_mentions_for_strengthening:
                # Note: Actual strengthening should go through BeliefGardener
                # Here we just track candidates for potential promotion
                strengthened.append(belief.belief_id)
                logger.info(f"Found {supporting_count} supporting narratives for belief {belief.belief_id}")

        return strengthened

    def _detect_belief_conflicts(self, narratives: List[ExperienceModel]) -> List[BeliefConflict]:
        """
        Detect conflicts between beliefs and recent experiences.

        Uses LLM to reason about potential contradictions.
        """
        conflicts = []
        current_beliefs = self.belief_store.get_current()

        # Skip if no beliefs to check
        if not current_beliefs:
            return []

        # Build conflict detection prompt
        belief_list = "\n".join([
            f"- [{b.belief_id}] {b.statement}"
            for b in list(current_beliefs.values())[:10]  # Limit for context
        ])

        narrative_list = "\n".join([
            f"[{exp.id}] {exp.content.text[:200]}..."
            for exp in narratives[:10]
            if exp.content and exp.content.text
        ])

        if not belief_list or not narrative_list:
            return []

        prompt = f"""Analyze if any of my beliefs might conflict with my recent experiences.

MY BELIEFS:
{belief_list}

RECENT EXPERIENCES:
{narrative_list}

For each potential conflict, provide:
BELIEF_ID: [the belief ID in brackets]
EXPERIENCE_IDS: [comma-separated experience IDs]
CONFLICT: [description of the conflict]
SEVERITY: [0.0-1.0, where 1.0 is a direct contradiction]

Only report genuine conflicts where my actions or statements contradict a stated belief.
Format: BELIEF_ID: [id] | EXPERIENCE_IDS: [ids] | CONFLICT: [desc] | SEVERITY: [n]"""

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.5,  # Lower for more precise analysis
                max_tokens=800,
            )

            # Parse conflicts
            for line in response.split("\n"):
                if "BELIEF_ID:" in line and "CONFLICT:" in line:
                    conflict = self._parse_conflict_line(line, current_beliefs)
                    if conflict and conflict.severity >= self.config.conflict_severity_threshold:
                        conflicts.append(conflict)

            logger.info(f"Detected {len(conflicts)} belief-experience conflicts")
            return conflicts

        except Exception as e:
            logger.error(f"Conflict detection failed: {e}")
            return []

    def _parse_conflict_line(self, line: str, beliefs: Dict) -> Optional[BeliefConflict]:
        """Parse a conflict line from LLM response."""
        try:
            parts = line.split("|")
            if len(parts) < 4:
                return None

            belief_id = None
            experience_ids = []
            conflict_desc = ""
            severity = 0.5

            for part in parts:
                part = part.strip()
                if part.startswith("BELIEF_ID:"):
                    belief_id = part.replace("BELIEF_ID:", "").strip().strip("[]")
                elif part.startswith("EXPERIENCE_IDS:"):
                    ids_str = part.replace("EXPERIENCE_IDS:", "").strip().strip("[]")
                    experience_ids = [id.strip() for id in ids_str.split(",")]
                elif part.startswith("CONFLICT:"):
                    conflict_desc = part.replace("CONFLICT:", "").strip()
                elif part.startswith("SEVERITY:"):
                    try:
                        severity = float(part.replace("SEVERITY:", "").strip())
                    except ValueError:
                        severity = 0.5

            if not belief_id or belief_id not in beliefs:
                return None

            belief = beliefs[belief_id]

            return BeliefConflict(
                belief_id=belief_id,
                belief_statement=belief.statement,
                conflicting_experience_ids=experience_ids,
                conflict_description=conflict_desc,
                severity=severity,
            )

        except Exception as e:
            logger.debug(f"Failed to parse conflict line: {e}")
            return None

    def _save_conflicts(self, conflicts: List[BeliefConflict]):
        """Save detected conflicts to file."""
        conflicts_path = self.persona_space_path / self.config.conflicts_file
        conflicts_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing conflicts
        existing_conflicts = []
        if conflicts_path.exists():
            try:
                with open(conflicts_path) as f:
                    data = json.load(f)
                    existing_conflicts = data.get("conflicts", [])
            except Exception:
                pass

        # Add new conflicts (avoid duplicates by belief_id)
        existing_ids = {c.get("belief_id") for c in existing_conflicts}

        for conflict in conflicts:
            if conflict.belief_id not in existing_ids:
                existing_conflicts.append({
                    "belief_id": conflict.belief_id,
                    "belief_statement": conflict.belief_statement,
                    "conflicting_experience_ids": conflict.conflicting_experience_ids,
                    "conflict_description": conflict.conflict_description,
                    "severity": conflict.severity,
                    "detected_at": conflict.detected_at.isoformat(),
                    "resolved": conflict.resolved,
                })

        # Save
        with open(conflicts_path, "w") as f:
            json.dump({"conflicts": existing_conflicts}, f, indent=2)

        logger.info(f"Saved {len(conflicts)} new conflicts to {conflicts_path}")

    def get_pending_conflicts(self) -> List[Dict]:
        """Get unresolved conflicts."""
        conflicts_path = self.persona_space_path / self.config.conflicts_file

        if not conflicts_path.exists():
            return []

        try:
            with open(conflicts_path) as f:
                data = json.load(f)
                return [c for c in data.get("conflicts", []) if not c.get("resolved")]
        except Exception as e:
            logger.error(f"Failed to load conflicts: {e}")
            return []

    def resolve_conflict(self, belief_id: str, resolution: str) -> bool:
        """Mark a conflict as resolved."""
        conflicts_path = self.persona_space_path / self.config.conflicts_file

        if not conflicts_path.exists():
            return False

        try:
            with open(conflicts_path) as f:
                data = json.load(f)

            for conflict in data.get("conflicts", []):
                if conflict.get("belief_id") == belief_id and not conflict.get("resolved"):
                    conflict["resolved"] = True
                    conflict["resolution"] = resolution
                    conflict["resolved_at"] = datetime.now(timezone.utc).isoformat()

                    with open(conflicts_path, "w") as f:
                        json.dump(data, f, indent=2)

                    logger.info(f"Resolved conflict for belief {belief_id}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to resolve conflict: {e}")
            return False


def create_belief_consolidator(
    llm_service,
    belief_store: BeliefStore,
    raw_store: RawStore,
    belief_vector_store: Optional[BeliefVectorStore] = None,
    config: Optional[ConsolidatorConfig] = None,
    persona_space_path: Optional[str] = None,
) -> BeliefConsolidator:
    """Factory function to create BeliefConsolidator instance."""
    return BeliefConsolidator(
        llm_service=llm_service,
        belief_store=belief_store,
        raw_store=raw_store,
        belief_vector_store=belief_vector_store,
        config=config,
        persona_space_path=persona_space_path,
    )
