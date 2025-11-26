"""
Session Consolidator - Compresses raw session experiences into narrative summaries.

Part of Memory Consolidation Layer (Phase 4).

Process:
1. Gather experiences from ended session
2. Filter and prioritize by type
3. LLM summarization into narrative
4. Create NARRATIVE experience linked to originals
5. Mark originals as consolidated
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from sqlmodel import Session as DBSession, select

from src.memory.raw_store import RawStore
from src.memory.models import (
    ExperienceModel,
    ExperienceType,
    ContentModel,
    ProvenanceModel,
    Actor,
    CaptureMethod,
    Session,
    SessionStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationConfig:
    """Configuration for session consolidation."""
    # Session consolidation
    max_experiences_per_narrative: int = 50
    min_session_duration_seconds: int = 60
    max_llm_tokens: int = 2000

    # Experience prioritization
    priority_types: List[str] = field(default_factory=lambda: [
        "dissonance_event",
        "task_execution",
        "occurrence",
        "learning_pattern",
    ])


@dataclass
class NarrativeSummary:
    """Structured narrative summary from consolidation."""
    session_id: str
    summary: str
    key_interactions: List[str]
    emotional_arc: Dict[str, float]  # {start: valence, end: valence, peak: valence}
    unresolved_threads: List[str]
    experience_count: int
    duration_seconds: float


class SessionConsolidator:
    """
    Consolidates session experiences into narrative summaries.

    Creates compressed NARRATIVE experiences that preserve the essential
    information while reducing storage requirements.
    """

    def __init__(
        self,
        raw_store: RawStore,
        llm_service,
        config: Optional[ConsolidationConfig] = None,
    ):
        """Initialize session consolidator.

        Args:
            raw_store: Experience raw store
            llm_service: LLM service for summarization
            config: Consolidation configuration
        """
        self.raw_store = raw_store
        self.llm = llm_service
        self.config = config or ConsolidationConfig()

        logger.info("SessionConsolidator initialized")

    def get_unconsolidated_sessions(self) -> List[Session]:
        """Get sessions that need consolidation."""
        with DBSession(self.raw_store.engine) as db:
            statement = select(Session).where(
                Session.status == SessionStatus.ENDED.value,
            )
            sessions = db.exec(statement).all()

            # Filter to those without consolidated narratives
            unconsolidated = [
                s for s in sessions
                if not s.consolidated_narrative_id
            ]

            logger.info(f"Found {len(unconsolidated)} unconsolidated sessions")
            return unconsolidated

    async def consolidate_session(self, session_id: str) -> Optional[str]:
        """
        Consolidate a single session into a narrative.

        Args:
            session_id: ID of session to consolidate

        Returns:
            ID of created NARRATIVE experience, or None if consolidation failed
        """
        # Get session
        with DBSession(self.raw_store.engine) as db:
            session = db.get(Session, session_id)
            if not session:
                logger.error(f"Session {session_id} not found")
                return None

            if session.consolidated_narrative_id:
                logger.info(f"Session {session_id} already consolidated")
                return session.consolidated_narrative_id

        # Get experiences for this session
        experiences = self._get_session_experiences(session_id)

        if len(experiences) == 0:
            logger.warning(f"Session {session_id} has no experiences to consolidate")
            return None

        # Check minimum duration
        if session.end_time and session.start_time:
            duration = (session.end_time - session.start_time).total_seconds()
            if duration < self.config.min_session_duration_seconds:
                logger.info(f"Session {session_id} too short ({duration}s), skipping")
                return None
        else:
            duration = 0

        # Prioritize and limit experiences
        prioritized = self._prioritize_experiences(experiences)
        limited = prioritized[:self.config.max_experiences_per_narrative]

        # Generate narrative summary
        summary = await self._generate_narrative(limited, session_id, duration)

        if not summary:
            logger.error(f"Failed to generate narrative for session {session_id}")
            return None

        # Create NARRATIVE experience
        narrative_id = self._create_narrative_experience(summary, limited)

        # Mark session as consolidated
        self._mark_session_consolidated(session_id, narrative_id)

        # Mark experiences as consolidated
        self._mark_experiences_consolidated([exp.id for exp in limited])

        logger.info(f"Consolidated session {session_id} into narrative {narrative_id} "
                   f"({len(limited)} experiences)")

        return narrative_id

    def _get_session_experiences(self, session_id: str) -> List[ExperienceModel]:
        """Get all experiences for a session."""
        # Query by session_id
        experiences = self.raw_store.list_by_session(session_id)
        return experiences

    def _prioritize_experiences(self, experiences: List[ExperienceModel]) -> List[ExperienceModel]:
        """Prioritize experiences by type for inclusion in narrative."""
        def priority_score(exp: ExperienceModel) -> int:
            exp_type = exp.type.value if hasattr(exp.type, 'value') else str(exp.type)
            try:
                return len(self.config.priority_types) - self.config.priority_types.index(exp_type)
            except ValueError:
                return 0  # Unlisted types get lowest priority

        return sorted(experiences, key=priority_score, reverse=True)

    async def _generate_narrative(
        self,
        experiences: List[ExperienceModel],
        session_id: str,
        duration_seconds: float,
    ) -> Optional[NarrativeSummary]:
        """Generate narrative summary from experiences using LLM."""
        if not experiences:
            return None

        # Build context for LLM
        exp_texts = []
        valences = []

        for i, exp in enumerate(experiences[:20]):  # Limit for context window
            text = exp.content.text if exp.content else ""
            exp_type = exp.type.value if hasattr(exp.type, 'value') else str(exp.type)
            timestamp = exp.created_at.strftime("%H:%M:%S") if exp.created_at else "unknown"

            exp_texts.append(f"[{i+1}] [{exp_type}] {timestamp}: {text[:300]}")

            # Track valence for emotional arc
            if exp.affect and exp.affect.vad:
                valences.append(exp.affect.vad.v)

        # Calculate emotional arc
        if valences:
            emotional_arc = {
                "start": valences[0] if valences else 0.0,
                "end": valences[-1] if valences else 0.0,
                "peak": max(valences, key=abs) if valences else 0.0,
                "average": sum(valences) / len(valences),
            }
        else:
            emotional_arc = {"start": 0.0, "end": 0.0, "peak": 0.0, "average": 0.0}

        # Build prompt
        prompt = self._build_summarization_prompt(exp_texts, session_id)

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=self.config.max_llm_tokens,
            )

            # Parse response
            summary_text, key_interactions, unresolved = self._parse_narrative_response(response)

            return NarrativeSummary(
                session_id=session_id,
                summary=summary_text,
                key_interactions=key_interactions,
                emotional_arc=emotional_arc,
                unresolved_threads=unresolved,
                experience_count=len(experiences),
                duration_seconds=duration_seconds,
            )

        except Exception as e:
            logger.error(f"LLM narrative generation failed: {e}")
            # Fallback: simple concatenation
            return NarrativeSummary(
                session_id=session_id,
                summary=f"Session with {len(experiences)} experiences over {duration_seconds:.0f}s",
                key_interactions=[exp.content.text[:100] for exp in experiences[:5] if exp.content],
                emotional_arc=emotional_arc,
                unresolved_threads=[],
                experience_count=len(experiences),
                duration_seconds=duration_seconds,
            )

    def _build_summarization_prompt(self, exp_texts: List[str], session_id: str) -> str:
        """Build prompt for narrative summarization."""
        return f"""Summarize this conversation session into a narrative memory.

SESSION ID: {session_id}
EXPERIENCES:
{chr(10).join(exp_texts)}

Create a narrative summary that:
1. Captures the main topic and flow of the conversation
2. Highlights key moments or decisions
3. Notes any unresolved questions or threads

Format your response as:

SUMMARY:
[2-3 sentence narrative summary of what happened]

KEY_INTERACTIONS:
- [Key moment 1]
- [Key moment 2]
- [Key moment 3]

UNRESOLVED:
- [Any open questions or threads to revisit]
- [Or "None" if conversation concluded naturally]
"""

    def _parse_narrative_response(self, response: str) -> tuple:
        """Parse LLM response into structured components."""
        summary = ""
        key_interactions = []
        unresolved = []

        current_section = None

        for line in response.split("\n"):
            line = line.strip()

            if line.startswith("SUMMARY:"):
                current_section = "summary"
                content = line.replace("SUMMARY:", "").strip()
                if content:
                    summary = content
            elif line.startswith("KEY_INTERACTIONS:"):
                current_section = "key"
            elif line.startswith("UNRESOLVED:"):
                current_section = "unresolved"
            elif line.startswith("-"):
                item = line[1:].strip()
                if current_section == "key" and item.lower() != "none":
                    key_interactions.append(item)
                elif current_section == "unresolved" and item.lower() != "none":
                    unresolved.append(item)
            elif current_section == "summary" and line:
                summary += " " + line

        return summary.strip(), key_interactions, unresolved

    def _create_narrative_experience(
        self,
        summary: NarrativeSummary,
        original_experiences: List[ExperienceModel],
    ) -> str:
        """Create NARRATIVE experience from summary."""
        # Generate ID
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        narrative_id = f"narrative_{summary.session_id}_{timestamp}"

        # Build structured content
        structured = {
            "session_id": summary.session_id,
            "key_interactions": summary.key_interactions,
            "emotional_arc": summary.emotional_arc,
            "unresolved_threads": summary.unresolved_threads,
            "experience_count": summary.experience_count,
            "duration_seconds": summary.duration_seconds,
        }

        # Create experience
        narrative_exp = ExperienceModel(
            id=narrative_id,
            type=ExperienceType.NARRATIVE,
            created_at=datetime.now(timezone.utc),
            content=ContentModel(
                text=summary.summary,
                structured=structured,
            ),
            provenance=ProvenanceModel(
                sources=[],
                actor=Actor.AGENT,
                method=CaptureMethod.RECONCILE,
            ),
            parents=[exp.id for exp in original_experiences],
            consolidated=False,  # This IS the consolidated form
            session_id=summary.session_id,
        )

        # Store
        self.raw_store.append_experience(narrative_exp)

        return narrative_id

    def _mark_session_consolidated(self, session_id: str, narrative_id: str):
        """Mark session as consolidated."""
        with DBSession(self.raw_store.engine) as db:
            session = db.get(Session, session_id)
            if session:
                session.status = SessionStatus.CONSOLIDATED.value
                session.consolidated_narrative_id = narrative_id
                db.add(session)
                db.commit()

    def _mark_experiences_consolidated(self, experience_ids: List[str]):
        """Mark experiences as consolidated."""
        from src.memory.models import Experience

        with DBSession(self.raw_store.engine) as db:
            for exp_id in experience_ids:
                exp = db.get(Experience, exp_id)
                if exp:
                    exp.consolidated = True
                    db.add(exp)
            db.commit()

        logger.debug(f"Marked {len(experience_ids)} experiences as consolidated")

    async def consolidate_all_pending(self) -> Dict[str, Any]:
        """Consolidate all pending sessions."""
        sessions = self.get_unconsolidated_sessions()

        results = {
            "sessions_found": len(sessions),
            "consolidated": 0,
            "failed": 0,
            "skipped": 0,
            "narratives": [],
        }

        for session in sessions:
            try:
                narrative_id = await self.consolidate_session(session.id)
                if narrative_id:
                    results["consolidated"] += 1
                    results["narratives"].append(narrative_id)
                else:
                    results["skipped"] += 1
            except Exception as e:
                logger.error(f"Failed to consolidate session {session.id}: {e}")
                results["failed"] += 1

        logger.info(f"Consolidation complete: {results}")
        return results


def create_session_consolidator(
    raw_store: RawStore,
    llm_service,
    config: Optional[ConsolidationConfig] = None,
) -> SessionConsolidator:
    """Factory function to create SessionConsolidator."""
    return SessionConsolidator(
        raw_store=raw_store,
        llm_service=llm_service,
        config=config,
    )
