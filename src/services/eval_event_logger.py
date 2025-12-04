"""
Eval Event Logger for HTN Self-Belief Decomposer.

Logs pipeline telemetry to JSONL files for debugging and regression analysis.
Each pipeline run produces a complete event capturing all stages.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.belief_config import BeliefSystemConfig, get_belief_config

logger = logging.getLogger(__name__)


@dataclass
class SourceContextEvent:
    """Source context information for the experience."""
    mode: str
    source_weight: float
    context_id: str
    vad_used: bool = False
    heuristics_used: bool = False
    penalties_applied: List[str] = field(default_factory=list)


@dataclass
class SegmentationEvent:
    """Segmentation stage results."""
    candidate_count: int


@dataclass
class InvalidAtom:
    """Record of an invalid atom and why it was rejected."""
    text: str
    reason: str


@dataclass
class AtomsEvent:
    """Atomization stage results."""
    raw_count: int
    valid_count: int
    deduped_count: int
    invalid: List[InvalidAtom] = field(default_factory=list)


@dataclass
class EpistemicsAtomEvent:
    """Epistemics extraction for a single atom."""
    atom_text: str
    frame: Dict[str, Any]
    confidence: float
    signals: List[Dict[str, Any]]
    used_llm_fallback: bool


@dataclass
class ResolutionAtomEvent:
    """Resolution results for a single atom."""
    atom_text: str
    outcome: str  # match, no_match, uncertain
    match_confidence: float
    candidate_ids: List[str]
    belief_id_final: str
    created_tentative_link: bool


@dataclass
class ConflictsEvent:
    """Conflict detection results."""
    hard_count: int
    tension_count: int
    edges_created: List[str]


@dataclass
class NodeUpdate:
    """Update to a single belief node."""
    belief_id: str
    activation_delta: float
    core_score_delta: float
    status_before: str
    status_after: str


@dataclass
class ScoringEvent:
    """Scoring updates."""
    nodes_updated: List[NodeUpdate]


@dataclass
class StreamAssignmentEvent:
    """Stream assignment for a single belief."""
    belief_id: str
    primary: str
    secondary: Optional[str]
    confidence: float
    migrated: bool


@dataclass
class StreamEvent:
    """Stream assignment results."""
    assignments: List[StreamAssignmentEvent]


@dataclass
class TentativeLinksEvent:
    """Tentative link operations."""
    created: List[str]
    updated: List[str]
    auto_accepted: List[str]
    auto_rejected: List[str]


@dataclass
class EvalEvent:
    """
    Complete evaluation event for a single experience extraction.

    This captures all stages of the pipeline for debugging and analysis.
    """
    timestamp: datetime
    experience_id: str
    extractor_version: str

    source_context: SourceContextEvent
    segmentation: SegmentationEvent
    atoms: AtomsEvent
    epistemics: List[EpistemicsAtomEvent]
    resolution: List[ResolutionAtomEvent]
    conflicts: ConflictsEvent
    scoring: ScoringEvent
    stream: StreamEvent
    tentative_links: TentativeLinksEvent

    error: Optional[str] = None
    duration_ms: Optional[float] = None


class EvalEventLogger:
    """
    JSONL logger for pipeline telemetry.

    Creates daily log files with one JSON object per line.
    """

    def __init__(self, config: Optional[BeliefSystemConfig] = None):
        """
        Initialize the eval event logger.

        Args:
            config: Configuration object. If None, loads from default.
        """
        if config is None:
            config = get_belief_config()

        self.enabled = config.logging.eval_events.enabled
        self.path = Path(config.logging.eval_events.path)
        self.format = config.logging.eval_events.format

        if self.enabled:
            self.path.mkdir(parents=True, exist_ok=True)

    def _get_log_file(self) -> Path:
        """Get the log file path for today."""
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self.path / f"eval_events_{date_str}.jsonl"

    def _serialize_event(self, event: EvalEvent) -> str:
        """Serialize an event to JSON string."""
        data = asdict(event)

        # Convert datetime to ISO format
        data['timestamp'] = event.timestamp.isoformat()

        return json.dumps(data, default=str)

    def log_event(self, event: EvalEvent) -> None:
        """
        Append an evaluation event to the daily log file.

        Args:
            event: Complete evaluation event to log
        """
        if not self.enabled:
            return

        try:
            log_file = self._get_log_file()
            line = self._serialize_event(event)

            with open(log_file, 'a') as f:
                f.write(line + '\n')

        except Exception as e:
            logger.warning(f"Failed to log eval event: {e}")

    def log_error(
        self,
        experience_id: str,
        stage: str,
        error: Exception,
        extractor_version: str = "unknown"
    ) -> None:
        """
        Log a terminal error event.

        Args:
            experience_id: ID of the experience being processed
            stage: Pipeline stage where error occurred
            error: The exception that was raised
            extractor_version: Extractor version string
        """
        if not self.enabled:
            return

        # Create minimal event with error
        event = EvalEvent(
            timestamp=datetime.now(timezone.utc),
            experience_id=experience_id,
            extractor_version=extractor_version,
            source_context=SourceContextEvent(
                mode="unknown",
                source_weight=0.0,
                context_id="",
            ),
            segmentation=SegmentationEvent(candidate_count=0),
            atoms=AtomsEvent(raw_count=0, valid_count=0, deduped_count=0),
            epistemics=[],
            resolution=[],
            conflicts=ConflictsEvent(hard_count=0, tension_count=0, edges_created=[]),
            scoring=ScoringEvent(nodes_updated=[]),
            stream=StreamEvent(assignments=[]),
            tentative_links=TentativeLinksEvent(
                created=[], updated=[], auto_accepted=[], auto_rejected=[]
            ),
            error=f"{stage}: {type(error).__name__}: {str(error)}",
        )

        self.log_event(event)

    def read_events(
        self,
        date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Read events from log file.

        Args:
            date: Date to read events for (default: today)
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries
        """
        if date is None:
            date = datetime.now(timezone.utc)

        date_str = date.strftime("%Y%m%d")
        log_file = self.path / f"eval_events_{date_str}.jsonl"

        if not log_file.exists():
            return []

        events = []
        with open(log_file) as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
                    if limit and len(events) >= limit:
                        break

        return events


# Builder helpers for constructing events incrementally

class EvalEventBuilder:
    """Helper class to build EvalEvent incrementally during pipeline execution."""

    def __init__(self, experience_id: str, extractor_version: str):
        self.experience_id = experience_id
        self.extractor_version = extractor_version
        self.start_time = datetime.now(timezone.utc)

        # Initialize with empty/default values
        self.source_context: Optional[SourceContextEvent] = None
        self.segmentation: Optional[SegmentationEvent] = None
        self.atoms: Optional[AtomsEvent] = None
        self.epistemics: List[EpistemicsAtomEvent] = []
        self.resolution: List[ResolutionAtomEvent] = []
        self.conflicts: Optional[ConflictsEvent] = None
        self.scoring: Optional[ScoringEvent] = None
        self.stream: Optional[StreamEvent] = None
        self.tentative_links: Optional[TentativeLinksEvent] = None
        self.error: Optional[str] = None

    def set_source_context(
        self,
        mode: str,
        source_weight: float,
        context_id: str,
        vad_used: bool = False,
        heuristics_used: bool = False,
        penalties_applied: Optional[List[str]] = None
    ) -> 'EvalEventBuilder':
        """Set source context information."""
        self.source_context = SourceContextEvent(
            mode=mode,
            source_weight=source_weight,
            context_id=context_id,
            vad_used=vad_used,
            heuristics_used=heuristics_used,
            penalties_applied=penalties_applied or [],
        )
        return self

    def set_segmentation(self, candidate_count: int) -> 'EvalEventBuilder':
        """Set segmentation results."""
        self.segmentation = SegmentationEvent(candidate_count=candidate_count)
        return self

    def set_atoms(
        self,
        raw_count: int,
        valid_count: int,
        deduped_count: int,
        invalid: Optional[List[InvalidAtom]] = None
    ) -> 'EvalEventBuilder':
        """Set atomization results."""
        self.atoms = AtomsEvent(
            raw_count=raw_count,
            valid_count=valid_count,
            deduped_count=deduped_count,
            invalid=invalid or [],
        )
        return self

    def add_epistemic(
        self,
        atom_text: str,
        frame: Dict[str, Any],
        confidence: float,
        signals: List[Dict[str, Any]],
        used_llm_fallback: bool
    ) -> 'EvalEventBuilder':
        """Add epistemics extraction for an atom."""
        self.epistemics.append(EpistemicsAtomEvent(
            atom_text=atom_text,
            frame=frame,
            confidence=confidence,
            signals=signals,
            used_llm_fallback=used_llm_fallback,
        ))
        return self

    def add_resolution(
        self,
        atom_text: str,
        outcome: str,
        match_confidence: float,
        candidate_ids: List[str],
        belief_id_final: str,
        created_tentative_link: bool
    ) -> 'EvalEventBuilder':
        """Add resolution results for an atom."""
        self.resolution.append(ResolutionAtomEvent(
            atom_text=atom_text,
            outcome=outcome,
            match_confidence=match_confidence,
            candidate_ids=candidate_ids,
            belief_id_final=belief_id_final,
            created_tentative_link=created_tentative_link,
        ))
        return self

    def set_conflicts(
        self,
        hard_count: int,
        tension_count: int,
        edges_created: List[str]
    ) -> 'EvalEventBuilder':
        """Set conflict detection results."""
        self.conflicts = ConflictsEvent(
            hard_count=hard_count,
            tension_count=tension_count,
            edges_created=edges_created,
        )
        return self

    def set_scoring(self, nodes_updated: List[NodeUpdate]) -> 'EvalEventBuilder':
        """Set scoring updates."""
        self.scoring = ScoringEvent(nodes_updated=nodes_updated)
        return self

    def set_stream(
        self,
        assignments: List[StreamAssignmentEvent]
    ) -> 'EvalEventBuilder':
        """Set stream assignments."""
        self.stream = StreamEvent(assignments=assignments)
        return self

    def set_tentative_links(
        self,
        created: List[str],
        updated: List[str],
        auto_accepted: List[str],
        auto_rejected: List[str]
    ) -> 'EvalEventBuilder':
        """Set tentative link operations."""
        self.tentative_links = TentativeLinksEvent(
            created=created,
            updated=updated,
            auto_accepted=auto_accepted,
            auto_rejected=auto_rejected,
        )
        return self

    def set_error(self, error: str) -> 'EvalEventBuilder':
        """Set error message."""
        self.error = error
        return self

    def build(self) -> EvalEvent:
        """Build the final EvalEvent."""
        end_time = datetime.now(timezone.utc)
        duration_ms = (end_time - self.start_time).total_seconds() * 1000

        return EvalEvent(
            timestamp=self.start_time,
            experience_id=self.experience_id,
            extractor_version=self.extractor_version,
            source_context=self.source_context or SourceContextEvent(
                mode="unknown", source_weight=0.0, context_id=""
            ),
            segmentation=self.segmentation or SegmentationEvent(candidate_count=0),
            atoms=self.atoms or AtomsEvent(raw_count=0, valid_count=0, deduped_count=0),
            epistemics=self.epistemics,
            resolution=self.resolution,
            conflicts=self.conflicts or ConflictsEvent(
                hard_count=0, tension_count=0, edges_created=[]
            ),
            scoring=self.scoring or ScoringEvent(nodes_updated=[]),
            stream=self.stream or StreamEvent(assignments=[]),
            tentative_links=self.tentative_links or TentativeLinksEvent(
                created=[], updated=[], auto_accepted=[], auto_rejected=[]
            ),
            error=self.error,
            duration_ms=duration_ms,
        )
