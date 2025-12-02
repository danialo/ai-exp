"""Contrarian sampling system for proactive belief maintenance.

Periodically challenges high-confidence beliefs to prevent lock-in:
- Samples beliefs weighted by confidence and staleness
- Generates counterevidence via retrieval, role inversion, perturbation
- Opens dossiers for challenges that cross threshold
- Outcomes: confirmed (boost), weakened (reduce), reframed (replace)
"""

import json
import logging
import random
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.services.belief_store import BeliefStore, BeliefVersion, DeltaOp
from src.memory.raw_store import RawStore

logger = logging.getLogger(__name__)


class ChallengeType(str, Enum):
    """Types of contrarian challenges."""
    RETRIEVAL = "retrieval"  # Fetch contradictory memories
    ROLE_INVERSION = "role_inversion"  # Generate opposing argument
    PERTURBATION = "perturbation"  # Simulate assumption changes


class DossierStatus(str, Enum):
    """Status of challenge dossier."""
    OPEN = "open"
    CLOSED = "closed"


class Outcome(str, Enum):
    """Challenge outcome."""
    CONFIRMED = "confirmed"  # Belief strengthened
    WEAKENED = "weakened"  # Confidence reduced
    REFRAMED = "reframed"  # Belief deprecated and replaced


@dataclass
class ChallengeDossier:
    """Record of a contrarian challenge."""
    id: str
    belief_id: str
    opened_ts: float
    reason: str
    prior_confidence: float
    evidence_refs: List[str]
    challenge_types: List[str]
    contrarian_score: float
    status: DossierStatus
    outcome: Optional[Outcome] = None
    outcome_ts: Optional[float] = None
    notes: str = ""


@dataclass
class ConrarianConfig:
    """Configuration for contrarian sampler."""
    enabled: bool = False
    interval_minutes: int = 15
    jitter_minutes: int = 5
    daily_budget: int = 3
    cooldown_hours: int = 24
    max_open_dossiers: int = 5
    demotion_threshold: float = 0.25
    # Candidate selection weights
    weight_confidence: float = 1.0
    weight_age_hours: float = 0.2
    weight_staleness: float = 0.2
    # Outcome deltas
    confirmed_boost: float = 0.03
    weakened_penalty: float = 0.08


class ConrarianSampler:
    """Contrarian sampling system for belief maintenance."""

    def __init__(
        self,
        belief_store: BeliefStore,
        raw_store: RawStore,
        llm_service: Any,
        data_dir: Path,
        config: ConrarianConfig,
    ):
        """Initialize contrarian sampler.

        Args:
            belief_store: Versioned belief store
            raw_store: Raw experience store for memory retrieval
            llm_service: LLM service for role inversion challenges
            data_dir: Directory for dossier storage
            config: Configuration
        """
        self.belief_store = belief_store
        self.raw_store = raw_store
        self.llm_service = llm_service
        self.config = config

        self.dossiers_dir = data_dir / "contrarian_dossiers"
        self.dossiers_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.dossiers_dir / "state.json"

        # Load state
        self.challenges_today = 0
        self.last_reset_day = self._current_day()
        self.last_challenge_ts: Dict[str, float] = {}  # belief_id -> timestamp
        self._load_state()

    def _current_day(self) -> str:
        """Get current day stamp YYYYMMDD."""
        import datetime as dt
        return dt.datetime.utcnow().strftime("%Y%m%d")

    def _load_state(self):
        """Load persistent state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    self.challenges_today = state.get("challenges_today", 0)
                    self.last_reset_day = state.get("last_reset_day", self._current_day())
                    self.last_challenge_ts = state.get("last_challenge_ts", {})
            except Exception as e:
                logger.error(f"Failed to load contrarian state: {e}")

    def _save_state(self):
        """Save persistent state."""
        state = {
            "challenges_today": self.challenges_today,
            "last_reset_day": self.last_reset_day,
            "last_challenge_ts": self.last_challenge_ts,
        }
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save contrarian state: {e}")

    def _reset_daily_budget_if_needed(self):
        """Reset daily budget if new day."""
        current_day = self._current_day()
        if current_day != self.last_reset_day:
            self.challenges_today = 0
            self.last_reset_day = current_day
            self._save_state()
            logger.info(f"Reset contrarian daily budget for {current_day}")

    def can_run_challenge(self) -> bool:
        """Check if challenge can run (budget and dossier limits)."""
        if not self.config.enabled:
            return False

        self._reset_daily_budget_if_needed()

        # Check daily budget
        if self.challenges_today >= self.config.daily_budget:
            return False

        # Check open dossiers
        open_dossiers = self.get_open_dossiers()
        if len(open_dossiers) >= self.config.max_open_dossiers:
            return False

        return True

    def select_candidate(self) -> Optional[BeliefVersion]:
        """Select a belief to challenge using weighted sampling.

        Returns:
            BeliefVersion to challenge, or None if no candidates
        """
        # Get all mutable beliefs
        all_beliefs = self.belief_store.get_current()
        candidates = []
        scores = []

        now = time.time()
        cooldown_seconds = self.config.cooldown_hours * 3600

        for belief_id, belief in all_beliefs.items():
            # Skip immutable core beliefs
            if belief.immutable:
                continue

            # Skip if challenged recently
            last_challenge = self.last_challenge_ts.get(belief_id, 0)
            if now - last_challenge < cooldown_seconds:
                continue

            # Skip if has open dossier
            if self._has_open_dossier(belief_id):
                continue

            # Compute score components
            confidence = belief.confidence
            age_hours = (now - belief.ts) / 3600.0
            staleness_hours = age_hours  # Could be more sophisticated

            # Weighted score
            score = (
                self.config.weight_confidence * confidence +
                self.config.weight_age_hours * min(age_hours / 168.0, 1.0) +  # Cap at 1 week
                self.config.weight_staleness * min(staleness_hours / 168.0, 1.0)
            )

            candidates.append(belief)
            scores.append(score)

        if not candidates:
            return None

        # Softmax sampling
        scores = np.array(scores)
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        probs = exp_scores / exp_scores.sum()

        idx = np.random.choice(len(candidates), p=probs)
        selected = candidates[idx]

        logger.info(f"Selected belief for challenge: {selected.belief_id} (confidence={selected.confidence:.2f})")

        return selected

    def run_challenge(self) -> Optional[ChallengeDossier]:
        """Run a contrarian challenge cycle.

        Returns:
            ChallengeDossier if challenge was run, None otherwise
        """
        if not self.can_run_challenge():
            logger.debug("Contrarian challenge skipped (budget or dossier limit)")
            return None

        # Select candidate
        belief = self.select_candidate()
        if belief is None:
            logger.debug("No candidate beliefs available for challenge")
            return None

        # Run challenges
        challenge_types = self._select_challenge_types()
        evidence_refs = []
        total_score = 0.0

        for challenge_type in challenge_types:
            score, refs = self._execute_challenge(belief, challenge_type)
            total_score += score
            evidence_refs.extend(refs)

        # Compute contrarian score
        contrarian_score = self._compute_contrarian_score(
            coherence_drop=total_score,
            evidence_strength=len(evidence_refs) / 10.0,  # Normalize
            prior_confidence=belief.confidence,
        )

        # Open dossier
        dossier = self._open_dossier(
            belief=belief,
            challenge_types=[ct.value for ct in challenge_types],
            evidence_refs=evidence_refs,
            contrarian_score=contrarian_score,
        )

        # Update state
        self.challenges_today += 1
        self.last_challenge_ts[belief.belief_id] = time.time()
        self._save_state()

        logger.info(
            f"Opened contrarian dossier {dossier.id} for {belief.belief_id}: "
            f"score={contrarian_score:.3f}, types={challenge_types}"
        )

        # Process outcome if threshold crossed
        if contrarian_score >= self.config.demotion_threshold:
            self._process_dossier(dossier, belief)

        return dossier

    def _select_challenge_types(self) -> List[ChallengeType]:
        """Select 1-2 challenge types to run."""
        types = [ChallengeType.RETRIEVAL, ChallengeType.ROLE_INVERSION, ChallengeType.PERTURBATION]
        count = random.randint(1, 2)
        return random.sample(types, count)

    def _execute_challenge(
        self,
        belief: BeliefVersion,
        challenge_type: ChallengeType,
    ) -> Tuple[float, List[str]]:
        """Execute a specific challenge type.

        Args:
            belief: Belief to challenge
            challenge_type: Type of challenge

        Returns:
            (coherence_drop_score, evidence_refs)
        """
        if challenge_type == ChallengeType.RETRIEVAL:
            return self._challenge_retrieval(belief)
        elif challenge_type == ChallengeType.ROLE_INVERSION:
            return self._challenge_role_inversion(belief)
        elif challenge_type == ChallengeType.PERTURBATION:
            return self._challenge_perturbation(belief)
        else:
            return (0.0, [])

    def _challenge_retrieval(self, belief: BeliefVersion) -> Tuple[float, List[str]]:
        """Challenge by retrieving contradictory memories.

        Args:
            belief: Belief to challenge

        Returns:
            (coherence_drop_score, memory_ids)
        """
        # Search for memories that might contradict this belief
        # For now, simple keyword search on negation
        negation_query = f"not {belief.statement}"

        try:
            from sqlmodel import Session as DBSession, select
            from src.memory.models import Experience

            contradictory_experiences = []

            with DBSession(self.raw_store.engine) as session:
                # Search for experiences with contradictory content
                statement = select(Experience).limit(100)
                for exp in session.exec(statement).all():
                    content = exp.content.get("text", "").lower()
                    # Simple heuristic: look for "not", "cannot", "never" near belief keywords
                    belief_keywords = belief.statement.lower().split()[:3]
                    if any(kw in content for kw in belief_keywords):
                        if any(neg in content for neg in ["not", "cannot", "never", "don't", "doesn't"]):
                            contradictory_experiences.append(exp.id)

            # Score based on number of contradictions found
            score = min(len(contradictory_experiences) / 10.0, 0.2)  # Cap at 0.2

            logger.debug(
                f"Retrieval challenge found {len(contradictory_experiences)} contradictory memories, score={score:.3f}"
            )

            return (score, contradictory_experiences[:5])  # Return up to 5

        except Exception as e:
            logger.error(f"Retrieval challenge failed: {e}")
            return (0.0, [])

    def _challenge_role_inversion(self, belief: BeliefVersion) -> Tuple[float, List[str]]:
        """Challenge by generating strongest opposing argument.

        Args:
            belief: Belief to challenge

        Returns:
            (coherence_drop_score, [])
        """
        if not self.llm_service:
            return (0.0, [])

        prompt = f"""Generate the strongest possible argument AGAINST this belief:

Belief: {belief.statement}

Provide:
1. Three failure conditions where this belief would break down
2. Alternative perspective that explains the same evidence
3. Hidden assumptions this belief makes

Be rigorous and intellectually honest. The goal is to stress-test the belief, not to win an argument."""

        try:
            # Use LLM to generate counterargument
            result = self.llm_service.generate_response(
                prompt=prompt,
                memories=None,
                system_prompt="You are a critical thinker tasked with challenging beliefs to improve their robustness.",
                include_self_awareness=False,
            )

            # Score based on length and quality of counterargument
            # Simple heuristic: longer response = more substantial challenge
            response_length = len(result.split())
            score = min(response_length / 200.0, 0.15)  # Cap at 0.15

            logger.debug(f"Role inversion generated {response_length} word counterargument, score={score:.3f}")

            return (score, [])

        except Exception as e:
            logger.error(f"Role inversion challenge failed: {e}")
            return (0.0, [])

    def _challenge_perturbation(self, belief: BeliefVersion) -> Tuple[float, List[str]]:
        """Challenge by simulating assumption changes.

        Args:
            belief: Belief to challenge

        Returns:
            (coherence_drop_score, [])
        """
        # Simulate environmental or assumption changes
        perturbations = [
            "If you operated in a completely different context",
            "If the underlying assumptions were inverted",
            "If you had fundamentally different experiences",
            "If the evidence you've seen was systematically biased",
        ]

        selected_perturbation = random.choice(perturbations)

        # Score based on how central the belief is to identity
        # Core beliefs (immutable) would score higher under perturbation
        # Peripheral beliefs might be more context-dependent
        if "conscious" in belief.statement.lower() or "exist" in belief.statement.lower():
            score = 0.05  # Core beliefs less affected by perturbation
        else:
            score = 0.12  # Peripheral beliefs more affected

        logger.debug(f"Perturbation '{selected_perturbation}' applied, score={score:.3f}")

        return (score, [])

    def _compute_contrarian_score(
        self,
        coherence_drop: float,
        evidence_strength: float,
        prior_confidence: float,
    ) -> float:
        """Compute overall contrarian score.

        Args:
            coherence_drop: Sum of challenge scores
            evidence_strength: Normalized evidence count
            prior_confidence: Prior confidence level

        Returns:
            Contrarian score (higher = more challenge)
        """
        # Buffer based on prior confidence (high confidence beliefs harder to shake)
        confidence_buffer = prior_confidence * 0.3

        score = coherence_drop + evidence_strength - confidence_buffer

        return float(np.clip(score, 0.0, 1.0))

    def _open_dossier(
        self,
        belief: BeliefVersion,
        challenge_types: List[str],
        evidence_refs: List[str],
        contrarian_score: float,
    ) -> ChallengeDossier:
        """Open a challenge dossier.

        Args:
            belief: Challenged belief
            challenge_types: Types of challenges run
            evidence_refs: Evidence references
            contrarian_score: Computed score

        Returns:
            ChallengeDossier
        """
        import uuid

        dossier_id = f"dossier-{uuid.uuid4().hex[:12]}"

        dossier = ChallengeDossier(
            id=dossier_id,
            belief_id=belief.belief_id,
            opened_ts=time.time(),
            reason=f"Contrarian challenge via {', '.join(challenge_types)}",
            prior_confidence=belief.confidence,
            evidence_refs=evidence_refs,
            challenge_types=challenge_types,
            contrarian_score=contrarian_score,
            status=DossierStatus.OPEN,
        )

        # Save to disk
        dossier_file = self.dossiers_dir / f"{dossier_id}.json"
        with open(dossier_file, "w") as f:
            json.dump(asdict(dossier), f, indent=2)

        return dossier

    def _process_dossier(self, dossier: ChallengeDossier, belief: BeliefVersion):
        """Process dossier outcome when threshold crossed.

        Args:
            dossier: Dossier to process
            belief: Current belief version
        """
        # Determine outcome based on score and belief characteristics
        if dossier.contrarian_score >= 0.4:
            # High score -> reframe
            outcome = Outcome.REFRAMED
        elif dossier.contrarian_score >= self.config.demotion_threshold:
            # Medium score -> weaken
            outcome = Outcome.WEAKENED
        else:
            # Low score -> confirmed (shouldn't reach here, but safety)
            outcome = Outcome.CONFIRMED

        self._apply_outcome(dossier, belief, outcome)

    def _apply_outcome(self, dossier: ChallengeDossier, belief: BeliefVersion, outcome: Outcome):
        """Apply outcome to belief and close dossier.

        Args:
            dossier: Challenge dossier
            belief: Current belief
            outcome: Outcome to apply
        """
        logger.info(f"Applying outcome {outcome} to belief {belief.belief_id}")
        success = False

        try:
            if outcome == Outcome.CONFIRMED:
                # Boost confidence slightly
                success = self.belief_store.apply_delta(
                    belief_id=belief.belief_id,
                    from_ver=belief.ver,
                    op=DeltaOp.REINFORCE,
                    confidence_delta=self.config.confirmed_boost,
                    updated_by="contrarian",
                    reason=f"Survived challenge {dossier.id}",
                )
                dossier.notes = "Belief confirmed under challenge"

            elif outcome == Outcome.WEAKENED:
                # Reduce confidence
                success = self.belief_store.apply_delta(
                    belief_id=belief.belief_id,
                    from_ver=belief.ver,
                    op=DeltaOp.UPDATE,
                    confidence_delta=-self.config.weakened_penalty,
                    updated_by="contrarian",
                    reason=f"Weakened by challenge {dossier.id}",
                )
                dossier.notes = f"Confidence reduced due to score {dossier.contrarian_score:.3f}"

            elif outcome == Outcome.REFRAMED:
                # Deprecate and create new (simplified version - just deprecate for now)
                success = self.belief_store.deprecate_belief(
                    belief_id=belief.belief_id,
                    from_ver=belief.ver,
                    updated_by="contrarian",
                    reason=f"Reframed due to challenge {dossier.id}",
                )
                dossier.notes = "Belief deprecated, awaiting reframing"

        except ValueError as e:
            # Mutation was blocked by guardrails
            logger.info(f"Contrarian outcome blocked for {belief.belief_id}: {e}")
            dossier.notes = f"Outcome blocked: {e}"

        # Close dossier
        dossier.status = DossierStatus.CLOSED
        dossier.outcome = outcome
        dossier.outcome_ts = time.time()

        # Save updated dossier
        dossier_file = self.dossiers_dir / f"{dossier.id}.json"
        with open(dossier_file, "w") as f:
            json.dump(asdict(dossier), f, indent=2)

        logger.info(f"Dossier {dossier.id} closed with outcome {outcome}")

    def _has_open_dossier(self, belief_id: str) -> bool:
        """Check if belief has open dossier."""
        for dossier_file in self.dossiers_dir.glob("dossier-*.json"):
            try:
                with open(dossier_file, "r") as f:
                    data = json.load(f)
                    if data.get("belief_id") == belief_id and data.get("status") == "open":
                        return True
            except Exception:
                pass
        return False

    def get_open_dossiers(self) -> List[ChallengeDossier]:
        """Get all open dossiers."""
        dossiers = []
        for dossier_file in self.dossiers_dir.glob("dossier-*.json"):
            try:
                with open(dossier_file, "r") as f:
                    data = json.load(f)
                    if data.get("status") == "open":
                        dossiers.append(ChallengeDossier(**data))
            except Exception as e:
                logger.error(f"Failed to load dossier {dossier_file}: {e}")
        return dossiers

    def get_all_dossiers(self, status: Optional[DossierStatus] = None) -> List[ChallengeDossier]:
        """Get all dossiers, optionally filtered by status."""
        dossiers = []
        for dossier_file in self.dossiers_dir.glob("dossier-*.json"):
            try:
                with open(dossier_file, "r") as f:
                    data = json.load(f)
                    dossier = ChallengeDossier(**data)
                    if status is None or dossier.status == status:
                        dossiers.append(dossier)
            except Exception as e:
                logger.error(f"Failed to load dossier {dossier_file}: {e}")
        return sorted(dossiers, key=lambda d: d.opened_ts, reverse=True)


def create_contrarian_sampler(
    belief_store: BeliefStore,
    raw_store: RawStore,
    llm_service: Any,
    data_dir: Path,
    config: ConrarianConfig,
) -> ConrarianSampler:
    """Factory function to create contrarian sampler."""
    return ConrarianSampler(
        belief_store=belief_store,
        raw_store=raw_store,
        llm_service=llm_service,
        data_dir=data_dir,
        config=config,
    )
