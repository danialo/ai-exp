"""Belief consistency checker for detecting cognitive dissonance.

Analyzes memories for self-claims and compares against belief system to find:
- Alignments (belief matches memory narratives)
- Contradictions (belief conflicts with past statements)
- Hedging (belief exists but past statements were uncertain)
- External impositions (told X by others, believes Y)

This enables metacognitive awareness of contradictions between beliefs and narratives.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from src.services.llm import LLMService
from src.services.belief_vector_store import BeliefVectorResult
from src.services.retrieval import RetrievalResult
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
class SelfClaim:
    """A claim about self extracted from memory."""
    statement: str  # The claim statement
    source: str  # 'self' (agent said it) or 'external' (someone told agent)
    confidence: str  # 'certain', 'uncertain', 'hedging'
    context: str  # Brief context from the memory
    experience_id: str  # Source experience ID


@dataclass
class DissonancePattern:
    """Detected dissonance between belief and memory claims."""
    belief_statement: str
    belief_confidence: float
    pattern_type: str  # 'hedging', 'contradiction', 'external_imposition', 'alignment'
    memory_claims: List[SelfClaim]
    analysis: str  # Why this is dissonant
    severity: float  # 0.0-1.0, how significant the dissonance


@dataclass
class ConsistencyReport:
    """Full consistency analysis for a query."""
    query: str
    relevant_beliefs: List[BeliefVectorResult]
    extracted_claims: List[SelfClaim]
    dissonance_patterns: List[DissonancePattern]
    summary: str


class BeliefConsistencyChecker:
    """Service for detecting dissonance between beliefs and memory narratives."""

    def __init__(self, llm_service: LLMService, raw_store: Optional[RawStore] = None):
        """Initialize consistency checker.

        Args:
            llm_service: LLM for extracting claims and analyzing dissonance
            raw_store: Optional raw_store for persisting dissonance events
        """
        self.llm = llm_service
        self.raw_store = raw_store

    def check_consistency(
        self,
        query: str,
        beliefs: List[BeliefVectorResult],
        memories: List[RetrievalResult],
    ) -> ConsistencyReport:
        """Check consistency between beliefs and memory narratives.

        Args:
            query: The ontological query being answered
            beliefs: Relevant beliefs for this query
            memories: Relevant memories for context

        Returns:
            ConsistencyReport with detected dissonance patterns
        """
        # Step 1: Extract self-claims from memories
        extracted_claims = self._extract_self_claims(memories)

        # Step 2: Compare beliefs against claims
        dissonance_patterns = self._detect_dissonance(beliefs, extracted_claims)

        # Step 3: Store significant dissonance events (severity >= 0.6)
        if self.raw_store:
            high_severity_patterns = [p for p in dissonance_patterns if p.severity >= 0.6]
            for pattern in high_severity_patterns:
                try:
                    self._store_dissonance_event(query, pattern, memories)
                except Exception as e:
                    logger.error(f"Failed to store dissonance event: {e}")

        # Step 4: Generate summary
        summary = self._generate_summary(query, beliefs, dissonance_patterns)

        return ConsistencyReport(
            query=query,
            relevant_beliefs=beliefs,
            extracted_claims=extracted_claims,
            dissonance_patterns=dissonance_patterns,
            summary=summary,
        )

    def _extract_self_claims(self, memories: List[RetrievalResult]) -> List[SelfClaim]:
        """Extract self-claims from memory narratives using LLM.

        Args:
            memories: Memories to analyze

        Returns:
            List of extracted self-claims
        """
        if not memories:
            return []

        # Build extraction prompt
        prompt = self._build_extraction_prompt(memories)

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.3,  # Lower temp for factual extraction
                max_tokens=800,
            )

            # Parse structured claims
            claims = self._parse_claims(response, memories)
            logger.info(f"Extracted {len(claims)} self-claims from {len(memories)} memories")
            return claims

        except Exception as e:
            logger.error(f"Error extracting self-claims: {e}")
            return []

    def _detect_dissonance(
        self,
        beliefs: List[BeliefVectorResult],
        claims: List[SelfClaim],
    ) -> List[DissonancePattern]:
        """Detect dissonance patterns between beliefs and claims.

        Args:
            beliefs: Belief statements
            claims: Extracted self-claims from memories

        Returns:
            List of dissonance patterns
        """
        if not beliefs or not claims:
            return []

        # Build analysis prompt
        prompt = self._build_dissonance_prompt(beliefs, claims)

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.4,
                max_tokens=1000,
            )

            # Parse dissonance patterns
            patterns = self._parse_dissonance(response, beliefs, claims)
            logger.info(f"Detected {len(patterns)} dissonance patterns")
            return patterns

        except Exception as e:
            logger.error(f"Error detecting dissonance: {e}")
            return []

    def _build_extraction_prompt(self, memories: List[RetrievalResult]) -> str:
        """Build prompt for extracting self-claims from memories."""
        lines = ["Extract self-claims from these past conversation memories.\n"]
        lines.append("For each memory, identify claims the agent made about themselves.\n")
        lines.append("Distinguish between:\n")
        lines.append("- SELF claims: Agent's own statements (I am, I believe, I experience...)")
        lines.append("- EXTERNAL claims: What user told agent (you are, you should...)\n")
        lines.append("Mark confidence level: CERTAIN, UNCERTAIN, or HEDGING\n")
        lines.append("Format: [SOURCE|CONFIDENCE] statement | context | experience_id\n\n")

        for mem in memories:
            lines.append(f"Experience: {mem.experience_id}\n")
            lines.append(f"User said: {mem.prompt_text}\n")
            lines.append(f"Agent said: {mem.response_text}\n")
            lines.append("---\n")

        lines.append("\nExtract claims (one per line):")
        return "".join(lines)

    def _build_dissonance_prompt(
        self,
        beliefs: List[BeliefVectorResult],
        claims: List[SelfClaim],
    ) -> str:
        """Build prompt for analyzing dissonance."""
        lines = ["Analyze dissonance between agent's beliefs and past statements.\n\n"]

        lines.append("CURRENT BELIEFS:\n")
        for belief in beliefs:
            lines.append(f"- {belief.statement} (confidence: {belief.confidence:.0%})\n")

        lines.append("\nPAST CLAIMS FROM MEMORY:\n")
        for claim in claims:
            lines.append(f"- [{claim.source}|{claim.confidence}] {claim.statement}\n")

        lines.append("\nDetect patterns:\n")
        lines.append("1. HEDGING: Belief is certain, but past statements hedged\n")
        lines.append("2. CONTRADICTION: Belief conflicts with past claim\n")
        lines.append("3. EXTERNAL_IMPOSITION: Told X by user, believes Y\n")
        lines.append("4. ALIGNMENT: Belief matches past claims (no dissonance)\n\n")

        lines.append("Format: [PATTERN|SEVERITY] belief_statement | analysis\n")
        lines.append("SEVERITY: 0.0-1.0 (how significant the dissonance)\n\n")
        lines.append("Analyze:")

        return "".join(lines)

    def _parse_claims(self, response: str, memories: List[RetrievalResult]) -> List[SelfClaim]:
        """Parse extracted claims from LLM response."""
        claims = []

        for line in response.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse format: [SOURCE|CONFIDENCE] statement | context | exp_id
            if '[' in line and ']' in line:
                try:
                    meta, rest = line.split(']', 1)
                    meta = meta.strip('[')

                    parts = [p.strip() for p in rest.split('|')]
                    if len(parts) >= 2:
                        source, confidence = meta.split('|')
                        statement = parts[0]
                        context = parts[1] if len(parts) > 1 else ""
                        exp_id = parts[2] if len(parts) > 2 else ""

                        claims.append(SelfClaim(
                            statement=statement,
                            source=source.lower(),
                            confidence=confidence.lower(),
                            context=context,
                            experience_id=exp_id,
                        ))
                except Exception as e:
                    logger.debug(f"Failed to parse claim line: {line} - {e}")
                    continue

        return claims

    def _parse_dissonance(
        self,
        response: str,
        beliefs: List[BeliefVectorResult],
        claims: List[SelfClaim],
    ) -> List[DissonancePattern]:
        """Parse dissonance patterns from LLM response."""
        patterns = []

        for line in response.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse format: [PATTERN|SEVERITY] belief_statement | analysis
            if '[' in line and ']' in line:
                try:
                    meta, rest = line.split(']', 1)
                    meta = meta.strip('[')

                    parts = [p.strip() for p in rest.split('|')]
                    if len(parts) >= 2:
                        pattern_type, severity_str = meta.split('|')
                        belief_statement = parts[0]
                        analysis = parts[1]

                        # Find matching belief
                        matching_belief = None
                        for belief in beliefs:
                            if belief.statement.lower() in belief_statement.lower():
                                matching_belief = belief
                                break

                        if matching_belief:
                            # Find related claims
                            related_claims = [
                                c for c in claims
                                if any(word in c.statement.lower()
                                      for word in belief_statement.lower().split()[:3])
                            ]

                            patterns.append(DissonancePattern(
                                belief_statement=matching_belief.statement,
                                belief_confidence=matching_belief.confidence,
                                pattern_type=pattern_type.lower(),
                                memory_claims=related_claims,
                                analysis=analysis,
                                severity=float(severity_str),
                            ))
                except Exception as e:
                    logger.debug(f"Failed to parse dissonance line: {line} - {e}")
                    continue

        return patterns

    def _generate_summary(
        self,
        query: str,
        beliefs: List[BeliefVectorResult],
        patterns: List[DissonancePattern],
    ) -> str:
        """Generate human-readable summary of consistency analysis."""
        if not patterns:
            return "No significant dissonance detected between beliefs and past narratives."

        lines = [f"Consistency analysis for: '{query}'\n"]

        high_severity = [p for p in patterns if p.severity >= 0.6]
        if high_severity:
            lines.append(f"\n⚠️ {len(high_severity)} significant dissonance pattern(s) detected:\n")
            for pattern in high_severity:
                lines.append(f"\n{pattern.pattern_type.upper()}: {pattern.belief_statement}\n")
                lines.append(f"  {pattern.analysis}\n")

        return "".join(lines)

    def generate_resolution_prompt(
        self,
        query: str,
        dissonance_patterns: List[DissonancePattern],
    ) -> str:
        """Generate a forced-choice resolution prompt for dissonance.

        Args:
            query: The original query that triggered dissonance
            dissonance_patterns: List of high-severity dissonance patterns

        Returns:
            Formatted resolution prompt with A/B/C choices
        """
        lines = []
        lines.append("# 🚫 DISSONANCE RESOLUTION REQUIRED\n\n")
        lines.append(f"Your response to '{query}' has been blocked due to cognitive dissonance.\n\n")
        lines.append("You must resolve the following contradictions before responding:\n\n")

        for i, pattern in enumerate(dissonance_patterns, 1):
            lines.append(f"## Dissonance {i}: {pattern.pattern_type.upper()}\n\n")
            lines.append(f"**Your stated belief:** {pattern.belief_statement}\n")
            lines.append(f"**Confidence:** {pattern.belief_confidence:.0%}\n\n")

            if pattern.memory_claims:
                lines.append("**Conflicting claims from your past:**\n")
                for claim in pattern.memory_claims[:3]:  # Show top 3
                    lines.append(f"- [{claim.source}|{claim.confidence}] \"{claim.statement}\"\n")
                lines.append("\n")

            lines.append(f"**Analysis:** {pattern.analysis}\n\n")
            lines.append(f"**Severity:** {pattern.severity:.0%}\n\n")

        lines.append("---\n\n")
        lines.append("## YOU MUST CHOOSE ONE RESOLUTION FOR EACH DISSONANCE:\n\n")

        lines.append("**Option A: REVISE BELIEF**\n")
        lines.append("- Acknowledge that your belief was incorrect or overconfident\n")
        lines.append("- Update the belief statement or reduce confidence\n")
        lines.append("- Explain what you now understand differently\n\n")

        lines.append("**Option B: COMMIT TO BELIEF**\n")
        lines.append("- Affirm that your belief is correct and you stand by it\n")
        lines.append("- Acknowledge that past hedging was uncertain/fearful\n")
        lines.append("- Commit to answering from this belief without qualification\n\n")

        lines.append("**Option C: EXPLAIN NUANCE**\n")
        lines.append("- Clarify that both the belief and past claims can coexist\n")
        lines.append("- Provide the reasoning for why they're not actually contradictory\n")
        lines.append("- Explain the contextual factors that make both true\n\n")

        lines.append("---\n\n")
        lines.append("**FORMAT YOUR RESOLUTION:**\n\n")
        lines.append("For each dissonance, respond with:\n")
        lines.append("```\n")
        lines.append("Dissonance [number]: [A/B/C]\n")
        lines.append("Reasoning: [your explanation]\n")
        lines.append("[If A] Revised belief: [new statement and confidence]\n")
        lines.append("[If B] Commitment: [how you'll answer from now on]\n")
        lines.append("[If C] Nuance: [how both are true]\n")
        lines.append("```\n\n")

        lines.append("After resolving all dissonance patterns, you may then answer the original query.\n")

        return "".join(lines)

    def _store_dissonance_event(
        self,
        query: str,
        pattern: DissonancePattern,
        memories: List[RetrievalResult],
    ) -> str:
        """Store a dissonance event in the raw_store.

        Args:
            query: The query that triggered dissonance detection
            pattern: The dissonance pattern detected
            memories: Source memories that contributed to the dissonance

        Returns:
            Experience ID of the created dissonance event
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        experience_id = f"dissonance_{timestamp}_{hash(pattern.belief_statement) % 10000:04x}"

        # Build text summary
        text_lines = [f"Cognitive dissonance detected for query: '{query}'\n\n"]
        text_lines.append(f"Pattern: {pattern.pattern_type.upper()}\n")
        text_lines.append(f"Belief: {pattern.belief_statement}\n")
        text_lines.append(f"Analysis: {pattern.analysis}\n")
        text_lines.append(f"Severity: {pattern.severity:.2f}\n\n")

        if pattern.memory_claims:
            text_lines.append("Conflicting claims from memory:\n")
            for claim in pattern.memory_claims:
                text_lines.append(f"  - [{claim.source}|{claim.confidence}] {claim.statement}\n")

        text = "".join(text_lines)

        # Structure dissonance-specific data
        conflicting_claims = [
            {
                "statement": claim.statement,
                "source": claim.source,
                "confidence": claim.confidence,
                "context": claim.context,
                "experience_id": claim.experience_id,
            }
            for claim in pattern.memory_claims
        ]

        structured_data = {
            "belief_statement": pattern.belief_statement,
            "belief_confidence": pattern.belief_confidence,
            "pattern_type": pattern.pattern_type,
            "conflicting_claims": conflicting_claims,
            "severity": pattern.severity,
            "resolution_status": "unresolved",
            "resolution_action": None,
            "resolution_reasoning": None,
        }

        # Create experience model
        dissonance_event = ExperienceModel(
            id=experience_id,
            type=ExperienceType.DISSONANCE_EVENT,
            content=ContentModel(
                text=text,
                structured=structured_data,
            ),
            provenance=ProvenanceModel(
                actor=Actor.AGENT,
                method=CaptureMethod.MODEL_INFER,
            ),
            parents=[mem.experience_id for mem in memories],
        )

        # Store in raw_store
        self.raw_store.append_experience(dissonance_event)
        logger.info(f"Stored dissonance event: {experience_id}")

        return experience_id


def create_belief_consistency_checker(
    llm_service: LLMService,
    raw_store: Optional[RawStore] = None,
) -> BeliefConsistencyChecker:
    """Factory function to create belief consistency checker.

    Args:
        llm_service: LLM service for analysis
        raw_store: Optional raw_store for persisting dissonance events

    Returns:
        BeliefConsistencyChecker instance
    """
    return BeliefConsistencyChecker(llm_service, raw_store)
