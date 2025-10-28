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
from typing import List, Optional, Dict, Any

from src.services.llm import LLMService
from src.services.belief_vector_store import BeliefVectorResult
from src.services.retrieval import RetrievalResult

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

    def __init__(self, llm_service: LLMService):
        """Initialize consistency checker.

        Args:
            llm_service: LLM for extracting claims and analyzing dissonance
        """
        self.llm = llm_service

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

        # Step 3: Generate summary
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


def create_belief_consistency_checker(llm_service: LLMService) -> BeliefConsistencyChecker:
    """Factory function to create belief consistency checker.

    Args:
        llm_service: LLM service for analysis

    Returns:
        BeliefConsistencyChecker instance
    """
    return BeliefConsistencyChecker(llm_service)
