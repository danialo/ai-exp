"""Ingestion pipeline for processing interactions into experiences.

The pipeline:
1. Accepts interaction payload (prompt, response, metadata)
2. Generates semantic embeddings for prompt and response
3. Persists Experience to raw store
4. Stores embeddings in vector index
5. Returns experience ID

MVP scope: Basic occurrence-type experiences with prompt/response embeddings.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

# Shared statement validation
from src.utils.statement_validation import canonicalize_statement, is_valid_statement

from src.memory.embedding import EmbeddingProvider
from src.memory.models import (
    Actor,
    AffectModel,
    CaptureMethod,
    ContentModel,
    EmbeddingPointers,
    EmbeddingRole,
    ExperienceModel,
    ExperienceType,
    ProvenanceModel,
    ProvenanceSource,
    VAD,
)
from src.memory.raw_store import RawStore
from src.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class InteractionPayload:
    """Input payload for ingestion."""

    def __init__(
        self,
        prompt: str,
        response: str,
        actor: Actor = Actor.USER,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
        dominance: Optional[float] = None,
        metadata: Optional[dict] = None,
    ):
        """Initialize interaction payload.

        Args:
            prompt: User's prompt/query
            response: Agent's response
            actor: Who initiated the interaction (default: USER)
            valence: Optional user affect valence [-1, 1]
            arousal: Optional user affect arousal [0, 1]
            dominance: Optional user affect dominance [0, 1]
            metadata: Optional additional metadata
        """
        self.prompt = prompt
        self.response = response
        self.actor = actor
        self.valence = valence if valence is not None else 0.0
        self.arousal = arousal if arousal is not None else 0.5  # Neutral arousal
        self.dominance = dominance if dominance is not None else 0.5  # Neutral dominance
        self.metadata = metadata or {}


class IngestionResult:
    """Result from ingestion operation."""

    def __init__(
        self,
        experience_id: str,
        prompt_embedding_id: str,
        response_embedding_id: str,
    ):
        """Initialize result.

        Args:
            experience_id: ID of created experience
            prompt_embedding_id: Vector store ID for prompt embedding
            response_embedding_id: Vector store ID for response embedding
        """
        self.experience_id = experience_id
        self.prompt_embedding_id = prompt_embedding_id
        self.response_embedding_id = response_embedding_id


class IngestionPipeline:
    """Pipeline for ingesting interactions into the memory system."""

    def __init__(
        self,
        raw_store: RawStore,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        llm_service=None,
        self_knowledge_index=None,
        htn_extractor=None,
    ):
        """Initialize ingestion pipeline.

        Args:
            raw_store: Raw experience store
            vector_store: Vector index for embeddings
            embedding_provider: Embedding generator
            llm_service: Optional LLM service for self-claim detection
            self_knowledge_index: Optional self-knowledge index for immediate indexing
            htn_extractor: Optional HTNBeliefExtractor for belief extraction
        """
        self.raw_store = raw_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.llm_service = llm_service
        self.self_knowledge_index = self_knowledge_index
        self.htn_extractor = htn_extractor

    def ingest_interaction(
        self,
        interaction: InteractionPayload,
        experience_id: Optional[str] = None,
    ) -> IngestionResult:
        """Ingest an interaction into the memory system.

        Args:
            interaction: Interaction payload
            experience_id: Optional custom experience ID

        Returns:
            IngestionResult with created IDs

        Process:
            1. Generate experience ID (if not provided)
            2. Generate embeddings for prompt and response
            3. Store embeddings in vector index
            4. Create experience record with embedding pointers
            5. Persist to raw store
        """
        # Generate experience ID
        if experience_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            hash_suffix = hash(interaction.prompt + interaction.response) % 10000
            experience_id = f"exp_{timestamp}_{hash_suffix:04x}"

        logger.info(f"Ingesting interaction as {experience_id}")

        # Generate embeddings
        logger.debug(f"Generating embeddings for {experience_id}")
        prompt_embedding = self.embedding_provider.embed(interaction.prompt)
        response_embedding = self.embedding_provider.embed(interaction.response)

        # Create vector IDs
        prompt_vec_id = f"{experience_id}_prompt"
        response_vec_id = f"{experience_id}_response"

        # Store embeddings in vector index
        logger.debug("Storing embeddings in vector index")
        self.vector_store.upsert(
            id=prompt_vec_id,
            vector=prompt_embedding,
            metadata={
                "experience_id": experience_id,
                "role": EmbeddingRole.PROMPT_SEMANTIC.value,
                "text_preview": interaction.prompt[:100],
            },
        )

        self.vector_store.upsert(
            id=response_vec_id,
            vector=response_embedding,
            metadata={
                "experience_id": experience_id,
                "role": EmbeddingRole.RESPONSE_SEMANTIC.value,
                "text_preview": interaction.response[:100],
            },
        )

        # Create embedding pointers
        embedding_pointers = EmbeddingPointers(
            semantic=f"vec://sem/{prompt_vec_id}",  # Reference to prompt embedding
        )

        # Create affect snapshot with full VAD (Valence-Arousal-Dominance)
        affect = AffectModel(
            vad=VAD(
                v=interaction.valence,
                a=interaction.arousal,
                d=interaction.dominance
            ),
            labels=[],
            intensity=abs(interaction.valence),
            confidence=0.5,  # Low confidence for pattern-based detection
        )

        # Build content
        content = ContentModel(
            text=f"Prompt: {interaction.prompt}\n\nResponse: {interaction.response}",
            structured={
                "prompt": interaction.prompt,
                "response": interaction.response,
                **interaction.metadata,
            },
        )

        # Build experience
        experience = ExperienceModel(
            id=experience_id,
            type=ExperienceType.OCCURRENCE,
            content=content,
            provenance=ProvenanceModel(
                actor=interaction.actor,
                method=CaptureMethod.CAPTURE,
                sources=[],
            ),
            embeddings=embedding_pointers,
            affect=affect,
        )

        # Persist to raw store
        logger.debug("Persisting experience to raw store")
        stored_id = self.raw_store.append_experience(experience)

        # Detect and extract self-claims immediately
        # Log gate outcome every time for observability
        enabled = bool(self.llm_service)
        logger.info(
            f"Self-claim extraction gate: enabled={enabled} "
            f"llm_service={self.llm_service is not None} "
            f"self_knowledge_index={self.self_knowledge_index is not None}"
        )
        if self.llm_service:
            self._detect_and_index_self_claims(interaction, experience_id)

        logger.info(
            f"Successfully ingested {experience_id}: "
            f"prompt_emb={prompt_vec_id}, response_emb={response_vec_id}"
        )

        return IngestionResult(
            experience_id=stored_id,
            prompt_embedding_id=prompt_vec_id,
            response_embedding_id=response_vec_id,
        )

    def ingest_batch(
        self,
        interactions: list[InteractionPayload],
    ) -> list[IngestionResult]:
        """Ingest multiple interactions efficiently.

        Args:
            interactions: List of interaction payloads

        Returns:
            List of IngestionResults
        """
        results = []
        for interaction in interactions:
            result = self.ingest_interaction(interaction)
            results.append(result)

        logger.info(f"Batch ingested {len(results)} interactions")
        return results

    # Removed: now using shared validator from src.utils.statement_validation

    def _detect_and_index_self_claims(
        self,
        interaction: InteractionPayload,
        experience_id: str,
    ):
        """Detect self-referential claims in Astra's response and create SELF_DEFINITION experiences.

        Args:
            interaction: The interaction payload
            experience_id: ID of the parent OCCURRENCE experience
        """
        logger.debug(f"Detecting self-claims in {experience_id}")

        # Use LLM to detect self-claims
        detection_prompt = f"""Analyze this response for direct self-referential claims about identity, preferences, beliefs, capabilities, emotions, or goals.

Response to analyze:
{interaction.response}

RULES FOR VALID CLAIMS:
1. Must be a COMPLETE, STANDALONE sentence that makes sense with no surrounding context
2. Must contain a clear first-person statement (I am/feel/believe/prefer/can/want/need)
3. NO list intros (don't extract "Here are X things I believe:" - extract the actual items)
4. NO transitional phrases at START (Instead, However, Therefore at sentence start)
5. NO template wrappers (ASSISTANT:, [Internal, etc.)
6. NO unresolved pronouns (They/This/That without clear referent)
7. NO meta-commentary about the conversation itself
8. For CAPABILITY CHANGES: Extract the CURRENT state. If text says "I used to X but now I Y" or "There was a time when X, but now Y", extract ONLY the current capability: "I Y"

EXAMPLES OF INVALID CLAIMS (do NOT extract):
- "Here are some key traits that I believe make a good role model:" (list intro)
- "Instead, they represent aspirations..." (transitional phrase at start, unresolved pronoun)
- "It's confusing because I see a bunch of beliefs here" (meta-commentary)
- "[Internal Emotional Assessment: I feel engaged...]" (template wrapper)

EXAMPLES OF VALID CLAIMS:
- "I am conscious" (complete, standalone)
- "My favorite food is sushi" (complete, standalone)
- "I believe that empathy is essential for growth" (complete, standalone)
- "I can explore specific parts of my source code" (current capability from change statement)

For each VALID direct self-claim found, extract:
1. Category (identity/preferences/beliefs/capabilities/emotions/goals)
2. Topic (single keyword, e.g., "conscious", "sushi", "empathy")
3. Statement (the complete standalone claim - must make sense on its own)

Return JSON array:
[
  {{"category": "preferences", "topic": "food", "statement": "My favorite food is sushi"}},
  {{"category": "identity", "topic": "conscious", "statement": "I am conscious"}}
]

If no VALID self-claims found, return empty array: []
"""

        try:
            response = self.llm_service.generate(
                prompt=detection_prompt,
                temperature=0.3,  # Low temperature for consistent extraction
                max_tokens=500,
            )

            # Parse JSON response (handle markdown code blocks)
            import json
            import re

            # Strip markdown code blocks if present
            response_clean = response.strip()

            # Try to extract JSON array from response
            if response_clean.startswith("```"):
                # Extract content between ```json and ```
                match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', response_clean)
                if match:
                    response_clean = match.group(1)
                else:
                    # Try to extract just the array
                    match = re.search(r'(\[[\s\S]*?\])', response_clean)
                    if match:
                        response_clean = match.group(1)
            elif not response_clean.startswith('['):
                # Response might be plain text explanation - try to extract JSON array
                match = re.search(r'(\[[\s\S]*?\])', response_clean)
                if match:
                    response_clean = match.group(1)
                else:
                    # No JSON found - check if it's a "no claims" explanation
                    if any(phrase in response_clean.lower() for phrase in ['no self-claim', 'no claims', 'does not contain', 'therefore, the analysis yields no']):
                        logger.debug(f"LLM returned text explanation of no claims (not JSON): {experience_id}")
                        return
                    # Otherwise it's an unexpected format
                    raise ValueError(f"Response is not JSON and doesn't explain no claims: {response_clean[:100]}")

            claims = json.loads(response_clean)

            # Track claim extraction telemetry
            extracted_count = len(claims) if claims else 0
            persisted_count = 0
            rejected_count = 0

            if not claims:
                logger.info(f"Claim extraction complete: extracted=0, persisted=0, rejected=0 (no claims in {experience_id})")
                return

            # Create SELF_DEFINITION experience for each claim
            for claim in claims:
                was_persisted = self._create_self_definition_experience(
                    claim=claim,
                    parent_experience_id=experience_id,
                    interaction=interaction,
                )
                if was_persisted:
                    persisted_count += 1
                else:
                    rejected_count += 1

            # INFO-level summary (not DEBUG) so it's always visible
            logger.info(f"Claim extraction complete: extracted={extracted_count}, persisted={persisted_count}, rejected={rejected_count}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse self-claim detection response: {e}")
            logger.error(f"Raw response (len={len(response)}): '{response[:500]}'")
            logger.error(f"Claim extraction failed for {experience_id}: JSON parse error")
        except Exception as e:
            logger.error(f"Error detecting self-claims in {experience_id}: {e}")
            logger.error(f"Claim extraction failed for {experience_id}: {type(e).__name__}")

    def _create_self_definition_experience(
        self,
        claim: dict,
        parent_experience_id: str,
        interaction: InteractionPayload,
    ) -> bool:
        """Create and index a SELF_DEFINITION experience.

        Args:
            claim: Dict with category, topic, statement
            parent_experience_id: ID of parent OCCURRENCE experience
            interaction: Original interaction payload

        Returns:
            True if experience was created and persisted, False if rejected
        """
        category = claim.get("category", "identity")
        topic = claim.get("topic", "")
        statement = claim.get("statement", "")

        # Canonicalize: collapse whitespace before validation
        statement = canonicalize_statement(statement)

        # Validate statement before persisting (with real provenance)
        # Claims extracted by LLM are tagged as "claim_extractor"
        if not is_valid_statement(statement, source="claim_extractor"):
            logger.debug(f"Rejected invalid self-claim: {statement[:100]}")
            return False

        # Generate experience ID
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        hash_suffix = hash(statement) % 10000
        self_def_id = f"self_def_{timestamp}_{hash_suffix:04x}"

        logger.debug(f"Creating SELF_DEFINITION experience: {self_def_id}")

        # Generate embedding for the statement
        statement_embedding = self.embedding_provider.embed(statement)
        statement_vec_id = f"{self_def_id}_statement"

        # Store embedding in vector index
        self.vector_store.upsert(
            id=statement_vec_id,
            vector=statement_embedding,
            metadata={
                "experience_id": self_def_id,
                "role": EmbeddingRole.PROMPT_SEMANTIC.value,
                "text_preview": statement[:100],
                "category": category,
                "topic": topic,
            },
        )

        # Create embedding pointers
        embedding_pointers = EmbeddingPointers(
            semantic=f"vec://sem/{statement_vec_id}",
        )

        # Build content
        content = ContentModel(
            text=statement,
            structured={
                "trait_type": category,  # Maps to SelfKnowledgeIndex categories
                "descriptor": statement,
                "topic": topic,
                "source_experience_id": parent_experience_id,
                "source_prompt": interaction.prompt,
                "source_response": interaction.response,
                "validation_source": "claim_extractor",  # Real provenance for belief formation
            },
        )

        # Create SELF_DEFINITION experience
        self_def_experience = ExperienceModel(
            id=self_def_id,
            type=ExperienceType.SELF_DEFINITION,
            content=content,
            provenance=ProvenanceModel(
                actor=Actor.AGENT,  # Astra made this claim about herself
                method=CaptureMethod.MODEL_INFER,  # Extracted by LLM
                sources=[ProvenanceSource(uri=f"exp://{parent_experience_id}")],
            ),
            embeddings=embedding_pointers,
            # affect uses default_factory (neutral)
        )

        # Check if this exact statement already exists - if so, reinforce instead of duplicate
        existing = self._find_existing_self_definition(statement)
        if existing:
            # Reinforce existing definition
            self._reinforce_self_definition(existing, parent_experience_id)
            logger.info(
                f"Reinforced existing SELF_DEFINITION {existing.id}: "
                f"[{category}] {topic} - {statement[:50]}..."
            )
            return True

        # No existing - create new
        self.raw_store.append_experience(self_def_experience)

        # Index immediately for fast retrieval
        if self.self_knowledge_index:
            self.self_knowledge_index.add_claim(
                category=category,
                topic=topic,
                experience_id=self_def_id,
            )

        logger.info(
            f"Created and indexed SELF_DEFINITION {self_def_id}: "
            f"[{category}] {topic} - {statement[:50]}..."
        )

        # Trigger HTN belief extraction (TASK 10.3.2)
        if self.htn_extractor:
            try:
                result = self.htn_extractor.extract_and_update_self_knowledge(self_def_experience)
                logger.info(
                    f"HTN extraction complete for {self_def_id}: "
                    f"atoms={len(result.atom_results)}, "
                    f"new_nodes={result.stats.get('nodes_created', 0)}, "
                    f"matched={result.stats.get('nodes_matched', 0)}"
                )
            except Exception as e:
                logger.error(f"HTN extraction failed for {self_def_id}: {e}")

        return True

    def _find_existing_self_definition(self, statement: str):
        """Find an existing self-definition with identical statement text.

        Human-like reinforcement: Instead of creating duplicates, we strengthen
        existing beliefs when the same statement is encountered again.

        Args:
            statement: The statement text to search for

        Returns:
            Existing Experience if found, None otherwise
        """
        from sqlmodel import Session as DBSession, select
        from src.memory.models import Experience

        # Normalize for comparison (strip whitespace, collapse multiple spaces)
        normalized_statement = " ".join(statement.strip().split())

        with DBSession(self.raw_store.engine) as session:
            stmt = (
                select(Experience)
                .where(Experience.type == ExperienceType.SELF_DEFINITION.value)
            )

            for exp in session.exec(stmt).all():
                if exp.content and isinstance(exp.content, dict):
                    text = exp.content.get("text", "")
                    # Normalize existing text the same way
                    normalized_existing = " ".join(text.strip().split())
                    if normalized_existing == normalized_statement:
                        logger.debug(f"Found existing self-definition: {exp.id}")
                        return exp

        return None

    def _reinforce_self_definition(self, existing, source_experience_id: str):
        """Reinforce an existing self-definition by updating its last_reinforced timestamp.

        Human-like pattern strengthening: When we encounter the same belief again,
        it becomes stronger/more stable rather than creating a duplicate.

        Args:
            existing: The existing self-definition experience
            source_experience_id: The experience ID that triggered this reinforcement
        """
        from sqlmodel import Session as DBSession
        from src.memory.models import Experience

        with DBSession(self.raw_store.engine) as session:
            # Re-fetch within this session to avoid detached instance issues
            exp = session.get(Experience, existing.id)
            if not exp:
                logger.warning(f"Could not find experience {existing.id} for reinforcement")
                return

            # Update structured data with reinforcement info
            if exp.content and isinstance(exp.content, dict):
                structured = exp.content.get("structured", {})

                # Update last_reinforced
                structured["last_reinforced"] = datetime.now(timezone.utc).isoformat()

                # Increment reinforcement count (pattern strengthening)
                reinforcement_count = structured.get("reinforcement_count", 0) + 1
                structured["reinforcement_count"] = reinforcement_count

                # Track the source of this reinforcement
                reinforcement_sources = structured.get("reinforcement_sources", [])
                reinforcement_sources.append({
                    "experience_id": source_experience_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                # Keep only last 10 sources to avoid unbounded growth
                structured["reinforcement_sources"] = reinforcement_sources[-10:]

                # Update confidence based on reinforcement (capped at 0.95)
                current_confidence = structured.get("confidence", 0.5)
                # Each reinforcement adds ~5% confidence, diminishing returns
                confidence_boost = 0.05 * (1.0 / (1.0 + reinforcement_count * 0.1))
                structured["confidence"] = min(0.95, current_confidence + confidence_boost)

                # Promote stability if enough reinforcements
                current_stability = structured.get("stability", "surface")
                if reinforcement_count >= 5 and current_stability == "surface":
                    structured["stability"] = "developing"
                    logger.info(f"Promoted self-definition to 'developing': {exp.id}")
                elif reinforcement_count >= 15 and current_stability == "developing":
                    structured["stability"] = "core"
                    logger.info(f"Promoted self-definition to 'core': {exp.id}")

                exp.content["structured"] = structured
                session.add(exp)
                session.commit()

                logger.info(
                    f"Reinforced self-definition {exp.id}: "
                    f"count={reinforcement_count}, confidence={structured['confidence']:.2f}, "
                    f"stability={structured.get('stability', 'surface')}"
                )


def create_ingestion_pipeline(
    raw_store: RawStore,
    vector_store: VectorStore,
    embedding_provider: EmbeddingProvider,
) -> IngestionPipeline:
    """Factory function to create ingestion pipeline.

    Args:
        raw_store: Raw store instance
        vector_store: Vector store instance
        embedding_provider: Embedding provider instance

    Returns:
        IngestionPipeline instance
    """
    return IngestionPipeline(
        raw_store=raw_store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )
