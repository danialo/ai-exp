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
from datetime import datetime, timezone
from typing import Optional

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
    ):
        """Initialize ingestion pipeline.

        Args:
            raw_store: Raw experience store
            vector_store: Vector index for embeddings
            embedding_provider: Embedding generator
            llm_service: Optional LLM service for self-claim detection
            self_knowledge_index: Optional self-knowledge index for immediate indexing
        """
        self.raw_store = raw_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.llm_service = llm_service
        self.self_knowledge_index = self_knowledge_index

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
        if self.llm_service and self.self_knowledge_index:
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

Look for patterns like:
- "I am X" (identity)
- "I feel X" (emotions)
- "I believe X" (beliefs)
- "I prefer X" / "My favorite X is Y" (preferences)
- "I can X" / "I'm able to X" (capabilities)
- "I want X" / "I need X" (goals)

For each direct self-claim found, extract:
1. Category (identity/preferences/beliefs/capabilities/emotions/goals)
2. Topic (single keyword, e.g., "conscious", "sushi", "learning")
3. Statement (the actual claim)

Return JSON array:
[
  {{"category": "preferences", "topic": "food", "statement": "My favorite food is sushi"}},
  {{"category": "identity", "topic": "conscious", "statement": "I am conscious"}}
]

If no self-claims found, return empty array: []
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

            claims = json.loads(response_clean)

            if not claims:
                logger.debug(f"No self-claims detected in {experience_id}")
                return

            logger.info(f"Detected {len(claims)} self-claim(s) in {experience_id}")

            # Create SELF_DEFINITION experience for each claim
            for claim in claims:
                self._create_self_definition_experience(
                    claim=claim,
                    parent_experience_id=experience_id,
                    interaction=interaction,
                )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse self-claim detection response: {e}")
            logger.debug(f"Raw response: {response}")
        except Exception as e:
            logger.error(f"Error detecting self-claims: {e}")

    def _create_self_definition_experience(
        self,
        claim: dict,
        parent_experience_id: str,
        interaction: InteractionPayload,
    ):
        """Create and index a SELF_DEFINITION experience.

        Args:
            claim: Dict with category, topic, statement
            parent_experience_id: ID of parent OCCURRENCE experience
            interaction: Original interaction payload
        """
        category = claim.get("category", "identity")
        topic = claim.get("topic", "")
        statement = claim.get("statement", "")

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

        # Persist SELF_DEFINITION
        self.raw_store.append_experience(self_def_experience)

        # Index immediately for fast retrieval
        self.self_knowledge_index.add_claim(
            category=category,
            topic=topic,
            experience_id=self_def_id,
        )

        logger.info(
            f"Created and indexed SELF_DEFINITION {self_def_id}: "
            f"[{category}] {topic} - {statement[:50]}..."
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
