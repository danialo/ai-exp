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
    ):
        """Initialize ingestion pipeline.

        Args:
            raw_store: Raw experience store
            vector_store: Vector index for embeddings
            embedding_provider: Embedding generator
        """
        self.raw_store = raw_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

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
