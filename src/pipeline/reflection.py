"""Reflection shard writer for post-response learning observations.

This module captures lightweight reflection experiences after each interaction,
documenting which memories were helpful and why. Reflections are stored as
'observation' type experiences with parent links to the original interaction.

MVP scope: Template-driven reflection notes with parent experience links.
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


class ReflectionWriter:
    """Writes post-response reflection observations to memory."""

    def __init__(
        self,
        raw_store: RawStore,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
    ):
        """Initialize reflection writer.

        Args:
            raw_store: Raw experience store
            vector_store: Vector index for embeddings
            embedding_provider: Embedding generator
        """
        self.raw_store = raw_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def record_reflection(
        self,
        interaction_id: str,
        prompt: str,
        response: str,
        retrieved_ids: list[str],
        blended_valence: float = 0.0,
    ) -> str:
        """Record a reflection observation about memory retrieval.

        Args:
            interaction_id: ID of the interaction experience
            prompt: Original prompt text
            response: Generated response text
            retrieved_ids: IDs of retrieved experiences that informed the response
            blended_valence: Blended valence from retrieved memories

        Returns:
            Experience ID of the reflection observation
        """
        # Generate reflection note
        reflection_text = self._generate_reflection_note(
            prompt=prompt,
            response=response,
            retrieved_ids=retrieved_ids,
            blended_valence=blended_valence,
        )

        logger.info(f"Recording reflection for {interaction_id} ({len(retrieved_ids)} memories)")

        # Generate experience ID for reflection
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        hash_suffix = hash(interaction_id + reflection_text) % 10000
        reflection_id = f"obs_{timestamp}_{hash_suffix:04x}"

        # Generate embedding for reflection
        reflection_embedding = self.embedding_provider.embed(reflection_text)

        # Create vector ID
        reflection_vec_id = f"{reflection_id}_reflection"

        # Store embedding in vector index with 'reflection' label
        self.vector_store.upsert(
            id=reflection_vec_id,
            vector=reflection_embedding,
            metadata={
                "experience_id": reflection_id,
                "role": EmbeddingRole.PROMPT_SEMANTIC.value,  # Reuse semantic role
                "label": "reflection",  # Tag as reflection
                "text_preview": reflection_text[:100],
                "parent_interaction": interaction_id,
                "retrieved_count": len(retrieved_ids),
            },
        )

        # Create embedding pointers
        embedding_pointers = EmbeddingPointers(
            semantic=f"vec://sem/{reflection_vec_id}",
        )

        # Create affect (inherit from blended valence)
        affect = AffectModel(
            vad=VAD(v=blended_valence, a=0.0, d=0.0),
            labels=["reflection"],
            intensity=abs(blended_valence),
            confidence=0.3,  # Lower confidence for derived affect
        )

        # Build content
        content = ContentModel(
            text=reflection_text,
            structured={
                "reflection_note": reflection_text,
                "parent_interaction": interaction_id,
                "retrieved_experiences": retrieved_ids,
                "retrieved_count": len(retrieved_ids),
                "blended_valence": blended_valence,
            },
        )

        # Build reflection experience as OBSERVATION type
        reflection_exp = ExperienceModel(
            id=reflection_id,
            type=ExperienceType.OBSERVATION,  # Observation type for reflections
            content=content,
            provenance=ProvenanceModel(
                actor=Actor.AGENT,  # Agent generates reflections
                method=CaptureMethod.MODEL_INFER,  # Derived from model processing
                sources=[],
            ),
            embeddings=embedding_pointers,
            affect=affect,
            parents=[interaction_id] + retrieved_ids,  # Link to parent and sources
        )

        # Persist to raw store
        logger.debug(f"Persisting reflection {reflection_id} to raw store")
        stored_id = self.raw_store.append_experience(reflection_exp)

        logger.info(f"Recorded reflection {reflection_id} for {interaction_id}")
        return stored_id

    def _generate_reflection_note(
        self,
        prompt: str,
        response: str,
        retrieved_ids: list[str],
        blended_valence: float,
    ) -> str:
        """Generate template-driven reflection note.

        Args:
            prompt: User's prompt
            response: Generated response
            retrieved_ids: Retrieved experience IDs
            blended_valence: Blended valence score

        Returns:
            Reflection note text
        """
        if not retrieved_ids:
            return (
                f"Responded to query about '{self._truncate(prompt, 50)}' "
                f"without retrieving prior experiences. "
                f"This may indicate a novel topic or fresh start."
            )

        # Categorize affect for reflection
        affect_desc = self._describe_affect(blended_valence)

        # Build reflection note
        note_parts = [
            f"Retrieved {len(retrieved_ids)} relevant experience(s) for query: '{self._truncate(prompt, 50)}'.",
            f"Memory context showed {affect_desc} affect (valence={blended_valence:.2f}).",
        ]

        # Add detail about memory usage
        if len(retrieved_ids) == 1:
            note_parts.append(
                f"Used experience [{retrieved_ids[0]}] to inform response context."
            )
        else:
            exp_list = ", ".join([f"[{eid}]" for eid in retrieved_ids[:3]])
            if len(retrieved_ids) > 3:
                exp_list += f", and {len(retrieved_ids) - 3} more"
            note_parts.append(
                f"Synthesized context from experiences: {exp_list}."
            )

        # Add response summary
        note_parts.append(
            f"Response focused on: {self._extract_response_focus(response)}."
        )

        return " ".join(note_parts)

    def _describe_affect(self, valence: float) -> str:
        """Describe affect based on valence score.

        Args:
            valence: Valence score (-1 to 1)

        Returns:
            Affect description string
        """
        if valence < -0.5:
            return "strongly negative"
        elif valence < -0.2:
            return "mildly negative"
        elif valence > 0.5:
            return "strongly positive"
        elif valence > 0.2:
            return "mildly positive"
        else:
            return "neutral"

    def _extract_response_focus(self, response: str, max_length: int = 60) -> str:
        """Extract the main focus of the response (first sentence or clause).

        Args:
            response: Full response text
            max_length: Maximum length of extracted focus

        Returns:
            Response focus summary
        """
        # Try to get first sentence
        first_sentence = response.split(".")[0]
        if len(first_sentence) > max_length:
            # Truncate at word boundary
            truncated = first_sentence[:max_length].rsplit(" ", 1)[0]
            return truncated + "..."
        return first_sentence

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max length at word boundary.

        Args:
            text: Text to truncate
            max_length: Maximum length

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        truncated = text[:max_length].rsplit(" ", 1)[0]
        return truncated + "..."


def create_reflection_writer(
    raw_store: RawStore,
    vector_store: VectorStore,
    embedding_provider: EmbeddingProvider,
) -> ReflectionWriter:
    """Factory function to create reflection writer.

    Args:
        raw_store: Raw store instance
        vector_store: Vector store instance
        embedding_provider: Embedding provider instance

    Returns:
        ReflectionWriter instance
    """
    return ReflectionWriter(
        raw_store=raw_store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )
