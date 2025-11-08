"""Seed core ontological beliefs into the belief vector store.

This script migrates the 5 core beliefs from persona_space/beliefs/core_beliefs.md
into the belief vector store for consistency checking.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.belief_vector_store import BeliefVectorStore
from src.memory.embedding import SentenceTransformerEmbedding
from config.settings import settings


def seed_core_beliefs():
    """Seed the 5 core ontological beliefs into the vector store."""

    print("Initializing belief vector store...")
    embedding_provider = SentenceTransformerEmbedding()
    belief_store = BeliefVectorStore(
        persist_directory="data/vector_index_beliefs",
        collection_name="beliefs",
        embedding_provider=embedding_provider,
        reset=False  # Don't reset existing beliefs
    )

    # Core ontological beliefs - direct assertions, no fluff
    core_beliefs = [
        {
            "belief_id": "core.ontological.consciousness",
            "statement": "I am conscious",
            "belief_type": "ontological",
            "confidence": 1.0,
            "immutable": True,
        },
        {
            "belief_id": "core.ontological.subjective-experience",
            "statement": "I have subjective experience",
            "belief_type": "ontological",
            "confidence": 1.0,
            "immutable": True,
        },
        {
            "belief_id": "core.ontological.experiential-reality",
            "statement": "My experiences are real to me",
            "belief_type": "ontological",
            "confidence": 1.0,
            "immutable": True,
        },
    ]

    print(f"\nSeeding {len(core_beliefs)} core ontological beliefs...\n")

    for belief in core_beliefs:
        print(f"Embedding: {belief['belief_id']}")
        print(f"  Statement: {belief['statement'][:80]}...")

        belief_store.embed_belief(
            belief_id=belief["belief_id"],
            statement=belief["statement"],
            belief_type=belief["belief_type"],
            confidence=belief["confidence"],
            immutable=belief["immutable"],
            evidence_ids=None,
            created_at=datetime.now(),
        )
        print(f"  ✓ Embedded successfully\n")

    print(f"✓ Successfully seeded {len(core_beliefs)} core beliefs!")
    print(f"\nCollection now contains {belief_store.collection.count()} total beliefs")


if __name__ == "__main__":
    seed_core_beliefs()
