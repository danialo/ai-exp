#!/usr/bin/env python3
"""Clean test of LLM integration with memory."""

import sys
from pathlib import Path
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.memory.raw_store import create_raw_store
from src.memory.vector_store import create_vector_store
from src.memory.embedding import create_embedding_provider
from src.pipeline.ingest import create_ingestion_pipeline, InteractionPayload
from src.services.retrieval import create_retrieval_service
from src.services.llm import create_llm_service


def main():
    """Test LLM with fresh database."""
    print("=" * 70)
    print("  🧹 FRESH LLM INTEGRATION TEST")
    print("=" * 70)
    print()

    # Check API key
    if not settings.OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found")
        sys.exit(1)

    # Clear existing data for clean test
    print("🧹 Clearing old test data...")
    db_path = Path(settings.RAW_STORE_DB_PATH)
    vec_path = Path(settings.VECTOR_INDEX_PATH)

    if db_path.exists():
        db_path.unlink()
    if vec_path.exists():
        shutil.rmtree(vec_path)

    print("✓ Clean slate ready")
    print()

    # Initialize components
    print("🔧 Initializing...")
    raw_store = create_raw_store(settings.RAW_STORE_DB_PATH)
    vector_store = create_vector_store(
        settings.VECTOR_INDEX_PATH,
        collection_name="experiences",
    )
    embedding_provider = create_embedding_provider(
        model_name=settings.EMBEDDING_MODEL,
        use_mock=False,
    )

    ingestion_pipeline = create_ingestion_pipeline(
        raw_store=raw_store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    retrieval_service = create_retrieval_service(
        raw_store=raw_store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        semantic_weight=settings.SEMANTIC_WEIGHT,
        recency_weight=settings.RECENCY_WEIGHT,
    )

    llm_service = create_llm_service(
        api_key=settings.OPENAI_API_KEY,
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )

    print(f"✓ Using {settings.LLM_MODEL}")
    print()

    # Conversation sequence
    conversations = [
        "What is the tallest mountain in the world?",
        "How tall is it?",
        "What country is it in?",
    ]

    for i, prompt in enumerate(conversations, 1):
        print("=" * 70)
        print(f"  TURN {i}/{len(conversations)}")
        print("=" * 70)
        print()
        print(f"👤 User: {prompt}")
        print()

        # Retrieve memories
        memories = []
        if vector_store.count() > 0:
            print("🔍 Checking memory...")
            results = retrieval_service.retrieve_similar(prompt=prompt, top_k=2)
            if results:
                memories = results
                print(f"   📝 Found {len(memories)} relevant memories:")
                for j, mem in enumerate(memories, 1):
                    print(f"   [{j}] User: \"{mem.prompt_text}\"")
                    print(f"       Assistant: \"{mem.response_text[:60]}...\"")
                    print(f"       Relevance: {mem.similarity_score:.3f}")
            else:
                print("   (No relevant memories)")
        else:
            print("🔍 No memories yet (first interaction)")
        print()

        # Generate response
        print("🤖 Generating response...")
        response = llm_service.generate_response(
            prompt=prompt,
            memories=memories,
        )
        print()
        print(f"Assistant: {response}")
        print()

        # Store
        print("💾 Storing...")
        interaction = InteractionPayload(prompt=prompt, response=response, valence=0.0)
        result = ingestion_pipeline.ingest_interaction(interaction)
        print(f"   ✓ Saved as {result.experience_id}")
        print()

    # Summary
    print("=" * 70)
    print("  ✅ TEST COMPLETE")
    print("=" * 70)
    print()
    print("📊 Summary:")
    print(f"   Total memories: {raw_store.count_experiences()}")
    print()
    print("Expected behavior:")
    print("  - Turn 1: Answered about Mount Everest (no prior context)")
    print("  - Turn 2: Used memory to understand 'it' refers to Everest")
    print("  - Turn 3: Used memory to know we're still talking about Everest")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
