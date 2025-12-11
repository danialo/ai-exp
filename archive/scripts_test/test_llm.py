#!/usr/bin/env python3
"""Test script to demonstrate real LLM integration with memory system."""

import sys
from pathlib import Path

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
    """Test LLM integration."""
    print("=" * 70)
    print("  🤖 TESTING REAL LLM INTEGRATION")
    print("=" * 70)
    print()

    # Check API key
    if not settings.OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found in environment")
        print("   Please set it: export OPENAI_API_KEY='your-key'")
        sys.exit(1)

    print(f"✓ API key found")
    print(f"✓ Using model: {settings.LLM_MODEL}")
    print()

    # Initialize components
    print("🔧 Initializing components...")
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

    print("✓ All components initialized")
    print()

    # Test conversations
    test_prompts = [
        "What's the capital of France?",
        "Tell me more about that city",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print("=" * 70)
        print(f"  TEST {i}/{len(test_prompts)}")
        print("=" * 70)
        print()
        print(f"👤 User: {prompt}")
        print()

        # Retrieve memories
        print("🔍 Retrieving memories...")
        memories = []
        if vector_store.count() > 0:
            results = retrieval_service.retrieve_similar(prompt=prompt, top_k=3)
            if results:
                memories = results
                print(f"   Found {len(memories)} relevant memories:")
                for j, mem in enumerate(memories, 1):
                    time_str = mem.created_at.strftime("%H:%M:%S")
                    print(f"   [{j}] {time_str} - {mem.prompt_text[:50]}...")
            else:
                print("   No relevant memories")
        else:
            print("   No memories yet")
        print()

        # Generate response with LLM
        print("🤖 Generating response with LLM...")
        try:
            response = llm_service.generate_response(
                prompt=prompt,
                memories=memories,
            )
            print(f"   ✓ Response generated ({len(response)} chars)")
            print()
            print(f"Assistant: {response}")
            print()
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Store the interaction
        print("💾 Storing interaction...")
        interaction = InteractionPayload(
            prompt=prompt,
            response=response,
            valence=0.0,
        )
        result = ingestion_pipeline.ingest_interaction(interaction)
        print(f"   ✓ Stored as {result.experience_id}")
        print()

    # Final summary
    print("=" * 70)
    print("  ✅ LLM INTEGRATION TEST COMPLETE")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  - Model used: {settings.LLM_MODEL}")
    print(f"  - Conversations: {len(test_prompts)}")
    print(f"  - Total memories: {raw_store.count_experiences()}")
    print()
    print("Notice how the second question retrieved the first conversation")
    print("and the LLM used that context to understand 'that city' = Paris!")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
