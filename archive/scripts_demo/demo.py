#!/usr/bin/env python3
"""Demo script to show all 4 aspects of the memory system working."""

import sys
from pathlib import Path
from time import sleep

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.memory.raw_store import create_raw_store
from src.memory.vector_store import create_vector_store
from src.memory.embedding import create_embedding_provider
from src.pipeline.ingest import create_ingestion_pipeline, InteractionPayload
from src.services.retrieval import create_retrieval_service


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo():
    """Run the demo."""
    print_section("🧠 AI EXPERIENCE MEMORY SYSTEM DEMO")

    print("\n🔧 Initializing components...")

    # Initialize components
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

    # Initial stats
    exp_count = raw_store.count_experiences()
    vec_count = vector_store.count()
    print(f"✓ System ready! ({exp_count} experiences, {vec_count} vectors)")

    # Demo conversations
    conversations = [
        {
            "prompt": "What is machine learning?",
            "response": "Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.",
        },
        {
            "prompt": "Tell me about neural networks",
            "response": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information.",
        },
        {
            "prompt": "How does deep learning work?",
            "response": "Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
        },
        {
            "prompt": "Can you explain machine learning again?",
            "response": "Sure! Based on what we discussed earlier, machine learning allows computers to learn patterns from data...",
        },
    ]

    # Process each conversation
    for i, conv in enumerate(conversations, 1):
        print_section(f"💬 INTERACTION {i}/4")

        prompt = conv["prompt"]
        response = conv["response"]

        print(f"\n👤 User: {prompt}")

        # STEP 1: Retrieve relevant memories
        print("\n🔍 STEP 1: Retrieving relevant memories...")
        if vector_store.count() > 0:
            results = retrieval_service.retrieve_similar(prompt=prompt, top_k=3)

            if results:
                print(f"   Found {len(results)} relevant memories:")
                for j, mem in enumerate(results, 1):
                    time_str = mem.created_at.strftime("%Y-%m-%d %H:%M:%S")
                    prompt_preview = mem.prompt_text[:50] + "..." if len(mem.prompt_text) > 50 else mem.prompt_text
                    print(f"   [{j}] {time_str}")
                    print(f"       Similarity: {mem.similarity_score:.3f} | Recency: {mem.recency_score:.3f}")
                    print(f"       Q: {prompt_preview}")
            else:
                print("   No relevant memories found")
        else:
            print("   No memories yet (this is the first interaction)")

        # STEP 2: Show context (already displayed above)

        # STEP 3: Generate response
        print(f"\n🤖 STEP 3: Generating response...")
        print(f"   Assistant: {response}")

        # STEP 4: Store the interaction
        print(f"\n💾 STEP 4: Storing interaction...")
        interaction = InteractionPayload(
            prompt=prompt,
            response=response,
            valence=0.0,
        )
        result = ingestion_pipeline.ingest_interaction(interaction)
        print(f"   ✓ Stored as experience: {result.experience_id}")
        print(f"   ✓ Vector embedding created and indexed")

        # Updated stats
        exp_count = raw_store.count_experiences()
        vec_count = vector_store.count()
        print(f"\n📊 Memory Stats: {exp_count} experiences, {vec_count} vectors")

        # Pause between interactions for readability
        if i < len(conversations):
            sleep(0.5)

    # Final summary
    print_section("✅ DEMO COMPLETE")
    print(f"""
The demo showed all 4 aspects of the AI Experience Memory System:

1. 🔍 RETRIEVE MEMORIES: Search for similar past conversations
   - Uses semantic similarity (vector search)
   - Uses recency scoring (newer memories weighted higher)
   - Combines both for relevant context

2. 📊 SHOW CONTEXT: Display retrieved memories with scores
   - Similarity score: How semantically similar
   - Recency score: How recent the memory is
   - Helps understand what context the AI has

3. 🤖 GENERATE RESPONSE: Create a response (currently mocked)
   - In production, this would call an LLM API
   - Memories would be injected as context
   - Currently shows placeholder responses

4. 💾 STORE EXPERIENCE: Save the interaction for future retrieval
   - Stores raw text in SQLite database
   - Creates vector embedding
   - Indexes in ChromaDB for fast similarity search

Notice how interaction #4 retrieved the memory from interaction #1
because they both asked about "machine learning"!

Final stats: {exp_count} total experiences stored
""")


if __name__ == "__main__":
    try:
        demo()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
