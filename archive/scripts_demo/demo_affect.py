#!/usr/bin/env python3
"""Demo script to show affect-aware lens in action."""

import sys
sys.path.insert(0, '/home/d/git/ai-exp')

from config.settings import settings
from src.memory.raw_store import create_raw_store
from src.memory.vector_store import create_vector_store
from src.memory.embedding import create_embedding_provider
from src.pipeline.ingest import create_ingestion_pipeline, InteractionPayload
from src.pipeline.lens import create_experience_lens
from src.pipeline.reflection import create_reflection_writer
from src.services.retrieval import create_retrieval_service
from src.services.llm import create_llm_service

print("🎭 Affect-Aware Lens Demo\n")

# Initialize components
raw_store = create_raw_store(settings.RAW_STORE_DB_PATH)
vector_store = create_vector_store(settings.VECTOR_INDEX_PATH)
embedding_provider = create_embedding_provider(settings.EMBEDDING_MODEL)
ingestion_pipeline = create_ingestion_pipeline(raw_store, vector_store, embedding_provider)
retrieval_service = create_retrieval_service(raw_store, vector_store, embedding_provider)
llm_service = create_llm_service(api_key=settings.OPENAI_API_KEY)
lens = create_experience_lens(llm_service, retrieval_service, valence_threshold=-0.2)
reflection_writer = create_reflection_writer(raw_store, vector_store, embedding_provider)

print("📝 Step 1: Creating past experiences with NEGATIVE affect about jealousy...\n")

# Seed some negative experiences about jealousy
negative_experiences = [
    ("I'm feeling really jealous and I hate it", "Jealousy can be painful. What's triggering it?", -0.8),
    ("Why do I get so jealous? It's destroying my relationships", "Jealousy often stems from insecurity or past experiences.", -0.7),
    ("I can't stop feeling jealous even though I know it's irrational", "Managing jealousy requires understanding its root causes.", -0.6),
]

for prompt, response, valence in negative_experiences:
    interaction = InteractionPayload(
        prompt=prompt,
        response=response,
        valence=valence,
    )
    result = ingestion_pipeline.ingest_interaction(interaction)
    print(f"   ✓ Stored: {prompt[:50]}... (valence={valence})")

print(f"\n📊 Total experiences now: {raw_store.count_experiences()}")
print(f"📊 Total vectors now: {vector_store.count()}\n")

print("="*70)
print("🔍 Step 2: Asking about jealousy NOW (should trigger empathetic response)\n")

# Now ask about jealousy - should get empathetic response
query = "Tell me about dealing with jealousy"

lens_result = lens.process(prompt=query, retrieve_memories=True)

print(f"Query: {query}\n")
print(f"Blended Valence: {lens_result.blended_valence:.3f} (from {len(lens_result.retrieved_experience_ids)} memories)")
print(f"Retrieved: {lens_result.retrieved_experience_ids}\n")

print("─"*70)
print("📝 DRAFT Response (without affect adjustment):")
print("─"*70)
print(lens_result.draft_response)
print()

print("─"*70)
print("✨ AUGMENTED Response (WITH affect-aware tone):")
print("─"*70)
print(lens_result.augmented_response)
print()

print("="*70)
print("\n💭 Notice the difference:")
print("   • Blended valence is NEGATIVE (past struggles)")
print("   • Augmented response has EMPATHETIC preface")
print("   • References past experiences")
print("   • Tone is supportive, not clinical")
print()

# Store this interaction too
interaction = InteractionPayload(
    prompt=query,
    response=lens_result.augmented_response,
    valence=lens_result.blended_valence,
)
result = ingestion_pipeline.ingest_interaction(interaction)

# Record reflection
reflection_writer.record_reflection(
    interaction_id=result.experience_id,
    prompt=query,
    response=lens_result.augmented_response,
    retrieved_ids=lens_result.retrieved_experience_ids,
    blended_valence=lens_result.blended_valence,
)

print(f"✅ Stored interaction: {result.experience_id}")
print(f"✅ Recorded reflection for meta-learning")
