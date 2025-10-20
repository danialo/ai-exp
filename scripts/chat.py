#!/usr/bin/env python3
"""Interactive chat CLI for AI Experience Memory System.

Allows you to have conversations that get stored as experiences
and retrieves relevant past memories during each interaction.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.memory.raw_store import create_raw_store
from src.memory.vector_store import create_vector_store
from src.memory.embedding import create_embedding_provider
from src.pipeline.ingest import create_ingestion_pipeline, InteractionPayload
from src.services.retrieval import create_retrieval_service
from src.services.llm import create_llm_service


class ChatSession:
    """Interactive chat session with memory."""

    def __init__(self, use_mock_embeddings: bool = False, use_real_llm: bool = True):
        """Initialize chat session.

        Args:
            use_mock_embeddings: Use mock embeddings for testing
            use_real_llm: Use real LLM for responses (requires OPENAI_API_KEY)
        """
        print("🧠 Initializing AI Experience Memory System...")

        # Initialize components
        self.raw_store = create_raw_store(settings.RAW_STORE_DB_PATH)
        self.vector_store = create_vector_store(
            settings.VECTOR_INDEX_PATH,
            collection_name="experiences",
        )
        self.embedding_provider = create_embedding_provider(
            model_name=settings.EMBEDDING_MODEL,
            use_mock=use_mock_embeddings,
        )

        self.ingestion_pipeline = create_ingestion_pipeline(
            raw_store=self.raw_store,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
        )

        self.retrieval_service = create_retrieval_service(
            raw_store=self.raw_store,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            semantic_weight=settings.SEMANTIC_WEIGHT,
            recency_weight=settings.RECENCY_WEIGHT,
        )

        # Initialize LLM service if requested
        self.use_real_llm = use_real_llm
        self.llm_service = None
        if use_real_llm:
            if not settings.OPENAI_API_KEY:
                print("⚠️  Warning: OPENAI_API_KEY not set. Using mock responses.")
                print("   Set OPENAI_API_KEY in .env to enable real LLM responses.\n")
                self.use_real_llm = False
            else:
                self.llm_service = create_llm_service(
                    api_key=settings.OPENAI_API_KEY,
                    model=settings.LLM_MODEL,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS,
                )
                print(f"✓ Using {settings.LLM_MODEL} for responses")

        # Stats
        exp_count = self.raw_store.count_experiences()
        vec_count = self.vector_store.count()
        print(f"✓ Loaded {exp_count} experiences, {vec_count} vectors")
        print()

    def retrieve_memories(self, prompt: str, top_k: int = 3) -> list:
        """Retrieve relevant memories for a prompt.

        Args:
            prompt: User prompt
            top_k: Number of memories to retrieve

        Returns:
            List of retrieval results
        """
        if self.vector_store.count() == 0:
            return []

        results = self.retrieval_service.retrieve_similar(
            prompt=prompt,
            top_k=top_k,
        )
        return results

    def display_memories(self, memories: list) -> None:
        """Display retrieved memories.

        Args:
            memories: List of retrieval results
        """
        if not memories:
            print("  💭 No relevant memories found\n")
            return

        print(f"  💭 Retrieved {len(memories)} relevant memories:")
        for i, mem in enumerate(memories, 1):
            # Format timestamp
            time_str = mem.created_at.strftime("%Y-%m-%d %H:%M")

            # Truncate long text
            prompt_preview = mem.prompt_text[:60] + "..." if len(mem.prompt_text) > 60 else mem.prompt_text

            # Show similarity and recency
            print(f"     [{i}] {time_str} | sim={mem.similarity_score:.2f} rec={mem.recency_score:.2f}")
            print(f"         Q: {prompt_preview}")
        print()

    def generate_response(self, prompt: str, memories: list) -> str:
        """Generate response using LLM with memory context.

        Args:
            prompt: User prompt
            memories: Retrieved memories

        Returns:
            Generated response
        """
        if self.use_real_llm and self.llm_service:
            # Use real LLM with memory context
            try:
                response = self.llm_service.generate_response(
                    prompt=prompt,
                    memories=memories,
                )
                return response
            except Exception as e:
                print(f"⚠️  Error calling LLM: {e}")
                print("   Falling back to mock response.\n")
                # Fall through to mock response

        # MOCK response (used if LLM not available or errors)
        if memories:
            response = f"Based on our past conversations, I recall we discussed similar topics. "
            response += f"To answer your question: {prompt}\n\n"
            response += "[Mock response - set OPENAI_API_KEY for real LLM responses]"
        else:
            response = f"[Mock response to: {prompt}]"

        return response

    def store_interaction(self, prompt: str, response: str, valence: float = 0.0) -> str:
        """Store interaction as an experience.

        Args:
            prompt: User prompt
            response: System response
            valence: Affect valence (-1 to 1)

        Returns:
            Experience ID
        """
        interaction = InteractionPayload(
            prompt=prompt,
            response=response,
            valence=valence,
        )

        result = self.ingestion_pipeline.ingest_interaction(interaction)
        return result.experience_id

    def chat_loop(self):
        """Main chat loop."""
        print("=" * 60)
        print("AI EXPERIENCE MEMORY CHAT")
        print("=" * 60)
        print()
        print("Commands:")
        print("  - Type your message to chat")
        print("  - Type 'stats' to see memory statistics")
        print("  - Type 'quit' or 'exit' to end session")
        print()
        print("-" * 60)
        print()

        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!")
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            if user_input.lower() == "stats":
                exp_count = self.raw_store.count_experiences()
                vec_count = self.vector_store.count()
                print(f"\n📊 Memory Statistics:")
                print(f"   Experiences: {exp_count}")
                print(f"   Vectors: {vec_count}")
                print()
                continue

            # Process message
            print()

            # 1. Retrieve relevant memories
            memories = self.retrieve_memories(user_input, top_k=3)
            if memories:
                self.display_memories(memories)

            # 2. Generate response (mock)
            response = self.generate_response(user_input, memories)

            # 3. Display response
            print(f"Assistant: {response}\n")

            # 4. Store interaction
            exp_id = self.store_interaction(user_input, response)
            print(f"  💾 Stored as {exp_id}\n")
            print("-" * 60)
            print()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive chat with AI Experience Memory"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock embeddings (for testing)",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use mock LLM responses instead of real API calls",
    )
    args = parser.parse_args()

    try:
        session = ChatSession(
            use_mock_embeddings=args.mock,
            use_real_llm=not args.mock_llm,
        )
        session.chat_loop()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
