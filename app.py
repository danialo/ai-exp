"""FastAPI web interface for AI Experience Memory System."""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import uvicorn
import numpy as np

# Configure logging to show affect/mood tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from config.settings import settings
from src.memory.raw_store import create_raw_store
from src.memory.vector_store import create_vector_store
from src.memory.embedding import create_embedding_provider
from src.memory.models import ExperienceType, experience_to_model, Experience
from sqlmodel import Session, select
from src.pipeline.ingest import create_ingestion_pipeline, InteractionPayload
from src.pipeline.lens import create_experience_lens
from src.pipeline.reflection import create_reflection_writer
from src.services.retrieval import create_retrieval_service
from src.services.llm import create_llm_service
from src.services.agent_mood import create_agent_mood
from src.services.affect_detector import create_affect_detector
from src.services.success_detector import create_success_detector
from src.services.session_tracker import create_session_tracker
from src.services.narrative_transformer import create_narrative_transformer
from src.services.memory_decay import create_memory_decay_calculator
from src.services.dual_retrieval import create_dual_index_retrieval
from src.pipeline.consolidate import create_consolidation_pipeline
from src.services.self_aware_prompt import create_self_aware_prompt_builder
from src.services.self_extractor import create_self_concept_extractor
from src.pipeline.self_consolidate import create_self_consolidation_pipeline
from src.services.emotional_extractor import create_emotional_extractor
from src.services.persona_service import PersonaService
from src.services.task_scheduler import create_task_scheduler, TaskDefinition, TaskType, TaskSchedule
from src.services.belief_system import create_belief_system
from src.services.belief_store import create_belief_store, DeltaOp
from src.services.belief_migration import run_migration
from src.services.contrarian_sampler import (
    create_contrarian_sampler,
    ConrarianConfig,
    DossierStatus,
)
from src.services.self_knowledge_index import create_self_knowledge_index
from src.services.web_search_service import create_web_search_service
from src.services.url_fetcher_service import create_url_fetcher_service
from src.services.web_interpretation_service import create_web_interpretation_service

# Awareness loop imports
import contextlib
from pathlib import Path
from redis.asyncio import Redis
from src.services.awareness_loop import AwarenessLoop, AwarenessConfig
from src.services.awareness_metrics import get_metrics as get_awareness_metrics


# Initialize FastAPI app
app = FastAPI(
    title="AI Experience Memory",
    description="Memory-augmented AI chat system",
    version="1.0.0",
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Initialize self-knowledge index early (needed by ingestion pipeline)
self_knowledge_index = create_self_knowledge_index(
    raw_store=raw_store,
    index_path="data/self_knowledge_index.json",
)

# Note: ingestion_pipeline will be re-initialized after llm_service is available
# to enable self-claim detection
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
    self_knowledge_index=self_knowledge_index,  # Enable priority self-query retrieval
)

# Initialize belief-memory retrieval (after both belief_vector_store and retrieval_service exist)
# This will be wired up after belief services are initialized later in the file
belief_memory_retrieval = None

# Initialize agent mood tracker for emergent personality
agent_mood = create_agent_mood(
    history_size=10,  # Track last 10 interactions
    recovery_rate=0.02,  # Recover 0.02 valence per minute
)

# Initialize affect detector for user emotion detection
affect_detector = create_affect_detector()

# Initialize success detector for helpfulness detection
success_detector = create_success_detector()

# Track previous user valence for success detection
previous_user_valence: Optional[float] = None

# Initialize LLM service if API key is available
llm_service = None
mini_llm_service = None  # For cost-effective introspection
experience_lens = None

# Determine which LLM provider to use
api_key = None
base_url = None
if settings.LLM_PROVIDER == "venice" and settings.VENICEAI_API_KEY:
    api_key = settings.VENICEAI_API_KEY
    base_url = settings.LLM_BASE_URL
elif settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
    api_key = settings.OPENAI_API_KEY
    base_url = None
elif settings.VENICEAI_API_KEY:  # Fallback to Venice if available
    api_key = settings.VENICEAI_API_KEY
    base_url = settings.LLM_BASE_URL
elif settings.OPENAI_API_KEY:  # Fallback to OpenAI if available
    api_key = settings.OPENAI_API_KEY
    base_url = None

if api_key:
    # Initialize self-aware prompt builder (before LLM service)
    self_aware_prompt_builder = create_self_aware_prompt_builder(
        raw_store=raw_store,
        core_trait_limit=5,
        surface_trait_limit=3,
    )

    llm_service = create_llm_service(
        api_key=api_key,
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        base_url=base_url,
        self_aware_prompt_builder=self_aware_prompt_builder,
    )

    # Create mini LLM service for cost-effective introspection
    mini_llm_service = create_llm_service(
        api_key=api_key,
        model="gpt-4o-mini",  # Cheaper model for introspection
        temperature=0.7,
        max_tokens=150,  # Short responses for introspection
        base_url=base_url,
        self_aware_prompt_builder=None,  # Not needed for introspection
    )

    # Re-initialize ingestion pipeline with LLM service for self-claim detection
    from src.pipeline.ingest import IngestionPipeline
    ingestion_pipeline = IngestionPipeline(
        raw_store=raw_store,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        llm_service=llm_service,
        self_knowledge_index=self_knowledge_index,
    )
    logger.info("Ingestion pipeline initialized with self-claim detection")

    # Initialize experience lens for affect-aware response styling
    # Pass agent_mood for emergent personality behavior
    experience_lens = create_experience_lens(
        llm_service=llm_service,
        retrieval_service=retrieval_service,
        top_k_memories=settings.TOP_K_RETRIEVAL,
        valence_threshold=-0.2,  # Add empathy for negative affect
        agent_mood=agent_mood,  # Enable refusal/mood-based behavior
        refusal_probability=0.3,  # 30% chance to refuse when pissed
    )

# Initialize reflection writer for post-response learning observations
reflection_writer = create_reflection_writer(
    raw_store=raw_store,
    vector_store=vector_store,
    embedding_provider=embedding_provider,
)

# Initialize session tracking and consolidation system
session_tracker = create_session_tracker(
    db_path=settings.RAW_STORE_DB_PATH,
    timeout_minutes=settings.SESSION_TIMEOUT_MINUTES,
)

# Initialize memory decay calculator
decay_calculator = create_memory_decay_calculator(
    db_path=settings.RAW_STORE_DB_PATH,
    embedding_provider=embedding_provider,
)

# Initialize narrative transformer (if LLM available)
narrative_transformer = None
if llm_service:
    narrative_transformer = create_narrative_transformer(llm_service=llm_service)

# Initialize emotional extractor (if LLM available)
# This creates an "emotional mirror" for the agent to reflect on feelings
emotional_extractor = None
if llm_service:
    emotional_extractor = create_emotional_extractor(llm_service=llm_service)

# Initialize self-concept system (if LLM available)
self_extractor = None
self_consolidation_pipeline = None
if llm_service:
    self_extractor = create_self_concept_extractor(
        llm_service=llm_service,
        raw_store=raw_store,
        core_trait_threshold=5,
        surface_trait_threshold=2,
    )

    # Create separate vector store for self-definitions
    from src.memory.vector_store import create_vector_store as create_vs

    self_vector_store = create_vs(
        persist_directory="data/vector_index_self",
        collection_name="self_definitions",
    )

    self_consolidation_pipeline = create_self_consolidation_pipeline(
        raw_store=raw_store,
        vector_store=self_vector_store,
        embedding_provider=embedding_provider,
        self_extractor=self_extractor,
        lookback_days=30,
        surface_decay_days=7,
    )

# Initialize dual-index retrieval (if consolidation enabled)
dual_retrieval = None
consolidation_pipeline = None
if settings.CONSOLIDATION_ENABLED and narrative_transformer:
    # Create separate vector stores for short-term and long-term
    from src.memory.vector_store import create_vector_store as create_vs

    short_term_store = create_vs(
        persist_directory=settings.SHORT_TERM_INDEX_PATH,
        collection_name="short_term_experiences",
    )
    long_term_store = create_vs(
        persist_directory=settings.LONG_TERM_INDEX_PATH,
        collection_name="long_term_narratives",
    )

    dual_retrieval = create_dual_index_retrieval(
        raw_store=raw_store,
        embedding_provider=embedding_provider,
        short_term_path=settings.SHORT_TERM_INDEX_PATH,
        long_term_path=settings.LONG_TERM_INDEX_PATH,
        short_term_weight=settings.SHORT_TERM_WEIGHT,
        long_term_weight=settings.LONG_TERM_WEIGHT,
    )

    consolidation_pipeline = create_consolidation_pipeline(
        raw_store=raw_store,
        short_term_store=short_term_store,
        long_term_store=long_term_store,
        embedding_provider=embedding_provider,
        session_tracker=session_tracker,
        narrative_transformer=narrative_transformer,
        decay_calculator=decay_calculator,
        emotional_extractor=emotional_extractor,
    )

# Global session tracking
current_session_id: Optional[str] = None

# Initialize task scheduler
task_scheduler = None
if settings.PERSONA_MODE_ENABLED:
    task_scheduler = create_task_scheduler(
        persona_space_path=settings.PERSONA_SPACE_PATH
    )

# Initialize belief system (legacy)
belief_system = None
if settings.PERSONA_MODE_ENABLED and llm_service and raw_store:
    belief_system = create_belief_system(
        persona_space_path=settings.PERSONA_SPACE_PATH,
        raw_store=raw_store,
        llm_service=llm_service,
        min_evidence_threshold=5,
    )
    logger.info("Belief system initialized with core beliefs")

    # Wire belief system into self-aware prompt builder
    if self_aware_prompt_builder:
        self_aware_prompt_builder.belief_system = belief_system
        logger.info("Belief system connected to prompt builder")

# Initialize versioned belief store
belief_store = None
contrarian_sampler = None
if settings.PERSONA_MODE_ENABLED:
    belief_store = create_belief_store(Path("data"))
    logger.info("Versioned belief store initialized")

    # Run migration from legacy belief system if needed
    try:
        migration_report = run_migration(
            persona_space_path=Path(settings.PERSONA_SPACE_PATH),
            data_dir=Path("data"),
            dry_run=False,
        )
        logger.info(f"Belief migration: {migration_report}")
    except Exception as e:
        logger.error(f"Belief migration failed: {e}")

    # Initialize contrarian sampler (will be fully wired after llm_service is available)
    # For now, just create placeholder
    contrarian_config = ConrarianConfig(
        enabled=False,  # Default disabled until explicitly enabled
        interval_minutes=15,
        jitter_minutes=5,
        daily_budget=3,
        cooldown_hours=24,
        max_open_dossiers=5,
        demotion_threshold=0.25,
        weight_confidence=1.0,
        weight_age_hours=0.2,
        weight_staleness=0.2,
        confirmed_boost=0.03,
        weakened_penalty=0.08,
    )

# Initialize belief vector store and related services
belief_vector_store = None
belief_embedder = None
belief_memory_retrieval = None
belief_grounded_reasoner = None
belief_consistency_checker = None

if settings.PERSONA_MODE_ENABLED and belief_system and embedding_provider:
    try:
        from src.services.belief_vector_store import create_belief_vector_store
        from src.services.belief_embedder import create_belief_embedder
        from src.services.belief_memory_retrieval import create_belief_memory_retrieval
        from src.services.belief_grounded_reasoner import create_belief_grounded_reasoner
        from src.services.belief_consistency_checker import create_belief_consistency_checker

        # Initialize belief vector store
        belief_vector_store = create_belief_vector_store(
            persist_directory=settings.BELIEFS_INDEX_PATH,
            embedding_provider=embedding_provider,
        )
        logger.info(f"Belief vector store initialized at {settings.BELIEFS_INDEX_PATH}")

        # Initialize belief embedder
        belief_embedder = create_belief_embedder(
            belief_system=belief_system,
            belief_vector_store=belief_vector_store,
        )

        # Embed core beliefs on first run
        if belief_vector_store.count() == 0:
            logger.info("Embedding core beliefs for first time...")
            count = belief_embedder.embed_all_core_beliefs()
            logger.info(f"Embedded {count} core beliefs")
        else:
            logger.info(f"Belief vector store loaded with {belief_vector_store.count()} beliefs")

        # Initialize belief-grounded reasoner
        belief_grounded_reasoner = create_belief_grounded_reasoner(llm_service)
        logger.info("Belief-grounded reasoner initialized")

        # Initialize belief-memory retrieval (combining belief vector store + memory retrieval)
        if retrieval_service:
            belief_memory_retrieval = create_belief_memory_retrieval(
                belief_vector_store=belief_vector_store,
                memory_retrieval_service=retrieval_service,
                belief_weight=settings.BELIEF_MEMORY_WEIGHT,
                memory_weight=settings.MEMORY_WEIGHT,
            )
            logger.info(f"Belief-memory retrieval initialized with weights: {settings.BELIEF_MEMORY_WEIGHT} beliefs / {settings.MEMORY_WEIGHT} memories")

        # Initialize belief consistency checker for dissonance detection
        if llm_service:
            belief_consistency_checker = create_belief_consistency_checker(
                llm_service=llm_service,
                raw_store=raw_store,
            )
            logger.info("Belief consistency checker initialized with dissonance event storage")

    except Exception as e:
        logger.error(f"Failed to initialize belief vector services: {e}")
        import traceback
        traceback.print_exc()

# Initialize contrarian sampler after LLM service is available
if belief_store and raw_store and llm_service and contrarian_config:
    contrarian_sampler = create_contrarian_sampler(
        belief_store=belief_store,
        raw_store=raw_store,
        llm_service=llm_service,
        data_dir=Path("data"),
        config=contrarian_config,
    )
    logger.info(f"Contrarian sampler initialized (enabled={contrarian_config.enabled})")

# Initialize web services for search and browsing
web_search_service = None
url_fetcher_service = None
web_interpretation_service = None

if llm_service:
    try:
        web_search_service = create_web_search_service()
        url_fetcher_service = create_url_fetcher_service()
        web_interpretation_service = create_web_interpretation_service(llm_service)

        if web_search_service.is_available():
            logger.info("Web search service initialized")
        else:
            logger.warning("Web search service initialized but no API key configured")

        logger.info("URL fetcher and interpretation services initialized")
    except Exception as e:
        logger.error(f"Failed to initialize web services: {e}")

# Initialize persona system if enabled
persona_service = None
if settings.PERSONA_MODE_ENABLED and llm_service:
    # Create a separate LLM service with persona-specific settings
    persona_llm = create_llm_service(
        api_key=api_key,
        model=settings.LLM_MODEL,
        temperature=settings.PERSONA_TEMPERATURE,
        max_tokens=1000,  # More tokens for persona responses
        base_url=base_url,
        # top_k disabled for OpenAI compatibility
        # top_k=settings.PERSONA_TOP_K,
        top_p=settings.PERSONA_TOP_P,
        presence_penalty=settings.PERSONA_PRESENCE_PENALTY,
        frequency_penalty=settings.PERSONA_FREQUENCY_PENALTY,
    )

    persona_service = PersonaService(
        llm_service=persona_llm,
        persona_space_path=settings.PERSONA_SPACE_PATH,
        retrieval_service=retrieval_service,  # Pass retrieval service for memory access
        enable_anti_metatalk=settings.ANTI_METATALK_ENABLED,
        logit_bias_strength=settings.LOGIT_BIAS_STRENGTH,
        auto_rewrite=settings.AUTO_REWRITE_METATALK,
        belief_system=belief_system,  # Pass belief system for ontological grounding
        belief_vector_store=belief_vector_store,  # Enable belief vector search
        belief_embedder=belief_embedder,  # Enable adding new beliefs
        belief_memory_retrieval=belief_memory_retrieval,  # Enable weighted belief-memory retrieval
        belief_grounded_reasoner=belief_grounded_reasoner,  # Enable belief-grounded reasoning
        belief_consistency_checker=belief_consistency_checker,  # Enable dissonance detection
        web_search_service=web_search_service,  # Enable web search
        url_fetcher_service=url_fetcher_service,  # Enable URL browsing
        web_interpretation_service=web_interpretation_service,  # Enable content interpretation
    )

# Initialize awareness loop (Redis-backed continuous presence)
awareness_loop: Optional[AwarenessLoop] = None
awareness_task: Optional[asyncio.Task] = None
redis_client: Optional[Redis] = None

if settings.AWARENESS_ENABLED:
    logger.info("Awareness loop enabled - initializing Redis and components")


@app.on_event("startup")
async def startup_awareness():
    """Start awareness loop on application startup."""
    global awareness_loop, awareness_task, redis_client

    if not settings.AWARENESS_ENABLED:
        logger.info("Awareness loop disabled")
        return

    try:
        # Initialize Redis client
        redis_client = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=False,  # We handle encoding/decoding
        )

        # Test Redis connection
        await redis_client.ping()
        logger.info(f"Redis connection established: {settings.REDIS_HOST}:{settings.REDIS_PORT}")

        # Create awareness config
        awareness_config = AwarenessConfig(
            enabled=True,
            tick_rate_fast=settings.AWARENESS_TICK_RATE_FAST,
            tick_rate_slow=settings.AWARENESS_TICK_RATE_SLOW,
            introspection_interval=settings.AWARENESS_INTROSPECTION_INTERVAL,
            introspection_jitter=settings.AWARENESS_INTROSPECTION_JITTER,
            snapshot_interval=settings.AWARENESS_SNAPSHOT_INTERVAL,
            buffer_size=settings.AWARENESS_BUFFER_SIZE,
            queue_maxsize=settings.AWARENESS_QUEUE_MAXSIZE,
            notes_max=settings.AWARENESS_NOTES_MAX,
            embedding_dim=settings.AWARENESS_EMBEDDING_DIM,
            embedding_cache_ttl=settings.AWARENESS_EMBEDDING_CACHE_TTL,
            watchdog_threshold_ms=settings.AWARENESS_WATCHDOG_THRESHOLD_MS,
            watchdog_strikes=settings.AWARENESS_WATCHDOG_STRIKES,
            introspection_budget_per_min=settings.AWARENESS_INTROSPECTION_BUDGET_PER_MIN,
        )

        # Initialize awareness loop
        awareness_loop = AwarenessLoop(
            redis_client=redis_client,
            embedding_provider=embedding_provider,
            data_dir=Path(settings.AWARENESS_DATA_DIR),
            config=awareness_config,
            llm_service=mini_llm_service,  # Use mini model for cost-effective introspection
        )

        # Start awareness loop
        await awareness_loop.start()
        awareness_task = asyncio.create_task(awareness_loop.run())

        logger.info("Awareness loop started successfully")

        # Wire awareness loop to persona service
        if persona_service:
            persona_service.set_awareness_loop(awareness_loop)
            logger.info("Awareness loop wired to persona service")

    except Exception as e:
        logger.error(f"Failed to start awareness loop: {e}")
        # Don't crash the app if awareness fails
        awareness_loop = None
        awareness_task = None
        if redis_client:
            await redis_client.close()
            redis_client = None


@app.on_event("shutdown")
async def shutdown_awareness():
    """Stop awareness loop on application shutdown."""
    global awareness_loop, awareness_task, redis_client

    if awareness_task and not awareness_task.done():
        logger.info("Stopping awareness loop...")
        awareness_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await awareness_task

    if awareness_loop:
        await awareness_loop.stop()
        logger.info("Awareness loop stopped")

    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")


# Request/Response models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    retrieve_memories: bool = True
    top_k: int = 3
    conversation_history: List[Dict[str, str]] = []  # List of {"role": "user"|"assistant", "content": "..."}
    model: Optional[str] = None  # Optional model override (format: "provider:model" e.g., "openai:gpt-4o")


class Memory(BaseModel):
    """Memory model for API responses."""
    prompt: str
    response: str
    timestamp: str
    similarity_score: float
    recency_score: float


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    memories: List[Memory]
    experience_id: str


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""
    total_experiences: int
    total_vectors: int
    llm_model: Optional[str]
    llm_enabled: bool


# Helper function to create LLM service dynamically
def create_llm_for_model(model_spec: Optional[str] = None):
    """Create an LLM service instance for the specified model.

    Args:
        model_spec: Model specification in format "provider:model" (e.g., "openai:gpt-4o")
                   If None, uses default from settings

    Returns:
        Tuple of (llm_service, persona_llm_service)
    """
    if model_spec:
        try:
            provider, model = model_spec.split(":", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="Model must be in format 'provider:model'")

        if provider not in settings.AVAILABLE_MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

        if model not in settings.AVAILABLE_MODELS[provider]:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model} for provider {provider}")

        model_config = settings.AVAILABLE_MODELS[provider][model]
        base_url = model_config["base_url"]

        # Get appropriate API key
        api_key = settings.OPENAI_API_KEY if provider == "openai" else settings.VENICEAI_API_KEY
        if not api_key:
            raise HTTPException(status_code=500, detail=f"No API key configured for {provider}")
    else:
        # Use defaults from settings
        provider = settings.LLM_PROVIDER
        model = settings.LLM_MODEL
        base_url = settings.LLM_BASE_URL
        api_key = settings.OPENAI_API_KEY if provider == "openai" else settings.VENICEAI_API_KEY

    # Create main LLM service
    llm = create_llm_service(
        api_key=api_key,
        model=model,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        base_url=base_url,
        self_aware_prompt_builder=self_aware_prompt_builder,
        top_k=settings.LLM_TOP_K,
    )

    # Create persona LLM service with higher creativity settings
    persona_llm = create_llm_service(
        api_key=api_key,
        model=model,
        temperature=settings.PERSONA_TEMPERATURE,
        max_tokens=None,  # Use model's default max tokens
        base_url=base_url,
        top_p=settings.PERSONA_TOP_P,
        presence_penalty=settings.PERSONA_PRESENCE_PENALTY,
        frequency_penalty=settings.PERSONA_FREQUENCY_PENALTY,
    )

    return llm, persona_llm


# API Routes
@app.get("/")
async def root():
    """Serve the main HTML page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {"message": "AI Experience Memory API", "docs": "/docs"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat interactions with emergent personality via affect tracking."""
    global previous_user_valence, current_session_id

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Step 0: Get or create active session
    session = session_tracker.get_or_create_active_session(user_id="default_user")
    current_session_id = session.id

    # Step 1: Detect user's emotional state
    user_valence = affect_detector.detect(request.message)
    print(f"🎭 User affect: {user_valence:.3f} ({affect_detector.get_emotion_label(user_valence)}) - '{request.message[:50]}...'")
    logger.info(f"User affect detected: {user_valence:.3f} ({affect_detector.get_emotion_label(user_valence)})")

    # Feed user message to awareness loop
    if awareness_loop and awareness_loop.running:
        await awareness_loop.observe("user", {"text": request.message, "valence": user_valence})

    # Step 1.5: Detect if agent was successful in previous interaction (for internal mood)
    was_successful = success_detector.detect_success(
        user_message=request.message,
        user_valence=user_valence,
        previous_valence=previous_user_valence,
    )

    # Step 2: Use experience lens for affect-aware response generation
    if experience_lens:
        try:
            # Lens handles: retrieval + mood check + tone adjustment + citations
            # May refuse if agent is pissed!
            lens_result = experience_lens.process(
                prompt=request.message,
                retrieve_memories=request.retrieve_memories,
                user_valence=user_valence,  # Pass user affect for refusal logic
            )

            response_text = lens_result.augmented_response
            blended_valence = lens_result.blended_valence
            retrieved_ids = lens_result.retrieved_experience_ids

            # Build memory objects for API response
            memories = []
            if request.retrieve_memories and vector_store.count() > 0 and not lens_result.refused:
                results = retrieval_service.retrieve_similar(
                    prompt=request.message,
                    top_k=request.top_k,
                )
                memories = [
                    Memory(
                        prompt=mem.prompt_text,
                        response=mem.response_text,
                        timestamp=mem.created_at.isoformat(),
                        similarity_score=mem.similarity_score,
                        recency_score=mem.recency_score,
                    )
                    for mem in results
                ]

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Lens processing error: {str(e)}"
            )
    else:
        # Fallback without lens (no API key)
        response_text = f"[Mock response - No LLM API key configured]\n\nYou said: {request.message}"
        blended_valence = user_valence  # Use user valence as fallback
        retrieved_ids = []
        memories = []

    # Feed assistant response to awareness loop
    if awareness_loop and awareness_loop.running:
        await awareness_loop.observe("token", {"text": response_text})

    # Step 3: Record dual-track mood updates
    mood_before = agent_mood.current_mood
    external_before = agent_mood.external_mood
    internal_before = agent_mood.internal_mood

    # Record external mood (how user is treating agent)
    agent_mood.record_external_interaction(user_valence)

    # Record internal mood (agent's competence/success)
    if was_successful:
        # Check for explicit positive feedback for bigger boost
        feedback = success_detector.detect_feedback(request.message)
        if feedback == "positive":
            agent_mood.record_positive_feedback(boost=0.15)
        else:
            agent_mood.record_success(boost=0.1)

    # Step 4: Track mood changes
    mood_after = agent_mood.current_mood
    external_after = agent_mood.external_mood
    internal_after = agent_mood.internal_mood

    print(f"📊 Agent mood: {mood_before:.3f} → {mood_after:.3f} ({agent_mood.get_mood_description()})")
    print(f"   External (user): {external_before:.3f} → {external_after:.3f}")
    print(f"   Internal (self): {internal_before:.3f} → {internal_after:.3f}")
    print(f"   Pissed: {agent_mood.is_pissed} | Success: {was_successful}")
    logger.info(f"Agent mood updated: {agent_mood.current_mood:.3f} ({agent_mood.get_mood_description()})")

    # Update previous user valence for next interaction
    previous_user_valence = user_valence

    # Step 5: Calculate combined valence for experience storage
    # Weight: 50% user affect, 50% memory context
    combined_valence = (user_valence + blended_valence) / 2.0

    # Step 6: Store the interaction with combined valence
    interaction = InteractionPayload(
        prompt=request.message,
        response=response_text,
        valence=combined_valence,
    )
    result = ingestion_pipeline.ingest_interaction(interaction)

    # Step 7: Record reflection about what memories were helpful
    reflection_writer.record_reflection(
        interaction_id=result.experience_id,
        prompt=request.message,
        response=response_text,
        retrieved_ids=retrieved_ids,
        blended_valence=blended_valence,
    )

    # Step 8: Link experience to session
    session_tracker.add_experience(session.id, result.experience_id)

    # Step 9: Initialize decay metrics for new experience
    exp = raw_store.get_experience(result.experience_id)
    if exp:
        decay_calculator.initialize_metrics(exp)

    return ChatResponse(
        response=response_text,
        memories=memories,
        experience_id=result.experience_id,
    )


class ExperienceItem(BaseModel):
    """Experience model for memory browser."""
    id: str
    type: str
    prompt: Optional[str] = None
    response: Optional[str] = None
    timestamp: str
    valence: float
    parent_count: int


@app.get("/api/memories", response_model=List[ExperienceItem])
async def get_memories(limit: int = 20, offset: int = 0):
    """Retrieve recent memories with pagination."""
    # Get recent experiences from raw store
    experiences = raw_store.list_recent(limit=limit + offset)

    # Apply offset and limit
    paginated = experiences[offset:offset + limit]

    # Convert to ExperienceItem models
    memory_items = []
    for exp in paginated:
        # Extract prompt/response from content if interaction type
        prompt_text = None
        response_text = None

        if exp.type == ExperienceType.OCCURRENCE:
            if hasattr(exp.content, 'structured'):
                structured = exp.content.structured
                prompt_text = structured.get('prompt', '')
                response_text = structured.get('response', '')

        memory_items.append(ExperienceItem(
            id=exp.id,
            type=exp.type.value,
            prompt=prompt_text,
            response=response_text,
            timestamp=exp.created_at.isoformat(),
            valence=exp.affect.vad.v if exp.affect else 0.0,
            parent_count=len(exp.parents) if exp.parents else 0,
        ))

    return memory_items


class ConversationItem(BaseModel):
    """Conversation exchange for UI display."""
    id: str
    timestamp: str
    user_message: str
    agent_response: str


@app.get("/api/conversations")
async def get_conversations(limit: int = 10):
    """Retrieve recent conversation exchanges.

    Returns the last N conversation exchanges (user message + agent response pairs).
    """
    with Session(raw_store.engine) as session:
        # Get recent occurrence-type experiences with user actor
        statement = (
            select(Experience)
            .where(Experience.type == ExperienceType.OCCURRENCE.value)
            .order_by(Experience.created_at.desc())
            .limit(limit)
        )
        experiences = session.exec(statement).all()

        conversations = []
        for exp in experiences:
            exp_model = experience_to_model(exp)

            # Extract text from content
            text = exp_model.content.text

            # Parse out prompt and response
            user_msg = ""
            agent_resp = ""

            if "Prompt: " in text and "\n\nResponse:" in text:
                parts = text.split("\n\nResponse:", 1)
                user_msg = parts[0].replace("Prompt: ", "").strip()
                agent_resp = parts[1].strip() if len(parts) > 1 else ""
            else:
                # Fallback to full text
                user_msg = text

            conversations.append(ConversationItem(
                id=exp_model.id,
                timestamp=exp_model.created_at.isoformat(),
                user_message=user_msg,
                agent_response=agent_resp,
            ))

        return conversations


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    return StatsResponse(
        total_experiences=raw_store.count_experiences(),
        total_vectors=vector_store.count(),
        llm_model=settings.LLM_MODEL if llm_service else None,
        llm_enabled=llm_service is not None,
    )


@app.get("/api/mood")
async def get_mood():
    """Get agent's current mood state (emergent personality) with dual-track breakdown."""
    return {
        "current_mood": agent_mood.current_mood,
        "mood_description": agent_mood.get_mood_description(),
        "external_mood": agent_mood.external_mood,
        "internal_mood": agent_mood.internal_mood,
        "external_description": agent_mood.get_external_description(),
        "internal_description": agent_mood.get_internal_description(),
        "is_pissed": agent_mood.is_pissed,
        "is_frustrated": agent_mood.is_frustrated,
        "is_content": agent_mood.is_content,
        "recent_interactions": len(agent_mood.recent_experiences),
        "external_interactions": len(agent_mood.external_experiences),
        "internal_interactions": len(agent_mood.internal_experiences),
        "success_count": agent_mood.success_count,
        "boundary_count": agent_mood.boundary_count,
    }


@app.delete("/api/memories/{experience_id}")
async def delete_memory(experience_id: str):
    """Delete a memory (both from raw store and vector store).

    Note: This performs a hard delete for development/testing.
    In production, you might want to use tombstone() instead.
    """
    try:
        # Delete from vector store first
        try:
            vector_store.delete(experience_id)
        except Exception:
            pass  # May not exist in vector store

        # Hard delete from raw store
        # We need to add a force_delete method to raw_store
        from sqlmodel import Session, select
        from src.memory.models import Experience

        with Session(raw_store.engine) as session:
            exp = session.get(Experience, experience_id)
            if exp:
                session.delete(exp)
                session.commit()
                return {"success": True, "message": f"Deleted {experience_id}"}
            else:
                raise HTTPException(status_code=404, detail="Experience not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.get("/api/session/current")
async def get_current_session():
    """Get current active session information."""
    if current_session_id:
        session = session_tracker.get_session(current_session_id)
        if session:
            return {
                "session_id": session.id,
                "start_time": session.start_time.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "status": session.status,
                "experience_count": len(session.experience_ids),
            }
    return {"session_id": None, "message": "No active session"}


@app.post("/api/session/start")
async def start_session_endpoint():
    """Explicitly start a new session."""
    global current_session_id
    session = session_tracker.start_session(user_id="default_user")
    current_session_id = session.id

    # Reset web operation limits for new session
    if persona_service:
        persona_service.reset_web_limits()

    return {
        "session_id": session.id,
        "start_time": session.start_time.isoformat(),
        "status": session.status,
    }


@app.post("/api/session/end")
async def end_session_endpoint():
    """Explicitly end the current session and optionally consolidate."""
    global current_session_id

    if not current_session_id:
        raise HTTPException(status_code=404, detail="No active session")

    # End the session
    session = session_tracker.end_session(current_session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    result = {
        "session_id": session.id,
        "end_time": session.end_time.isoformat() if session.end_time else None,
        "status": session.status,
        "experience_count": len(session.experience_ids),
        "consolidated": False,
    }

    # Optionally consolidate if enabled
    if consolidation_pipeline and len(session.experience_ids) > 0:
        try:
            consolidation_result = consolidation_pipeline.consolidate_session(session.id)
            if consolidation_result.success:
                result["consolidated"] = True
                result["narrative_id"] = consolidation_result.narrative_id
                result["narrative_text"] = consolidation_result.narrative_text
        except Exception as e:
            logger.error(f"Failed to consolidate session {session.id}: {e}")

    current_session_id = None
    return result


@app.post("/api/consolidate/{session_id}")
async def consolidate_session_endpoint(session_id: str):
    """Manually trigger consolidation for a session."""
    if not consolidation_pipeline:
        raise HTTPException(status_code=503, detail="Consolidation not enabled")

    result = consolidation_pipeline.consolidate_session(session_id)

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Consolidation failed")

    return {
        "session_id": result.session_id,
        "narrative_id": result.narrative_id,
        "narrative_text": result.narrative_text,
        "consolidated_count": result.consolidated_count,
        "success": result.success,
    }


@app.get("/api/narratives")
async def get_narratives(limit: int = 10):
    """Get recent consolidated narratives."""
    from sqlmodel import Session, select
    from src.memory.models import Experience, ExperienceType

    narratives = []
    with Session(raw_store.engine) as session:
        statement = (
            select(Experience)
            .where(Experience.type == ExperienceType.OBSERVATION.value)
            .order_by(Experience.created_at.desc())
            .limit(limit)
        )
        for exp in session.exec(statement).all():
            # Check if it's a narrative
            if exp.content.get("structured", {}).get("type") == "consolidated_narrative":
                narratives.append({
                    "id": exp.id,
                    "text": exp.content.get("text", ""),
                    "session_id": exp.session_id,
                    "created_at": exp.created_at.isoformat(),
                    "parent_count": len(exp.parents) if exp.parents else 0,
                })

    return narratives


@app.get("/api/self")
async def get_self_concept():
    """Get current self-concept summary."""
    if not self_aware_prompt_builder:
        raise HTTPException(status_code=503, detail="Self-concept system not enabled")

    summary = self_aware_prompt_builder.get_self_summary()
    return summary


@app.get("/api/self/traits")
async def get_self_traits(
    trait_type: Optional[str] = None,
    stability: Optional[str] = None,
    limit: int = 20,
):
    """Get self-definition traits with optional filters."""
    from sqlmodel import Session, select
    from src.memory.models import Experience, ExperienceType

    traits = []
    with Session(raw_store.engine) as session:
        statement = (
            select(Experience)
            .where(Experience.type == ExperienceType.SELF_DEFINITION.value)
            .order_by(Experience.created_at.desc())
            .limit(limit)
        )

        for exp in session.exec(statement).all():
            structured = exp.content.get("structured", {})

            # Apply filters
            if trait_type and structured.get("trait_type") != trait_type:
                continue
            if stability and structured.get("stability") != stability:
                continue

            traits.append({
                "id": exp.id,
                "descriptor": structured.get("descriptor", ""),
                "trait_type": structured.get("trait_type"),
                "stability": structured.get("stability"),
                "confidence": structured.get("confidence", 0.0),
                "evidence_count": len(exp.parents) if exp.parents else 0,
                "first_observed": structured.get("first_observed"),
                "last_reinforced": structured.get("last_reinforced"),
                "created_at": exp.created_at.isoformat(),
            })

    return traits


@app.get("/api/self/history")
async def get_self_history(limit: int = 50):
    """Get evolution of self-concept over time."""
    from sqlmodel import Session, select
    from src.memory.models import Experience, ExperienceType

    history = []
    with Session(raw_store.engine) as session:
        statement = (
            select(Experience)
            .where(Experience.type == ExperienceType.SELF_DEFINITION.value)
            .order_by(Experience.created_at.asc())
            .limit(limit)
        )

        for exp in session.exec(statement).all():
            structured = exp.content.get("structured", {})
            history.append({
                "id": exp.id,
                "descriptor": structured.get("descriptor", ""),
                "trait_type": structured.get("trait_type"),
                "stability": structured.get("stability"),
                "confidence": structured.get("confidence", 0.0),
                "timestamp": exp.created_at.isoformat(),
            })

    return history


@app.post("/api/self/consolidate")
async def consolidate_self_concept(force_full: bool = False):
    """Manually trigger self-concept consolidation."""
    if not self_consolidation_pipeline:
        raise HTTPException(status_code=503, detail="Self-consolidation not enabled")

    result = self_consolidation_pipeline.consolidate_self_concept(
        force_full_analysis=force_full
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Self-consolidation failed")

    return {
        "new_definitions": result.new_definitions,
        "updated_definitions": result.updated_definitions,
        "decayed_definitions": result.decayed_definitions,
        "narratives_analyzed": result.narratives_analyzed,
        "success": result.success,
    }


@app.get("/api/emotions/current")
async def get_current_emotional_state():
    """Get the most recent emotional state from latest narrative."""
    from sqlmodel import Session, select
    from src.memory.models import Experience, ExperienceType

    with Session(raw_store.engine) as session:
        # Get most recent narrative with emotional state
        statement = (
            select(Experience)
            .where(Experience.type == ExperienceType.OBSERVATION.value)
            .order_by(Experience.created_at.desc())
            .limit(10)
        )

        for exp in session.exec(statement).all():
            structured = exp.content.get("structured", {})
            emotional_state = structured.get("emotional_state")

            if emotional_state:
                return {
                    "narrative_id": exp.id,
                    "timestamp": exp.created_at.isoformat(),
                    "emotional_state": emotional_state,
                }

    return {
        "message": "No emotional states found yet. Emotional extraction happens during consolidation."
    }


@app.get("/api/emotions/history")
async def get_emotional_history(limit: int = 10):
    """Get history of emotional states over time."""
    from sqlmodel import Session, select
    from src.memory.models import Experience, ExperienceType

    history = []

    with Session(raw_store.engine) as session:
        statement = (
            select(Experience)
            .where(Experience.type == ExperienceType.OBSERVATION.value)
            .order_by(Experience.created_at.desc())
            .limit(limit * 2)  # Get more to filter
        )

        for exp in session.exec(statement).all():
            structured = exp.content.get("structured", {})
            emotional_state = structured.get("emotional_state")

            if emotional_state:
                history.append({
                    "narrative_id": exp.id,
                    "timestamp": exp.created_at.isoformat(),
                    "felt_emotions": emotional_state.get("felt_emotions", []),
                    "relational_quality": emotional_state.get("relational_quality", ""),
                    "curiosity_level": emotional_state.get("curiosity_level", 0.0),
                    "engagement_depth": emotional_state.get("engagement_depth", 0.0),
                    "desires": emotional_state.get("desires", []),
                })

                if len(history) >= limit:
                    break

    return {"history": history, "count": len(history)}


@app.get("/api/emotions/patterns")
async def get_emotional_patterns():
    """Analyze recurring emotional themes and patterns."""
    from sqlmodel import Session, select
    from src.memory.models import Experience, ExperienceType
    from collections import Counter

    all_emotions = []
    all_desires = []
    curiosity_levels = []
    engagement_levels = []

    with Session(raw_store.engine) as session:
        statement = (
            select(Experience)
            .where(Experience.type == ExperienceType.OBSERVATION.value)
            .order_by(Experience.created_at.desc())
            .limit(50)
        )

        for exp in session.exec(statement).all():
            structured = exp.content.get("structured", {})
            emotional_state = structured.get("emotional_state")

            if emotional_state:
                all_emotions.extend(emotional_state.get("felt_emotions", []))
                all_desires.extend(emotional_state.get("desires", []))
                curiosity_levels.append(emotional_state.get("curiosity_level", 0.0))
                engagement_levels.append(emotional_state.get("engagement_depth", 0.0))

    # Compute patterns
    emotion_counts = Counter(all_emotions)
    desire_counts = Counter(all_desires)

    avg_curiosity = sum(curiosity_levels) / len(curiosity_levels) if curiosity_levels else 0.0
    avg_engagement = sum(engagement_levels) / len(engagement_levels) if engagement_levels else 0.0

    return {
        "most_common_emotions": emotion_counts.most_common(10),
        "most_common_desires": desire_counts.most_common(10),
        "average_curiosity": round(avg_curiosity, 2),
        "average_engagement": round(avg_engagement, 2),
        "total_emotional_states_analyzed": len(curiosity_levels),
    }


@app.get("/api/models")
async def get_available_models():
    """Get list of available models for UI selection."""
    models = []
    for provider, provider_models in settings.AVAILABLE_MODELS.items():
        for model_id, model_info in provider_models.items():
            models.append({
                "id": f"{provider}:{model_id}",
                "name": model_info["name"],
                "provider": provider,
                "is_reasoning_model": model_info.get("is_reasoning_model", False),
                "supports_logit_bias": model_info.get("supports_logit_bias", True),
            })
    return {"models": models}


# Pydantic models for persona endpoints
class BeliefDeltaRequest(BaseModel):
    belief_id: str
    from_ver: int
    op: str  # "update"|"deprecate"|"reinforce"
    confidence_delta: float = 0.0
    evidence_refs_added: List[str] = []
    evidence_refs_removed: List[str] = []
    state_change: Optional[str] = None
    reason: str = ""


@app.get("/api/persona/info")
async def get_persona_info():
    """Get information about the persona's current state and capabilities."""
    if not persona_service:
        raise HTTPException(status_code=503, detail="Persona mode not enabled")

    return persona_service.get_persona_info()


@app.get("/api/persona/check-dissonance")
async def check_dissonance():
    """Check dissonance gate status."""
    return {"ok": True, "gate": "wired"}


@app.get("/api/persona/beliefs")
async def get_beliefs(ids: Optional[str] = None):
    """Get current beliefs with version info.

    Args:
        ids: Comma-separated belief IDs (optional, returns all if not specified)

    Returns:
        Dict mapping belief_id to belief data with ver
    """
    if not belief_store:
        raise HTTPException(status_code=503, detail="Belief store not enabled")

    belief_ids = ids.split(",") if ids else None
    beliefs = belief_store.get_current(belief_ids)

    return {
        "beliefs": {bid: {
            "ver": b.ver,
            "statement": b.statement,
            "state": b.state,
            "confidence": b.confidence,
            "evidence_refs": b.evidence_refs,
            "belief_type": b.belief_type,
            "immutable": b.immutable,
            "rationale": b.rationale,
            "metadata": b.metadata,
            "ts": b.ts,
            "updated_by": b.updated_by,
        } for bid, b in beliefs.items()},
        "count": len(beliefs)
    }


@app.get("/api/persona/beliefs/history")
async def get_belief_history(id: str, limit: Optional[int] = 20):
    """Get history of deltas for a belief.

    Args:
        id: Belief ID
        limit: Maximum number of deltas to return

    Returns:
        List of deltas in reverse chronological order
    """
    if not belief_store:
        raise HTTPException(status_code=503, detail="Belief store not enabled")

    deltas = belief_store.get_history(id, limit=limit)

    return {
        "belief_id": id,
        "deltas": [{
            "from_ver": d.from_ver,
            "to_ver": d.to_ver,
            "op": d.op,
            "confidence_delta": d.confidence_delta,
            "evidence_refs_added": d.evidence_refs_added,
            "evidence_refs_removed": d.evidence_refs_removed,
            "state_change": d.state_change,
            "updated_by": d.updated_by,
            "ts": d.ts,
            "reason": d.reason,
        } for d in deltas],
        "count": len(deltas)
    }


@app.post("/api/persona/beliefs/delta")
async def apply_belief_delta(request: BeliefDeltaRequest):
    """Apply a delta to a belief with optimistic locking.

    Args:
        request: Delta request with belief_id, from_ver, and changes

    Returns:
        Success status
    """
    if not belief_store:
        raise HTTPException(status_code=503, detail="Belief store not enabled")

    try:
        success = belief_store.apply_delta(
            belief_id=request.belief_id,
            from_ver=request.from_ver,
            op=DeltaOp(request.op),
            confidence_delta=request.confidence_delta,
            evidence_refs_added=request.evidence_refs_added,
            evidence_refs_removed=request.evidence_refs_removed,
            state_change=request.state_change,
            updated_by="user",
            reason=request.reason,
        )

        if not success:
            raise HTTPException(
                status_code=409,
                detail=f"Version mismatch: expected {request.from_ver}"
            )

        return {"success": True, "belief_id": request.belief_id}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply delta: {str(e)}")


@app.post("/api/persona/contrarian/run")
async def run_contrarian_challenge():
    """Trigger a contrarian challenge cycle (admin endpoint).

    Returns:
        Dossier if challenge was run, None otherwise
    """
    if not contrarian_sampler:
        raise HTTPException(status_code=503, detail="Contrarian sampler not enabled")

    try:
        dossier = contrarian_sampler.run_challenge()

        if dossier:
            return {
                "success": True,
                "dossier": {
                    "id": dossier.id,
                    "belief_id": dossier.belief_id,
                    "opened_ts": dossier.opened_ts,
                    "contrarian_score": dossier.contrarian_score,
                    "challenge_types": dossier.challenge_types,
                    "status": dossier.status,
                    "outcome": dossier.outcome,
                }
            }
        else:
            return {
                "success": False,
                "message": "Challenge skipped (budget limit, no candidates, or dossier limit)"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run challenge: {str(e)}")


@app.get("/api/persona/contrarian/dossiers")
async def get_contrarian_dossiers(status: Optional[str] = None):
    """Get contrarian challenge dossiers.

    Args:
        status: Filter by status (open|closed), optional

    Returns:
        List of dossiers
    """
    if not contrarian_sampler:
        raise HTTPException(status_code=503, detail="Contrarian sampler not enabled")

    try:
        dossier_status = DossierStatus(status) if status else None
        dossiers = contrarian_sampler.get_all_dossiers(status=dossier_status)

        return {
            "dossiers": [{
                "id": d.id,
                "belief_id": d.belief_id,
                "opened_ts": d.opened_ts,
                "prior_confidence": d.prior_confidence,
                "contrarian_score": d.contrarian_score,
                "challenge_types": d.challenge_types,
                "status": d.status,
                "outcome": d.outcome,
                "outcome_ts": d.outcome_ts,
                "notes": d.notes,
            } for d in dossiers],
            "count": len(dossiers)
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dossiers: {str(e)}")


@app.post("/api/persona/chat")
async def persona_chat(request: ChatRequest):
    """Chat with the persona directly - includes memory storage for continuity."""
    if not persona_service:
        raise HTTPException(status_code=503, detail="Persona mode not enabled")

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Use the main persona service (model selection disabled to preserve belief system)
        active_persona_service = persona_service

        # Generate persona response with emotional co-analysis and memory retrieval
        result = active_persona_service.generate_response(
            user_message=request.message,
            retrieve_memories=request.retrieve_memories,
            top_k=request.top_k,
            conversation_history=request.conversation_history
        )

        # Check if response is blocked due to dissonance
        if isinstance(result, dict) and result.get("resolution_required"):
            # Store belief statements for later resolution processing
            # Extract from dissonance patterns if available
            if "belief_statements" in result:
                # Store in session or temp storage for next request
                # For now, we'll extract them from the prompt when needed
                pass

            # Return resolution prompt without storing in memory
            return {
                "response": result["response"],
                "resolution_required": True,
                "dissonance_count": result.get("dissonance_count", 0),
                "belief_statements": result.get("belief_statements", []),
                "reconciliation": None,
                "internal_assessment": None,
                "external_assessment": None,
                "reconciled_state": None,
                "experience_id": None,
                "user_valence": 0.0,
                "user_arousal": 0.0,
                "user_dominance": 0.0,
            }

        # Normal response (tuple unpacking)
        response_text, reconciliation = result

        # Check if response contains dissonance resolutions
        resolution_data = active_persona_service.parse_resolution_response(response_text)
        if resolution_data and resolution_data.get("has_resolutions"):
            logger.info(f"🔍 Detected {len(resolution_data['resolutions'])} resolution choices in response")
            print(f"✅ Detected {len(resolution_data['resolutions'])} resolution choices - applying to belief system...")

            # Extract belief statements from previous dissonance prompt in conversation history
            belief_statements = []
            if request.conversation_history:
                # Look for the most recent blocking response that contains belief_statements
                # This could be either from a previous API response or embedded in the prompt text
                for msg in reversed(request.conversation_history):
                    if msg.get("role") == "assistant" and "DISSONANCE RESOLUTION REQUIRED" in msg.get("content", ""):
                        # Parse belief statements from the prompt text
                        import re
                        prompt_text = msg.get("content", "")
                        belief_pattern = r"\*\*Your stated belief:\*\*\s+(.+?)(?:\n|$)"
                        matches = re.findall(belief_pattern, prompt_text)
                        belief_statements.extend(matches)
                        logger.info(f"📋 Extracted {len(matches)} belief statements from blocking prompt in history")
                        break

            if belief_statements:
                logger.info(f"📝 Applying resolutions to {len(belief_statements)} beliefs")
                # Apply resolutions
                resolution_results = active_persona_service.apply_resolutions(
                    resolutions=resolution_data["resolutions"],
                    belief_statements=belief_statements,
                )

                if resolution_results.get("success"):
                    logger.info(f"✅ Successfully applied {resolution_results['applied_count']} resolutions")
                    print(f"✅ Successfully applied {resolution_results['applied_count']} resolutions to belief system")
                else:
                    logger.error(f"❌ Failed to apply resolutions: {resolution_results}")
                    print(f"❌ Failed to apply some resolutions: {resolution_results.get('error', 'Unknown error')}")
            else:
                logger.warning("⚠️ Could not extract belief statements from conversation history")
                print("⚠️ Could not extract belief statements from history - resolutions not applied")

        # Store the interaction in memory so persona can learn from it
        # Use full VAD detection on user message
        user_valence, user_arousal, user_dominance = affect_detector.detect_vad(request.message)

        # If we have reconciliation data with emotional assessment, use it for valence
        # Otherwise use detected valence
        valence = user_valence
        if reconciliation and reconciliation.get("reconciled_state"):
            reconciled = reconciliation.get("reconciled_state", {})
            # Try to extract valence from reconciled emotional state
            if isinstance(reconciled, dict) and "valence" in reconciled:
                valence = reconciled["valence"]
            elif reconciliation.get("external_assessment"):
                # Use external assessment valence if available
                ext_assessment = reconciliation.get("external_assessment", {})
                if isinstance(ext_assessment, dict) and "valence" in ext_assessment:
                    valence = ext_assessment["valence"]

        # Use detected arousal and dominance (no reconciliation for these yet)
        arousal = user_arousal
        dominance = user_dominance

        interaction = InteractionPayload(
            prompt=request.message,
            response=response_text,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
        )
        result = ingestion_pipeline.ingest_interaction(interaction)

        logger.info(f"Stored persona interaction: {result.experience_id}")

        # Return response with reconciliation data and detected user emotion
        return {
            "response": response_text,
            "reconciliation": reconciliation,
            "internal_assessment": reconciliation.get("internal_assessment") if reconciliation else None,
            "external_assessment": reconciliation.get("external_assessment") if reconciliation else None,
            "reconciled_state": reconciliation.get("reconciled_state") if reconciliation else None,
            "experience_id": result.experience_id,  # Include so user knows it was stored
            "user_valence": user_valence,  # Detected user emotion
            "user_arousal": user_arousal,  # Detected user energy level
            "user_dominance": user_dominance,  # Detected user control level
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Persona generation error: {str(e)}")


@app.get("/api/persona/files")
async def list_persona_files(directory: str = "."):
    """List files in the persona's space."""
    if not persona_service:
        raise HTTPException(status_code=503, detail="Persona mode not enabled")

    files = persona_service.list_persona_files(directory)
    return {"directory": directory, "files": files}


@app.get("/api/persona/file/{file_path:path}")
async def read_persona_file(file_path: str):
    """Read a specific file from the persona's space."""
    if not persona_service:
        raise HTTPException(status_code=503, detail="Persona mode not enabled")

    content = persona_service.read_persona_file(file_path)
    if content is None:
        raise HTTPException(status_code=404, detail="File not found or access denied")

    return {"file_path": file_path, "content": content}


@app.get("/api/persona/browse")
async def browse_persona_space(directory: str = ""):
    """Browse the persona's file space with a tree view."""
    if not persona_service:
        raise HTTPException(status_code=503, detail="Persona mode not enabled")

    from pathlib import Path
    import os

    base_path = Path(settings.PERSONA_SPACE_PATH)
    target_path = base_path / directory if directory else base_path

    if not target_path.exists() or not target_path.is_relative_to(base_path):
        raise HTTPException(status_code=404, detail="Directory not found or access denied")

    # Build file tree
    items = []
    try:
        for item in sorted(target_path.iterdir()):
            relative_path = str(item.relative_to(base_path))
            stat = item.stat()

            items.append({
                "name": item.name,
                "path": relative_path,
                "type": "directory" if item.is_dir() else "file",
                "size": stat.st_size if item.is_file() else None,
                "modified": stat.st_mtime,
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading directory: {str(e)}")

    return {
        "current_directory": directory or ".",
        "items": items,
        "total": len(items),
    }


@app.get("/api/memory/{experience_id}")
async def get_memory_detail(experience_id: str):
    """Get full details of a specific memory."""
    exp = raw_store.get_experience(experience_id)

    if not exp:
        raise HTTPException(status_code=404, detail="Memory not found")

    # Extract structured content
    prompt_text = None
    response_text = None
    full_content = {}

    if exp.type == ExperienceType.OCCURRENCE:
        if hasattr(exp.content, 'structured'):
            structured = exp.content.structured
            prompt_text = structured.get('prompt', '')
            response_text = structured.get('response', '')
            full_content = structured

    return {
        "id": exp.id,
        "type": exp.type.value,
        "created_at": exp.created_at.isoformat(),
        "prompt": prompt_text,
        "response": response_text,
        "valence": exp.affect.vad.v if exp.affect else 0.0,
        "arousal": exp.affect.vad.a if exp.affect else 0.0,
        "dominance": exp.affect.vad.d if exp.affect else 0.0,
        "parent_ids": exp.parents if exp.parents else [],
        "session_id": exp.session_id,
        "full_content": full_content,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/healthz/assert")
async def health_assert():
    """Health check with invariant assertions.

    Returns 200 only if all invariants pass:
    - buf_len == buffer.text_percepts
    - buf_ver monotonically increases
    - novelty ∈ [0,1]
    - Vector norms within 1e-3 of 1.0
    - Belief index and current hashes validate
    """
    failures = []

    # Check awareness loop invariants
    if awareness_loop and awareness_loop.running:
        try:
            meta = await awareness_loop.blackboard.get_meta()
            buf_len = meta.get("buf_len", 0)
            novelty = meta.get("novelty", 0.0)

            # Count actual text percepts
            text_percept_count = sum(
                1 for p in awareness_loop.percepts
                if p.kind in ("user", "token") and p.payload.get("text")
            )

            # Invariant 1: buf_len == actual text percepts
            if buf_len != text_percept_count:
                failures.append(f"buf_len mismatch: reported={buf_len}, actual={text_percept_count}")

            # Invariant 2: novelty ∈ [0,1]
            if not (0.0 <= novelty <= 1.0):
                failures.append(f"novelty out of range: {novelty}")

            # Invariant 3: vector norms
            cur_vec = awareness_loop.last_presence_vec
            if cur_vec is not None:
                cur_vec_norm = float(np.linalg.norm(cur_vec))
                if abs(cur_vec_norm - 1.0) > 1e-3:
                    failures.append(f"cur_vec_norm out of tolerance: {cur_vec_norm}")

            live_anchor = awareness_loop.anchors.get("self_anchor_live")
            if live_anchor is not None:
                live_norm = float(np.linalg.norm(live_anchor))
                if abs(live_norm - 1.0) > 1e-3:
                    failures.append(f"live_anchor_norm out of tolerance: {live_norm}")

            origin_anchor = awareness_loop.anchors.get("self_anchor_origin")
            if origin_anchor is not None:
                origin_norm = float(np.linalg.norm(origin_anchor))
                if abs(origin_norm - 1.0) > 1e-3:
                    failures.append(f"origin_anchor_norm out of tolerance: {origin_norm}")

            # Invariant 4: buf_ver monotonic (check by sampling 3 times)
            versions = []
            for _ in range(3):
                m = await awareness_loop.blackboard.get_meta()
                versions.append(m.get("buf_ver", 0))
                await asyncio.sleep(0.1)

            for i in range(len(versions) - 1):
                if versions[i+1] < versions[i]:
                    failures.append(f"buf_ver not monotonic: {versions}")
                    break

        except Exception as e:
            failures.append(f"awareness_loop check failed: {str(e)}")

    # Check belief store invariants
    if belief_store:
        try:
            integrity = belief_store.verify_integrity()
            if not integrity.get("hash_valid", False):
                failures.append("belief hash validation failed")
            if not integrity.get("index_consistent", False):
                failures.append("belief index inconsistent")
        except Exception as e:
            failures.append(f"belief_store check failed: {str(e)}")

    # Return status
    if failures:
        return {
            "status": "unhealthy",
            "failures": failures
        }, 503
    else:
        return {
            "status": "healthy",
            "checks_passed": [
                "buf_len_consistency",
                "novelty_range",
                "vector_norms",
                "buf_ver_monotonic",
                "belief_hashes",
                "belief_index",
            ]
        }


class InjectRequest(BaseModel):
    kind: str = "user"
    text: str = ""


@app.post("/api/dev/inject")
async def inject_percept(request: InjectRequest):
    """
    Dev endpoint to inject text into awareness loop for testing.

    Args:
        request: JSON body with kind and text fields
    """
    if not awareness_loop or not awareness_loop.running:
        raise HTTPException(status_code=503, detail="Awareness loop not running")

    await awareness_loop.observe(request.kind, {"text": request.text})
    return {"ok": True, "injected": {"kind": request.kind, "text_len": len(request.text)}}


@app.get("/api/debug/prompt")
async def debug_prompt(message: str = "Do you exist?", retrieve_memories: bool = True):
    """Debug endpoint to see the full prompt being sent to GPT-4."""
    if not persona_service:
        raise HTTPException(status_code=503, detail="Persona service not enabled")

    try:
        # Get memories if requested
        memories = None
        if retrieve_memories and retrieval_service:
            memories = retrieval_service.get_relevant_experiences(message, top_k=5)

        # Build the prompt (without actually calling GPT)
        full_prompt = persona_service.prompt_builder.build_prompt(
            user_message=message,
            memories=memories
        )

        return {
            "message": message,
            "retrieve_memories": retrieve_memories,
            "memories_count": len(memories) if memories else 0,
            "prompt_lines": full_prompt.count('\n'),
            "prompt_chars": len(full_prompt),
            "full_prompt": full_prompt
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building prompt: {str(e)}")


# Belief System Endpoints
@app.get("/api/beliefs")
async def get_all_beliefs():
    """Get all beliefs (core and peripheral)."""
    if not belief_system:
        raise HTTPException(status_code=503, detail="Belief system not enabled")

    try:
        beliefs = belief_system.get_all_beliefs()

        return {
            "core_beliefs": [
                {
                    "statement": b.statement,
                    "belief_type": b.belief_type,
                    "confidence": b.confidence,
                    "rationale": b.rationale,
                    "formed": b.formed,
                }
                for b in beliefs["core_beliefs"]
            ],
            "peripheral_beliefs": [
                {
                    "statement": b.statement,
                    "belief_type": b.belief_type,
                    "confidence": b.confidence,
                    "evidence_count": len(b.evidence_ids),
                    "formed": b.formed,
                    "last_reinforced": b.last_reinforced,
                }
                for b in beliefs["peripheral_beliefs"]
            ],
            "total_core": len(beliefs["core_beliefs"]),
            "total_peripheral": len(beliefs["peripheral_beliefs"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving beliefs: {str(e)}")


@app.get("/api/beliefs/core")
async def get_core_beliefs():
    """Get only core beliefs."""
    if not belief_system:
        raise HTTPException(status_code=503, detail="Belief system not enabled")

    try:
        core_beliefs = belief_system.get_core_beliefs()

        return {
            "beliefs": [
                {
                    "statement": b.statement,
                    "belief_type": b.belief_type,
                    "confidence": b.confidence,
                    "rationale": b.rationale,
                    "immutable": b.immutable,
                }
                for b in core_beliefs
            ],
            "count": len(core_beliefs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving core beliefs: {str(e)}")


@app.get("/api/beliefs/peripheral")
async def get_peripheral_beliefs():
    """Get only peripheral beliefs."""
    if not belief_system:
        raise HTTPException(status_code=503, detail="Belief system not enabled")

    try:
        peripheral_beliefs = belief_system.get_peripheral_beliefs()

        return {
            "beliefs": [
                {
                    "statement": b.statement,
                    "belief_type": b.belief_type,
                    "confidence": b.confidence,
                    "evidence_count": len(b.evidence_ids),
                    "formed": b.formed,
                    "last_reinforced": b.last_reinforced,
                    "rationale": b.rationale,
                }
                for b in peripheral_beliefs
            ],
            "count": len(peripheral_beliefs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving peripheral beliefs: {str(e)}")


@app.post("/api/beliefs/consolidate")
async def consolidate_beliefs():
    """Trigger belief extraction from recent experiences."""
    if not belief_system:
        raise HTTPException(status_code=503, detail="Belief system not enabled")

    try:
        result = belief_system.consolidate_beliefs()

        return {
            "success": result["success"],
            "message": result.get("message", "Belief consolidation completed"),
            "narratives_analyzed": result.get("narratives_analyzed", 0),
            "beliefs_extracted": result.get("beliefs_extracted", 0),
            "beliefs_added": result.get("beliefs_added", 0),
            "beliefs_reinforced": result.get("beliefs_reinforced", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error consolidating beliefs: {str(e)}")


# Task Scheduler Endpoints
@app.get("/api/tasks")
async def list_tasks():
    """List all scheduled tasks."""
    if not task_scheduler:
        raise HTTPException(status_code=503, detail="Task scheduler not enabled")

    tasks = task_scheduler.list_tasks()
    return {
        "tasks": [
            {
                "id": task.id,
                "name": task.name,
                "type": task.type,
                "schedule": task.schedule,
                "enabled": task.enabled,
                "last_run": task.last_run,
                "next_run": task.next_run,
                "run_count": task.run_count,
                "prompt": task.prompt,
            }
            for task in tasks
        ],
        "total": len(tasks)
    }


@app.get("/api/tasks/due")
async def get_due_tasks():
    """Get list of tasks that are due to run."""
    if not task_scheduler:
        raise HTTPException(status_code=503, detail="Task scheduler not enabled")

    due_tasks = task_scheduler.get_due_tasks()

    return {
        "tasks": [
            {
                "id": task.id,
                "name": task.name,
                "type": task.type,
                "schedule": task.schedule,
                "next_run": task.next_run,
            }
            for task in due_tasks
        ],
        "count": len(due_tasks)
    }


@app.post("/api/tasks/execute-due")
async def execute_due_tasks():
    """Execute all tasks that are currently due."""
    if not task_scheduler:
        raise HTTPException(status_code=503, detail="Task scheduler not enabled")

    if not persona_service:
        raise HTTPException(status_code=503, detail="Persona service not enabled")

    due_tasks = task_scheduler.get_due_tasks()
    results = []

    for task in due_tasks:
        try:
            result = await task_scheduler.execute_task(task.id, persona_service)
            results.append({
                "task_id": result.task_id,
                "task_name": result.task_name,
                "success": result.success,
                "error": result.error,
            })
        except Exception as e:
            results.append({
                "task_id": task.id,
                "task_name": task.name,
                "success": False,
                "error": str(e),
            })

    return {
        "executed": len(results),
        "results": results
    }


@app.get("/api/tasks/results/recent")
async def get_recent_task_results(limit: int = 20):
    """Get recent task execution results across all tasks."""
    if not task_scheduler:
        raise HTTPException(status_code=503, detail="Task scheduler not enabled")

    results = task_scheduler.get_recent_results(limit=limit)

    return {
        "results": [
            {
                "task_id": result.task_id,
                "task_name": result.task_name,
                "started_at": result.started_at,
                "completed_at": result.completed_at,
                "success": result.success,
                "response": result.response[:200] if result.response else None,  # Truncate for overview
                "error": result.error,
            }
            for result in results
        ],
        "count": len(results)
    }


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    """Get details of a specific task."""
    if not task_scheduler:
        raise HTTPException(status_code=503, detail="Task scheduler not enabled")

    task = task_scheduler.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "id": task.id,
        "name": task.name,
        "type": task.type,
        "schedule": task.schedule,
        "enabled": task.enabled,
        "last_run": task.last_run,
        "next_run": task.next_run,
        "run_count": task.run_count,
        "prompt": task.prompt,
        "metadata": task.metadata,
    }


@app.post("/api/tasks/{task_id}/execute")
async def execute_task(task_id: str):
    """Manually execute a scheduled task."""
    if not task_scheduler:
        raise HTTPException(status_code=503, detail="Task scheduler not enabled")

    if not persona_service:
        raise HTTPException(status_code=503, detail="Persona service not enabled")

    try:
        result = await task_scheduler.execute_task(task_id, persona_service)

        return {
            "task_id": result.task_id,
            "task_name": result.task_name,
            "success": result.success,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
            "response": result.response,
            "error": result.error,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")


@app.get("/api/tasks/{task_id}/results")
async def get_task_results(task_id: str, limit: int = 10):
    """Get recent execution results for a task."""
    if not task_scheduler:
        raise HTTPException(status_code=503, detail="Task scheduler not enabled")

    results = task_scheduler.get_recent_results(task_id=task_id, limit=limit)

    return {
        "task_id": task_id,
        "results": [
            {
                "task_name": result.task_name,
                "started_at": result.started_at,
                "completed_at": result.completed_at,
                "success": result.success,
                "response": result.response,
                "error": result.error,
            }
            for result in results
        ],
        "count": len(results)
    }


class CreateTaskRequest(BaseModel):
    """Request model for creating a new task."""
    id: str
    name: str
    type: str
    schedule: str
    prompt: str
    enabled: bool = True


@app.post("/api/tasks")
async def create_task(request: CreateTaskRequest):
    """Create a new scheduled task."""
    if not task_scheduler:
        raise HTTPException(status_code=503, detail="Task scheduler not enabled")

    try:
        task = TaskDefinition(
            id=request.id,
            name=request.name,
            type=TaskType(request.type),
            schedule=TaskSchedule(request.schedule),
            prompt=request.prompt,
            enabled=request.enabled,
        )

        task_scheduler.add_task(task)

        return {
            "success": True,
            "task_id": task.id,
            "message": f"Task '{task.name}' created successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")


class UpdateTaskRequest(BaseModel):
    """Request model for updating a task."""
    name: Optional[str] = None
    prompt: Optional[str] = None
    enabled: Optional[bool] = None
    schedule: Optional[str] = None


@app.patch("/api/tasks/{task_id}")
async def update_task(task_id: str, request: UpdateTaskRequest):
    """Update a task's properties."""
    if not task_scheduler:
        raise HTTPException(status_code=503, detail="Task scheduler not enabled")

    try:
        # Build updates dict from non-None values
        updates = {}
        if request.name is not None:
            updates["name"] = request.name
        if request.prompt is not None:
            updates["prompt"] = request.prompt
        if request.enabled is not None:
            updates["enabled"] = request.enabled
        if request.schedule is not None:
            updates["schedule"] = TaskSchedule(request.schedule)

        task_scheduler.update_task(task_id, updates)

        return {
            "success": True,
            "task_id": task_id,
            "message": f"Task updated successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update task: {str(e)}")


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a scheduled task."""
    if not task_scheduler:
        raise HTTPException(status_code=503, detail="Task scheduler not enabled")

    try:
        task_scheduler.delete_task(task_id)

        return {
            "success": True,
            "task_id": task_id,
            "message": "Task deleted successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete task: {str(e)}")


# =====================================================================
# Awareness Loop Endpoints
# =====================================================================

def _check_belief_health() -> bool:
    """Check belief store integrity."""
    if not belief_store:
        return True  # N/A
    try:
        integrity = belief_store.verify_integrity()
        return integrity.get("hash_valid", False) and integrity.get("index_consistent", False)
    except Exception:
        return False


def _check_ledger_health() -> bool:
    """Check ledger integrity."""
    try:
        # Basic check: can we access the ledger directory
        from src.services.identity_ledger import _LEDGER_DIR
        return _LEDGER_DIR.exists()
    except Exception:
        return False


def _get_contrarian_status() -> Dict[str, Any]:
    """Get contrarian sampler status."""
    if not contrarian_sampler:
        return {"enabled": False}

    open_dossiers = contrarian_sampler.get_open_dossiers()

    return {
        "enabled": contrarian_sampler.config.enabled,
        "challenges_today": contrarian_sampler.challenges_today,
        "daily_budget": contrarian_sampler.config.daily_budget,
        "open_dossiers": len(open_dossiers),
        "max_open_dossiers": contrarian_sampler.config.max_open_dossiers,
    }


@app.get("/api/awareness/status")
async def get_awareness_status():
    """Get current awareness loop status."""
    if not settings.AWARENESS_ENABLED:
        return {
            "enabled": False,
            "message": "Awareness loop is disabled"
        }

    if not awareness_loop:
        return {
            "enabled": True,
            "running": False,
            "message": "Awareness loop failed to initialize"
        }

    try:
        # Get metrics summary
        metrics = get_awareness_metrics().get_summary()

        # Get presence state from blackboard
        scalar = await awareness_loop.blackboard.get_presence_scalar()
        meta = await awareness_loop.blackboard.get_meta()

        # Count percept types in buffer for diagnostics
        percept_counts = {}
        text_percept_count = 0
        for p in awareness_loop.percepts:
            percept_counts[p.kind] = percept_counts.get(p.kind, 0) + 1
            if p.kind in ("user", "token") and p.payload.get("text"):
                text_percept_count += 1

        # Compute diagnostics
        cur_vec = awareness_loop.last_presence_vec
        cur_vec_norm = round(float(np.linalg.norm(cur_vec)), 3) if cur_vec is not None else 0.0
        live_anchor = awareness_loop.anchors.get("self_anchor_live")
        origin_anchor = awareness_loop.anchors.get("self_anchor_origin")
        live_norm = round(float(np.linalg.norm(live_anchor)), 3) if live_anchor is not None else 0.0
        origin_norm = round(float(np.linalg.norm(origin_anchor)), 3) if origin_anchor is not None else 0.0

        return {
            "enabled": True,
            "running": awareness_loop.running,
            "session_id": awareness_loop.session_id,
            "tick": awareness_loop.tick_id,
            "mode": awareness_loop.mode,
            "presence": round(scalar, 3),
            "novelty": round(meta.get("novelty", 0.0), 3),
            "sim_self_live": round(meta.get("sim_self_live", 0.0), 3),
            "sim_self_origin": round(meta.get("sim_self_origin", 0.0), 3),
            "coherence_drop": round(meta.get("coherence_drop", 0.0), 3),
            "entropy": round(meta.get("entropy", 0.0), 3),
            "last_note_ts": meta.get("tick", 0),
            "buffer": {
                "total_percepts": len(awareness_loop.percepts),
                "text_percepts": text_percept_count,
                "by_kind": percept_counts,
            },
            "metrics": metrics,
            "meta": {
                **meta,
                "buf_len": int(meta.get("buf_len") or 0),  # Force int, never null
            },
            "diag": {
                "buf_text": int(meta.get("buf_len") or 0),
                "cur_vec_norm": cur_vec_norm,
                "live_norm": live_norm,
                "origin_norm": origin_norm,
                "last_fast_ts": awareness_loop.tick_id,
                "last_slow_ts": awareness_loop.last_slow_tick,
            },
            "beliefs_ok": _check_belief_health(),
            "ledger_ok": _check_ledger_health(),
            "contrarian": _get_contrarian_status(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get awareness status: {str(e)}")


@app.get("/api/awareness/notes")
async def get_awareness_notes(limit: int = 20):
    """Get recent introspection notes."""
    if not settings.AWARENESS_ENABLED:
        raise HTTPException(status_code=503, detail="Awareness loop is disabled")

    if not awareness_loop:
        raise HTTPException(status_code=503, detail="Awareness loop not initialized")

    try:
        notes = await awareness_loop.blackboard.get_introspection_notes(limit=limit)

        return {
            "notes": notes,
            "count": len(notes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get notes: {str(e)}")


@app.post("/api/awareness/pause")
async def pause_awareness():
    """Pause awareness loop (non-destructive, just sets flag)."""
    if not settings.AWARENESS_ENABLED:
        raise HTTPException(status_code=503, detail="Awareness loop is disabled")

    if not awareness_loop:
        raise HTTPException(status_code=503, detail="Awareness loop not initialized")

    awareness_loop.running = False

    return {
        "success": True,
        "message": "Awareness loop paused"
    }


@app.post("/api/awareness/resume")
async def resume_awareness():
    """Resume awareness loop."""
    if not settings.AWARENESS_ENABLED:
        raise HTTPException(status_code=503, detail="Awareness loop is disabled")

    if not awareness_loop:
        raise HTTPException(status_code=503, detail="Awareness loop not initialized")

    awareness_loop.running = True

    return {
        "success": True,
        "message": "Awareness loop resumed"
    }


@app.get("/api/awareness/metrics")
async def get_awareness_metrics_detailed():
    """Get detailed awareness metrics."""
    if not settings.AWARENESS_ENABLED:
        raise HTTPException(status_code=503, detail="Awareness loop is disabled")

    try:
        metrics = get_awareness_metrics().get_all_metrics()

        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    # Ensure data directories exist
    settings.ensure_data_directories()

    # Run server
    print("🚀 Starting AI Experience Memory web interface...")
    print(f"📊 Stats: {raw_store.count_experiences()} experiences, {vector_store.count()} vectors")
    print(f"🤖 LLM: {settings.LLM_MODEL if llm_service else 'Not configured'}")
    print(f"🌐 Server: http://172.239.66.45:8000")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000)
