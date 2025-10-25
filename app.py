"""FastAPI web interface for AI Experience Memory System."""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from pathlib import Path
import uvicorn

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
from src.memory.models import ExperienceType
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
    )


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
            })
    return {"models": models}


@app.get("/api/persona/info")
async def get_persona_info():
    """Get information about the persona's current state and capabilities."""
    if not persona_service:
        raise HTTPException(status_code=503, detail="Persona mode not enabled")

    return persona_service.get_persona_info()


@app.post("/api/persona/chat")
async def persona_chat(request: ChatRequest):
    """Chat with the persona directly - includes memory storage for continuity."""
    if not persona_service:
        raise HTTPException(status_code=503, detail="Persona mode not enabled")

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # If model override specified, create temporary persona service with that model
        active_persona_service = persona_service
        if request.model:
            _, persona_llm = create_llm_for_model(request.model)
            active_persona_service = PersonaService(
                llm_service=persona_llm,
                persona_space_path=settings.PERSONA_SPACE_PATH,
                retrieval_service=retrieval_service,
                enable_anti_metatalk=settings.ANTI_METATALK_ENABLED,
                logit_bias_strength=settings.LOGIT_BIAS_STRENGTH,
                auto_rewrite=settings.AUTO_REWRITE_METATALK,
            )

        # Generate persona response with emotional co-analysis and memory retrieval
        response_text, reconciliation = active_persona_service.generate_response(
            user_message=request.message,
            retrieve_memories=request.retrieve_memories,
            top_k=request.top_k,
            conversation_history=request.conversation_history
        )

        # Store the interaction in memory so persona can learn from it
        # Use reconciled emotional state if available, otherwise neutral
        valence = 0.0
        if reconciliation and reconciliation.get("reconciled_state"):
            # Try to extract valence from reconciled state description
            # For now, use neutral - could be enhanced to parse emotional state
            valence = 0.0

        interaction = InteractionPayload(
            prompt=request.message,
            response=response_text,
            valence=valence,
        )
        result = ingestion_pipeline.ingest_interaction(interaction)

        logger.info(f"Stored persona interaction: {result.experience_id}")

        # Return response with reconciliation data
        return {
            "response": response_text,
            "reconciliation": reconciliation,
            "internal_assessment": reconciliation.get("internal_assessment") if reconciliation else None,
            "external_assessment": reconciliation.get("external_assessment") if reconciliation else None,
            "reconciled_state": reconciliation.get("reconciled_state") if reconciliation else None,
            "experience_id": result.experience_id,  # Include so user knows it was stored
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
