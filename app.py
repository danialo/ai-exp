"""FastAPI web interface for AI Experience Memory System."""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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
    llm_service = create_llm_service(
        api_key=api_key,
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        base_url=base_url,
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
    )

# Global session tracking
current_session_id: Optional[str] = None


# Request/Response models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    retrieve_memories: bool = True
    top_k: int = 3


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
