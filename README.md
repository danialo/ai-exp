# AI Experience Memory System

An experience-based memory system with affect-aware retrieval, implementing immutable experience storage, semantic embeddings, and tone-adaptive responses. The system learns from past interactions and adapts its communication style based on the emotional context of retrieved memories.

## Overview

This system implements a complete MVP of an experience memory architecture featuring:

- **Immutable Raw Store**: SQLite-based persistence for experience records
- **Vector Retrieval**: Semantic search using ChromaDB with sentence transformers
- **Affect-Aware Lens**: Valence-based tone adjustment without altering facts
- **Reflection Shards**: Post-response learning observations
- **Web Interface**: FastAPI-based chat UI with memory visualization

## How It Works

1. **User interacts** via web UI or API
2. **System retrieves** semantically similar past experiences using vector search
3. **Lens pass** applies affect-aware tone adjustment based on memory valence
4. **Response generation** uses LLM with memory context and tone styling
5. **Reflection** captures what memories were helpful for future learning
6. **Storage** persists both the interaction and reflection as immutable experiences

## Project Structure

```
ai-exp/
├── src/
│   ├── memory/          # Data models, raw store, embeddings, vector store
│   ├── pipeline/        # Ingestion, lens pass, reflection writer
│   └── services/        # Retrieval service, LLM service
├── tests/               # Unit and integration tests (127 tests)
├── scripts/             # CLI utilities and harness tools
├── config/              # Configuration and settings
├── data/                # Database and vector index storage
├── static/              # Web UI assets
└── docs/                # Architecture and build documentation
```

## Quick Start

### 1. Setup

Create virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_key_here
```

### 3. Initialize Database

```bash
python scripts/init_db.py
```

### 4. Run Web Interface

```bash
python app.py
```

The web interface will be available at `http://localhost:8000` (or your VPS IP).

### 5. Test the System

```bash
# Run all tests
pytest

# Test a specific component
pytest tests/test_lens.py -v

# Run CLI chat demo
python scripts/chat.py
```

## Usage Examples

### Web API

**Chat with memory retrieval:**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is machine learning?",
    "retrieve_memories": true,
    "top_k": 3
  }'
```

**Get system statistics:**

```bash
curl http://localhost:8000/api/stats
```

### Python API

```python
from config.settings import settings
from src.memory.raw_store import create_raw_store
from src.memory.vector_store import create_vector_store
from src.memory.embedding import create_embedding_provider
from src.services.retrieval import create_retrieval_service
from src.services.llm import create_llm_service
from src.pipeline.lens import create_experience_lens

# Initialize components
raw_store = create_raw_store(settings.RAW_STORE_DB_PATH)
vector_store = create_vector_store(settings.VECTOR_INDEX_PATH)
embedding_provider = create_embedding_provider(settings.EMBEDDING_MODEL)
retrieval_service = create_retrieval_service(
    raw_store, vector_store, embedding_provider
)
llm_service = create_llm_service(api_key=settings.OPENAI_API_KEY)
lens = create_experience_lens(llm_service, retrieval_service)

# Process a prompt with affect-aware styling
result = lens.process(prompt="How do Python imports work?")

print(f"Response: {result.augmented_response}")
print(f"Blended valence: {result.blended_valence}")
print(f"Citations: {result.citations}")
```

## Development

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/test_lens.py -v
```

### Format Code

```bash
black src tests
ruff check src tests
```

### Type Checking

```bash
mypy src
```

## Configuration

All configuration is managed through environment variables in `.env`. Key settings:

### Database & Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `RAW_STORE_DB_PATH` | `data/raw_store.db` | SQLite database path for raw experiences |
| `VECTOR_INDEX_PATH` | `data/vector_index/` | ChromaDB vector index directory |

### Embedding & LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence transformer model for embeddings |
| `OPENAI_API_KEY` | _required_ | OpenAI API key for LLM chat responses |
| `LLM_MODEL` | `gpt-3.5-turbo` | OpenAI model to use |
| `LLM_TEMPERATURE` | `0.7` | Temperature for LLM generation (0-2) |
| `LLM_MAX_TOKENS` | `500` | Maximum tokens in LLM response |

### Retrieval Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `TOP_K_RETRIEVAL` | `5` | Number of similar experiences to retrieve |
| `SEMANTIC_WEIGHT` | `0.8` | Weight for semantic similarity in ranking |
| `RECENCY_WEIGHT` | `0.2` | Weight for recency in ranking |

**Note**: `SEMANTIC_WEIGHT` + `RECENCY_WEIGHT` should equal 1.0

### Affect Blending Weights

| Variable | Default | Description |
|----------|---------|-------------|
| `AFFECT_WEIGHTS` | `0.5,0.3,0.2` | Comma-separated weights for (user, memory, self) affect blending |

These weights control how affect/valence is calculated from different sources. Must sum to 1.0.

## Architecture

### Data Flow

```
User Input
    ↓
[Retrieval Service] → Semantic search in vector index
    ↓
[Experience Lens] → Apply affect-aware tone adjustment
    ↓
[LLM Service] → Generate response with memory context
    ↓
[Ingestion Pipeline] → Store interaction as experience
    ↓
[Reflection Writer] → Capture learning observation
    ↓
[Vector Store] → Embed and index for future retrieval
```

### Key Components

- **RawStore** (`src/memory/raw_store.py`): Immutable SQLite storage for experiences
- **VectorStore** (`src/memory/vector_store.py`): ChromaDB wrapper for semantic search
- **EmbeddingProvider** (`src/memory/embedding.py`): Sentence transformer embeddings
- **RetrievalService** (`src/services/retrieval.py`): Recency-biased similarity search
- **LLMService** (`src/services/llm.py`): OpenAI chat completion wrapper
- **ExperienceLens** (`src/pipeline/lens.py`): Affect-aware tone adjustment
- **ReflectionWriter** (`src/pipeline/reflection.py`): Post-response learning observations
- **IngestionPipeline** (`src/pipeline/ingest.py`): Process and store interactions

## MVP Status

✅ **Completed Stages (10/10)**

- [x] Stage 0: Environment & scaffolding
- [x] Stage 1: Data contracts & models
- [x] Stage 2: Raw store persistence
- [x] Stage 3: Embedding & vector index
- [x] Stage 4: Ingestion pipeline
- [x] Stage 5: Retrieval service
- [x] Stage 6: Experience lens pass
- [x] Stage 7: Reflection writer
- [x] Stage 8: CLI harness
- [x] Stage 9: Tests & guardrails (127 tests)
- [x] Stage 10: Documentation
- [x] **Bonus**: Web interface with FastAPI

See `docs/mvp_build_plan.md` for detailed build plan.

## Roadmap & Future Features

### Out of Scope (for future iterations)

The following features are documented in `docs/experience_schema.md` but not implemented in this MVP:

- **Uncertainty Re-ranking**: Confidence-based result filtering and ordering
- **Full Experience Lattice**: Inference and reconciliation type experiences
- **Stance Builder**: Multi-experience synthesis for complex queries
- **Temporal Profiles**: Time-based embedding for temporal reasoning
- **Causal Profiles**: Cause-effect relationship embeddings
- **Multi-modal Content**: Image, audio, and structured data support
- **Cryptographic Signatures**: Content verification and provenance chains
- **Advanced Affect Models**: Multi-dimensional emotion tracking beyond VAD

### Potential Enhancements

- Integration with local LLMs (Ollama, LlamaCpp)
- Advanced affect detection using sentiment analysis models
- Batch import/export of experiences
- Query filtering by date range, affect, or confidence
- Experience visualization and analytics dashboard
- Multi-user support with privacy controls
- Vector index optimization (HNSW, IVF)
- Streaming responses for long-running queries

## Documentation

- `docs/experience_schema.md` - Complete schema and architecture specification
- `docs/mvp_build_plan.md` - Stage-by-stage build plan and validation criteria
- `README.md` - This file: setup, usage, and operations guide

## Troubleshooting

### "No module named 'src'"

Make sure you're running from the project root directory and your virtual environment is activated.

### "OPENAI_API_KEY not configured"

The system will work without an API key but will return mock responses. Add your key to `.env`:

```bash
OPENAI_API_KEY=sk-your-key-here
```

### Vector store initialization errors

Delete and recreate the vector index:

```bash
rm -rf data/vector_index/
python scripts/init_db.py
```

### Database locked errors

Only one process should write to the database at a time. Stop the web server before running CLI scripts that write to the database.

## Contributing

This is an experimental MVP. Contributions welcome via issues and pull requests.

## License

MIT
