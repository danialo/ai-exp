# Repository Guidelines

## Project Structure & Module Organization
- `src/memory/` houses storage backends (SQLite raw store, Chroma vector index) and embedding providers.
- `src/pipeline/` orchestrates ingestion, affect-aware lens, and reflection shards; `src/services/` wraps retrieval, LLM, and awareness loops.
- `app.py` starts the FastAPI web UI; `config/settings.py` centralizes environment settings. Runtime artefacts live under `data/` (databases, vector index) and `logs/`.
- Tests reside in `tests/` with mirrors of the module layout; docs and design notes are in `docs/`. Static web assets live in `static/`.

## Build, Test, and Development Commands
- Create an environment with `python3 -m venv venv && source venv/bin/activate`.
- Install dependencies via `pip install -r requirements.txt`; add `-r requirements-test.txt` when running full suites.
- Initialize local stores using `python scripts/init_db.py` (rerun after schema updates).
- Launch the development server with `python app.py`; watch the console for uvicorn startup.
- Run linting and formatting before committing: `ruff check .` and `black src tests scripts`.
- Execute unit tests with `pytest`; filter marked suites with `pytest -m "not slow"`. For a running system sanity check, execute `tests/regression_quick.sh` while the server is up.

## Coding Style & Naming Conventions
- Target Python 3.12 with 4-space indentation and keep lines ≤100 characters (Black/Ruff defaults).
- Modules and functions use `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`. Prefer explicit type hints for public APIs and async boundaries.
- Maintain docstrings for service entry points and API handlers; inline comments only for non-obvious data flows.

## Testing Guidelines
- Follow the `test_*.py` pattern defined in `pytest.ini`; co-locate fixtures in `tests/conftest.py`.
- Use `pytest.mark.slow` and `pytest.mark.load` to gate heavier scenarios; ensure default runs pass without selecting them.
- Integration tests that require openness (e.g., awareness loop) should stub external APIs; avoid hitting live services in CI.

## Commit & Pull Request Guidelines
- Write imperative, concise commit messages (e.g., “Add task execution tracking”); group related changes logically.
- For PRs, include context, testing evidence (`pytest` output or regression script), and reference relevant docs or issues.
- Update `CHANGELOG.md` or docs when user-facing behaviour shifts; attach screenshots/gifs when UI changes affect `static/`.

## Configuration & Operations Notes
- Copy `.env.example` to `.env` and keep secrets local; never commit real keys.
- Generated databases and vector indexes in `data/` are developer-local. Use `scripts/backfill_task_executions.py` cautiously and call out migrations in PR descriptions.
- Monitor `logs/errors/` when investigating failures; rotate large files before pushing.
