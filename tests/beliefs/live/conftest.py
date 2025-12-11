"""
Fixtures for live end-to-end tests.

Uses real LLM client and database - no mocks.
"""

import os
import pytest
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from dataclasses import dataclass
from typing import Optional, Dict, Any

from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy.pool import NullPool

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class MockExperience:
    """Experience object for testing."""
    id: str
    type: str
    content: str
    interaction_mode: str
    conversation_id: Optional[str]
    affect: Dict[str, float]
    created_at: datetime
    session_id: Optional[str] = None

    def get_text(self) -> str:
        return self.content


@pytest.fixture(scope="session")
def live_llm_client():
    """
    Create a real LLM client using the configured API.

    Skips tests if no API key is available.
    """
    # Try to get API key from environment or settings
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        # Try loading from settings
        try:
            import sys
            sys.path.insert(0, str(PROJECT_ROOT))
            from config.settings import settings
            api_key = getattr(settings, 'OPENAI_API_KEY', None)
        except Exception:
            pass

    if not api_key:
        pytest.skip("No OPENAI_API_KEY available - skipping live tests")

    from src.services.llm import LLMService

    # Use a smaller, cheaper model for testing
    return LLMService(
        api_key=api_key,
        model="gpt-4o-mini",  # Cheaper for testing
        temperature=0,  # Deterministic
        max_tokens=500,
    )


@pytest.fixture(scope="session")
def live_db_session():
    """
    Create a database session for belief tables.

    Uses a separate test database to avoid polluting production.
    """
    import src.memory.models.belief_node  # noqa: F401
    import src.memory.models.belief_occurrence  # noqa: F401
    import src.memory.models.tentative_link  # noqa: F401
    import src.memory.models.conflict_edge  # noqa: F401
    import src.memory.models.stream_assignment  # noqa: F401

    db_path = PROJECT_ROOT / "data" / "test_beliefs_live.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


@pytest.fixture
def sample_self_statement():
    """A simple self-referential statement."""
    return "I am patient and I tend to overthink things."


@pytest.fixture
def complex_self_statement():
    """A statement with multiple beliefs and temporal markers."""
    return "I've always loved hiking, but lately I've been feeling more anxious about outdoor activities."


@pytest.fixture
def journaling_experience():
    """High-weight journaling experience."""
    return MockExperience(
        id=str(uuid4()),
        type="self_definition",
        content="I've always been curious about how things work. I find myself constantly asking questions.",
        interaction_mode="journaling",
        conversation_id=f"conv_{uuid4().hex[:8]}",
        affect={"arousal": 0.2, "valence": 0.6},
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def normal_chat_experience():
    """Standard chat experience."""
    return MockExperience(
        id=str(uuid4()),
        type="self_definition",
        content="I think I'm pretty creative, but I can be impatient sometimes.",
        interaction_mode="normal_chat",
        conversation_id=f"conv_{uuid4().hex[:8]}",
        affect={"arousal": 0.4, "valence": 0.5},
        created_at=datetime.now(timezone.utc),
    )
