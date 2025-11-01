"""Pytest fixtures for Phase 5 test suite."""

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import AsyncGenerator, Generator
import pytest
import redis.asyncio as aioredis
from freezegun import freeze_time

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Create temporary data directory for tests."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def belief_store_dir(temp_data_dir: Path) -> Path:
    """Create belief store directory."""
    beliefs_dir = temp_data_dir / "beliefs"
    beliefs_dir.mkdir(parents=True, exist_ok=True)
    return temp_data_dir


@pytest.fixture
def ledger_dir(temp_data_dir: Path) -> Path:
    """Create ledger directory."""
    ledger_dir = temp_data_dir / "identity"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    return temp_data_dir


@pytest.fixture
async def redis_client() -> AsyncGenerator[aioredis.Redis, None]:
    """Create Redis client for tests."""
    client = await aioredis.from_url(
        "redis://localhost:6379",
        encoding="utf-8",
        decode_responses=True
    )

    # Clean test keys
    test_prefix = "test:*"
    async for key in client.scan_iter(match=test_prefix):
        await client.delete(key)

    yield client

    # Cleanup
    async for key in client.scan_iter(match=test_prefix):
        await client.delete(key)
    await client.close()


@pytest.fixture
def frozen_time():
    """Freeze time for deterministic tests."""
    frozen = freeze_time("2025-10-31 12:00:00")
    frozen.start()
    yield frozen
    frozen.stop()


@pytest.fixture
def seeded_random():
    """Seed random for reproducible tests."""
    import random
    import numpy as np

    random.seed(42)
    np.random.seed(42)
    yield


@pytest.fixture(autouse=True)
def cleanup_test_redis():
    """Cleanup Redis test keys after each test."""
    yield
    # Post-test cleanup happens in redis_client fixture


# Timeout configuration
def pytest_configure(config):
    """Configure pytest with timeouts."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "load: marks tests as load tests (deselect with '-m \"not load\"')"
    )
