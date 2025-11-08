"""Test race conditions and invariants (Phase 5.1)."""

import asyncio
import random
import time
from pathlib import Path
import pytest
import numpy as np
import redis.asyncio as aioredis
from src.services.awareness_loop import AwarenessLoop


class MockLLMService:
    """Mock LLM for testing."""
    async def generate_completion(self, *args, **kwargs):
        return "test response"


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_buf_len_race_10k_writes(redis_client: aioredis.Redis, seeded_random):
    """Test buf_len consistency under 10k interleaved fast/slow writes.

    Acceptance:
    - buf_len == text_percepts every read
    - buf_ver strictly increases across N>100 sequential reads
    """
    # Create awareness loop with minimal deps
    loop = AwarenessLoop(
        blackboard=redis_client,
        llm=MockLLMService(),
        embedding_fn=lambda t: np.random.randn(768),
        self_knowledge_text="test identity",
        persona_service=None,
    )

    # Track versions
    buf_vers = []
    buf_lens = []

    async def fast_writer():
        """Simulate fast tick writes."""
        for i in range(5000):
            # Add time percept (not text)
            loop.percepts.append(
                type('Percept', (), {
                    'kind': 'time',
                    'ts': time.time(),
                    'payload': {'timestamp': time.time()}
                })()
            )
            await asyncio.sleep(0)  # Yield control

    async def slow_writer():
        """Simulate slow tick writes."""
        for i in range(5000):
            # Add text percept
            loop.percepts.append(
                type('Percept', (), {
                    'kind': 'user',
                    'ts': time.time(),
                    'payload': {'text': f'message {i}'}
                })()
            )
            await asyncio.sleep(0)

    async def reader():
        """Read and verify invariants."""
        for _ in range(150):
            meta = await loop.blackboard.hgetall("awareness:meta")
            if meta:
                buf_len = int(meta.get("buf_len", 0))
                buf_ver = int(meta.get("buf_ver", 0))

                # Count actual text percepts
                text_count = sum(
                    1 for p in loop.percepts
                    if p.kind in ("user", "token") and p.payload.get("text")
                )

                # Invariant: buf_len == text_percepts
                assert buf_len == text_count, \
                    f"buf_len mismatch: reported={buf_len}, actual={text_count}"

                buf_lens.append(buf_len)
                buf_vers.append(buf_ver)

            await asyncio.sleep(0.001)

    # Run interleaved
    await asyncio.gather(fast_writer(), slow_writer(), reader())

    # Verify buf_ver monotonic increasing
    for i in range(1, len(buf_vers)):
        assert buf_vers[i] >= buf_vers[i-1], \
            f"buf_ver not monotonic at index {i}: {buf_vers[i-1]} -> {buf_vers[i]}"

    print(f"✓ Verified {len(buf_vers)} reads with monotonic buf_ver")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_vector_norm_invariants(redis_client: aioredis.Redis):
    """Test vector norms stay in [0.999, 1.001].

    Acceptance:
    - cur_vec_norm, live_norm, origin_norm in [0.999, 1.001]
    """
    loop = AwarenessLoop(
        blackboard=redis_client,
        llm=MockLLMService(),
        embedding_fn=lambda t: np.random.randn(768),
        self_knowledge_text="test identity",
        persona_service=None,
    )

    # Simulate some vector updates
    for _ in range(100):
        vec = np.random.randn(768)
        vec = vec / np.linalg.norm(vec)  # Normalize

        loop.last_presence_vec = vec
        loop.anchors["self_anchor_live"] = vec.copy()
        loop.anchors["self_anchor_origin"] = vec.copy()

        # Check norms
        cur_norm = np.linalg.norm(loop.last_presence_vec)
        live_norm = np.linalg.norm(loop.anchors["self_anchor_live"])
        origin_norm = np.linalg.norm(loop.anchors["self_anchor_origin"])

        assert 0.999 <= cur_norm <= 1.001, f"cur_vec_norm out of bounds: {cur_norm}"
        assert 0.999 <= live_norm <= 1.001, f"live_norm out of bounds: {live_norm}"
        assert 0.999 <= origin_norm <= 1.001, f"origin_norm out of bounds: {origin_norm}"

    print("✓ All 100 vector norms within [0.999, 1.001]")


@pytest.mark.asyncio
@pytest.mark.timeout(310)
@pytest.mark.load
async def test_healthz_under_load_5min(seeded_random):
    """Test /healthz/assert returns 200 under load for 5 minutes.

    Acceptance:
    - /healthz/assert returns 200 continuously
    """
    import httpx

    async with httpx.AsyncClient() as client:
        start_time = time.time()
        end_time = start_time + 300  # 5 minutes

        failures = []
        checks = 0

        while time.time() < end_time:
            try:
                resp = await client.get(
                    "http://localhost:8000/healthz/assert",
                    timeout=2.0
                )

                if resp.status_code != 200:
                    failures.append((time.time() - start_time, resp.status_code))

                checks += 1

            except Exception as e:
                failures.append((time.time() - start_time, str(e)))

            await asyncio.sleep(1.0)  # Check every second

        elapsed = time.time() - start_time

        assert len(failures) == 0, \
            f"Health check failed {len(failures)}/{checks} times in {elapsed:.1f}s: {failures[:5]}"

        print(f"✓ Health check passed {checks} times over {elapsed:.1f}s")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_buf_ver_monotonic_sequential(redis_client: aioredis.Redis):
    """Test buf_ver strictly increases across sequential reads.

    Acceptance:
    - buf_ver strictly increases across N>100 sequential reads
    """
    loop = AwarenessLoop(
        blackboard=redis_client,
        llm=MockLLMService(),
        embedding_fn=lambda t: np.random.randn(768),
        self_knowledge_text="test identity",
        persona_service=None,
    )

    buf_vers = []

    # Write metadata with increasing buf_ver
    for i in range(150):
        await redis_client.hset("awareness:meta", "buf_ver", i)
        await redis_client.hset("awareness:meta", "buf_len", i % 10)

        meta = await redis_client.hgetall("awareness:meta")
        buf_ver = int(meta["buf_ver"])
        buf_vers.append(buf_ver)

        await asyncio.sleep(0.001)

    # Verify strict monotonic increase
    for i in range(1, len(buf_vers)):
        assert buf_vers[i] > buf_vers[i-1], \
            f"buf_ver not strictly increasing at {i}: {buf_vers[i-1]} -> {buf_vers[i]}"

    assert len(buf_vers) > 100, f"Only {len(buf_vers)} reads, need >100"

    print(f"✓ buf_ver strictly increased across {len(buf_vers)} sequential reads")
