"""Test telemetry metrics (Phase 5.7)."""

import pytest
import httpx
from typing import Dict, Any


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_contrarian_counters_monotonic():
    """Test contrarian_challenges_total counters are monotonic.

    Acceptance:
    - Counters only increase
    """
    async with httpx.AsyncClient() as client:
        # Get initial status
        resp1 = await client.get("http://localhost:8000/api/awareness/status")
        assert resp1.status_code == 200
        data1 = resp1.json()

        challenges_before = data1["contrarian"]["challenges_today"]

        # Trigger contrarian run (may or may not execute)
        await client.post("http://localhost:8000/api/persona/contrarian/run")

        # Get status again
        resp2 = await client.get("http://localhost:8000/api/awareness/status")
        data2 = resp2.json()

        challenges_after = data2["contrarian"]["challenges_today"]

        # Should never decrease
        assert challenges_after >= challenges_before, \
            f"Challenges counter decreased: {challenges_before} -> {challenges_after}"

    print(f"✓ Contrarian counters monotonic: {challenges_before} -> {challenges_after}")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_belief_confidence_gauge_range():
    """Test belief_confidence gauges stay in [0, 1].

    Acceptance:
    - All belief confidences in [0.0, 1.0]
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8000/api/persona/beliefs")
        assert resp.status_code == 200

        data = resp.json()
        beliefs = data["beliefs"]

        for belief_id, belief in beliefs.items():
            confidence = belief["confidence"]
            assert 0.0 <= confidence <= 1.0, \
                f"Belief {belief_id} confidence {confidence} out of [0, 1] range"

    print(f"✓ Belief confidence gauges: all {len(beliefs)} in [0, 1]")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_buf_len_gauge_non_negative():
    """Test buf_len gauge is always non-negative.

    Acceptance:
    - buf_len >= 0
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8000/api/awareness/status")
        assert resp.status_code == 200

        data = resp.json()
        buf_len = data.get("buf_len", 0)

        assert buf_len >= 0, f"buf_len should be non-negative, got {buf_len}"

    print(f"✓ buf_len gauge non-negative: {buf_len}")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_novelty_gauge_range():
    """Test novelty gauge stays in [0, 1].

    Acceptance:
    - novelty in [0.0, 1.0]
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8000/api/awareness/status")
        assert resp.status_code == 200

        data = resp.json()
        novelty = data.get("novelty", 0.0)

        assert 0.0 <= novelty <= 1.0, \
            f"novelty should be in [0, 1], got {novelty}"

    print(f"✓ novelty gauge in range: {novelty}")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_dossier_count_gauge():
    """Test open_dossiers gauge is accurate.

    Acceptance:
    - open_dossiers count matches actual open dossiers
    """
    async with httpx.AsyncClient() as client:
        # Get status
        resp = await client.get("http://localhost:8000/api/awareness/status")
        data = resp.json()
        open_count_reported = data["contrarian"]["open_dossiers"]

        # Get actual dossiers
        resp = await client.get("http://localhost:8000/api/persona/contrarian/dossiers")
        dossiers_data = resp.json()
        dossiers = dossiers_data["dossiers"]

        open_count_actual = len([d for d in dossiers if d["status"] == "open"])

        assert open_count_reported == open_count_actual, \
            f"Reported open dossiers {open_count_reported} != actual {open_count_actual}"

    print(f"✓ Dossier count gauge accurate: {open_count_reported}")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_budget_gauge_consistency():
    """Test daily_budget gauge is consistent.

    Acceptance:
    - challenges_today <= daily_budget
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8000/api/awareness/status")
        data = resp.json()

        challenges_today = data["contrarian"]["challenges_today"]
        daily_budget = data["contrarian"]["daily_budget"]

        assert challenges_today <= daily_budget, \
            f"challenges_today {challenges_today} > daily_budget {daily_budget}"

    print(f"✓ Budget gauge consistent: {challenges_today}/{daily_budget}")


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_metrics_stability_over_time():
    """Test metrics remain stable over multiple samples.

    Acceptance:
    - Sample metrics 10 times, verify no crashes
    - Monotonic counters never decrease
    - Gauges stay in valid ranges
    """
    import asyncio

    async with httpx.AsyncClient() as client:
        samples = []

        for i in range(10):
            resp = await client.get("http://localhost:8000/api/awareness/status")
            assert resp.status_code == 200

            data = resp.json()
            samples.append({
                "buf_len": data.get("buf_len", 0),
                "novelty": data.get("novelty", 0.0),
                "challenges": data["contrarian"]["challenges_today"],
            })

            await asyncio.sleep(0.5)

        # Verify monotonic counters
        for i in range(1, len(samples)):
            prev_challenges = samples[i-1]["challenges"]
            curr_challenges = samples[i]["challenges"]

            assert curr_challenges >= prev_challenges, \
                f"Challenges decreased: sample {i-1}={prev_challenges} -> {i}={curr_challenges}"

        # Verify gauge ranges
        for i, sample in enumerate(samples):
            assert sample["buf_len"] >= 0, f"Sample {i}: buf_len negative"
            assert 0.0 <= sample["novelty"] <= 1.0, f"Sample {i}: novelty out of range"

    print(f"✓ Metrics stable over {len(samples)} samples")
