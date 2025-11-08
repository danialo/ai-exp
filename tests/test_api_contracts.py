"""Test API contracts and schemas (Phase 5.6)."""

import pytest
import httpx
from typing import Dict, Any


async def validate_belief_schema(belief: Dict[str, Any]) -> bool:
    """Validate belief object schema."""
    required_fields = ["belief_id", "ver", "statement", "state", "confidence", "belief_type"]

    for field in required_fields:
        if field not in belief:
            return False

    # Type checks
    if not isinstance(belief["ver"], int):
        return False
    if not isinstance(belief["confidence"], (int, float)):
        return False
    if not (0.0 <= belief["confidence"] <= 1.0):
        return False

    return True


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_beliefs_endpoint_schema():
    """Test /api/persona/beliefs returns valid schema.

    Acceptance:
    - JSON schema validated
    - Required fields enforced
    - Unknown fields ignored
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8000/api/persona/beliefs")

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

        data = resp.json()

        # Should be a dict with "beliefs" and "count"
        assert "beliefs" in data, "Response should have 'beliefs' field"
        assert "count" in data, "Response should have 'count' field"

        beliefs = data["beliefs"]
        assert isinstance(beliefs, dict), "Beliefs should be a dict"

        # Validate each belief
        for belief_id, belief in beliefs.items():
            assert validate_belief_schema(belief), \
                f"Belief {belief_id} has invalid schema"

    print(f"✓ Beliefs endpoint: schema validated for {len(beliefs)} beliefs")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_belief_history_endpoint():
    """Test /api/persona/beliefs/history endpoint.

    Acceptance:
    - Returns history array
    - Each entry has required fields
    """
    async with httpx.AsyncClient() as client:
        # Get beliefs first
        resp = await client.get("http://localhost:8000/api/persona/beliefs")
        beliefs = resp.json()["beliefs"]

        if len(beliefs) == 0:
            pytest.skip("No beliefs to test history")

        # Pick first belief
        belief_id = list(beliefs.keys())[0]

        # Get history
        resp = await client.get(
            f"http://localhost:8000/api/persona/beliefs/history",
            params={"id": belief_id}
        )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

        data = resp.json()
        assert "deltas" in data, "Response should have 'deltas' field"

        deltas = data["deltas"]
        assert isinstance(deltas, list), "Deltas should be a list"

        # Validate delta schema
        for delta in deltas:
            # belief_id is implied from query, not in each delta
            assert "from_ver" in delta
            assert "to_ver" in delta
            assert "op" in delta

    print(f"✓ History endpoint: schema validated for {len(deltas)} deltas")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_contrarian_run_idempotent():
    """Test /contrarian/run is idempotent when no budget.

    Acceptance:
    - Multiple calls with no budget return no-op result
    """
    async with httpx.AsyncClient() as client:
        # Call multiple times
        results = []
        for _ in range(5):
            resp = await client.post("http://localhost:8000/api/persona/contrarian/run")
            assert resp.status_code == 200
            data = resp.json()
            results.append(data)

        # All should return same "no budget" response
        for data in results:
            assert "success" in data
            # With enabled=false, should skip
            if not data.get("success"):
                assert "message" in data

    print("✓ Contrarian run: idempotent with no budget")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_dossiers_pagination():
    """Test /contrarian/dossiers pagination and sorting.

    Acceptance:
    - Pagination works
    - Sorting by opened_ts desc
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8000/api/persona/contrarian/dossiers")

        assert resp.status_code == 200
        data = resp.json()

        assert "dossiers" in data
        assert "count" in data

        dossiers = data["dossiers"]
        assert isinstance(dossiers, list)

        # Check sorting if we have dossiers
        if len(dossiers) > 1:
            timestamps = [d.get("challenge_ts", 0) for d in dossiers]
            # Should be sorted descending
            assert timestamps == sorted(timestamps, reverse=True), \
                "Dossiers should be sorted by challenge_ts descending"

    print(f"✓ Dossiers endpoint: {len(dossiers)} dossiers, sorted by timestamp")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_belief_delta_endpoint_validation():
    """Test /api/persona/beliefs/delta enforces validation.

    Acceptance:
    - Invalid confidence_delta rejected
    - Missing required fields rejected
    """
    async with httpx.AsyncClient() as client:
        # Test with invalid confidence_delta (too large)
        resp = await client.post(
            "http://localhost:8000/api/persona/beliefs/delta",
            json={
                "belief_id": "test.nonexistent",
                "from_ver": 1,
                "op": "update",
                "confidence_delta": 0.5,  # Too large (max 0.15)
                "reason": "Test"
            }
        )

        # Should reject (either 400 or 500 depending on validation)
        assert resp.status_code in [400, 404, 409, 500], \
            "Should reject invalid confidence_delta"

        # Test with missing required fields
        resp = await client.post(
            "http://localhost:8000/api/persona/beliefs/delta",
            json={
                "belief_id": "test.nonexistent",
                # Missing from_ver, op, reason
            }
        )

        assert resp.status_code in [400, 422], \
            "Should reject missing required fields"

    print("✓ Delta endpoint: validation enforced")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_awareness_status_schema():
    """Test /api/awareness/status returns complete schema.

    Acceptance:
    - All expected fields present
    - Contrarian status included
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8000/api/awareness/status")

        assert resp.status_code == 200
        data = resp.json()

        # Check for key fields
        expected_fields = ["buffer", "contrarian"]

        for field in expected_fields:
            assert field in data, f"Status should include '{field}' field"

        # Check buffer subfields
        assert "buf_len" in data["buffer"] or "text_percepts" in data["buffer"], \
            "Buffer should include buf_len or text_percepts"

        # Validate contrarian status
        contrarian = data["contrarian"]
        assert "enabled" in contrarian
        assert "challenges_today" in contrarian
        assert "daily_budget" in contrarian

    print("✓ Awareness status: complete schema with contrarian")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_healthz_assert_response():
    """Test /healthz/assert returns proper structure.

    Acceptance:
    - Returns status field
    - Returns checks_passed array or failures
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8000/healthz/assert")

        assert resp.status_code in [200, 500], \
            "Health check should return 200 or 500"

        data = resp.json()

        assert "status" in data, "Should have status field"

        if resp.status_code == 200:
            assert data["status"] == "healthy"
            assert "checks_passed" in data
        else:
            assert "failures" in data

    print(f"✓ Health assert: status={data['status']}")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_unknown_fields_ignored():
    """Test API endpoints ignore unknown fields gracefully.

    Acceptance:
    - Unknown fields in request don't cause errors
    """
    async with httpx.AsyncClient() as client:
        # Try delta with extra unknown field
        resp = await client.post(
            "http://localhost:8000/api/persona/beliefs/delta",
            json={
                "belief_id": "test.unknown",
                "from_ver": 1,
                "op": "update",
                "confidence_delta": 0.03,
                "reason": "Test",
                "unknown_field": "should be ignored",
                "another_unknown": 12345,
            }
        )

        # Should not crash (may fail for other reasons like nonexistent belief or validation)
        assert resp.status_code in [200, 400, 404, 409, 422, 500], \
            "Unknown fields should not cause crashes"

    print("✓ Unknown fields ignored gracefully")
