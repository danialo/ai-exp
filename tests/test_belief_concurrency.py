"""Test belief concurrency and optimistic locking (Phase 5.3)."""

import asyncio
import pytest
import json
from pathlib import Path
from src.services.belief_store import BeliefStore, BeliefState, DeltaOp


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_concurrent_delta_409_conflict(belief_store_dir: Path):
    """Test two writers with same from_ver produce one 409.

    Acceptance:
    - One writer gets 200, other gets conflict (returns False)
    - After retry with updated from_ver, both apply
    - index.json matches current.json versions
    """
    store = BeliefStore(belief_store_dir)

    # Create initial belief
    belief_id = "test.concurrent-write"
    store.create_belief(
        belief_id=belief_id,
        statement="Initial statement",
        state=BeliefState.TENTATIVE,
        confidence=0.5,
        evidence_refs=["exp1"],
        belief_type="experiential",
        immutable=False,
        rationale="Test belief",
        updated_by="test",
    )

    # Get current version
    beliefs = store.get_current([belief_id])
    current_ver = beliefs[belief_id].ver
    assert current_ver == 1

    # Simulate two concurrent writers
    results = []

    async def writer1():
        """First writer attempts delta."""
        success = store.apply_delta(
            belief_id=belief_id,
            from_ver=1,
            op=DeltaOp.UPDATE,
            confidence_delta=0.05,
            updated_by="writer1",
            reason="Writer 1 update",
        )
        results.append(("writer1", success))

    async def writer2():
        """Second writer attempts delta with same from_ver."""
        # Small delay to ensure writer1 goes first
        await asyncio.sleep(0.01)
        success = store.apply_delta(
            belief_id=belief_id,
            from_ver=1,
            op=DeltaOp.UPDATE,
            confidence_delta=0.03,
            updated_by="writer2",
            reason="Writer 2 update",
        )
        results.append(("writer2", success))

    # Run both concurrently
    await asyncio.gather(writer1(), writer2())

    # Verify results
    assert len(results) == 2
    successes = [r for (writer, r) in results if r]
    failures = [r for (writer, r) in results if not r]

    assert len(successes) == 1, "Exactly one writer should succeed"
    assert len(failures) == 1, "Exactly one writer should fail (conflict)"

    # Get current version (should be 2 now)
    beliefs = store.get_current([belief_id])
    assert beliefs[belief_id].ver == 2

    # Failed writer retries with updated from_ver
    success = store.apply_delta(
        belief_id=belief_id,
        from_ver=2,  # Updated from_ver
        op=DeltaOp.UPDATE,
        confidence_delta=0.03,
        updated_by="writer2_retry",
        reason="Writer 2 retry",
    )
    assert success, "Retry with correct from_ver should succeed"

    # Verify final version
    beliefs = store.get_current([belief_id])
    assert beliefs[belief_id].ver == 3

    # Verify index.json matches current.json
    with open(belief_store_dir / "beliefs" / "current.json") as f:
        current = json.load(f)
    with open(belief_store_dir / "beliefs" / "index.json") as f:
        index = json.load(f)

    assert index[belief_id] == current[belief_id]["ver"], \
        "index.json version should match current.json"

    print(f"✓ Optimistic locking: 1 conflict, retry succeeded, versions consistent")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_hash_integrity_verification(belief_store_dir: Path):
    """Test hash integrity by recomputing SHA-256.

    Acceptance:
    - Recompute sha256(json_without_hash) for all current items
    - All hashes match
    """
    store = BeliefStore(belief_store_dir)

    # Create multiple beliefs
    for i in range(5):
        store.create_belief(
            belief_id=f"test.hash-check-{i}",
            statement=f"Statement {i}",
            state=BeliefState.ASSERTED,
            confidence=0.5 + i * 0.1,
            evidence_refs=[f"exp{i}"],
            belief_type="experiential",
            immutable=False,
            rationale=f"Test belief {i}",
            updated_by="test",
        )

    # Verify integrity
    results = store.verify_integrity()

    assert results["hash_valid"], "All hashes should be valid"
    assert results["index_consistent"], "Index should be consistent"

    # Manually recompute hashes
    beliefs = store.get_current()
    for belief_id, belief in beliefs.items():
        computed_hash = belief.compute_hash()
        assert belief.hash == computed_hash, \
            f"Hash mismatch for {belief_id}: stored={belief.hash}, computed={computed_hash}"

    print(f"✓ Hash integrity verified for {len(beliefs)} beliefs")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_concurrent_create_race(belief_store_dir: Path):
    """Test concurrent creation of same belief.

    Acceptance:
    - First create succeeds, second returns False
    """
    store = BeliefStore(belief_store_dir)

    belief_id = "test.concurrent-create"
    results = []

    async def creator1():
        """First creator."""
        success = store.create_belief(
            belief_id=belief_id,
            statement="Statement 1",
            state=BeliefState.TENTATIVE,
            confidence=0.5,
            evidence_refs=["exp1"],
            belief_type="experiential",
            immutable=False,
            rationale="Creator 1",
            updated_by="creator1",
        )
        results.append(("creator1", success))

    async def creator2():
        """Second creator with slight delay."""
        await asyncio.sleep(0.01)
        success = store.create_belief(
            belief_id=belief_id,
            statement="Statement 2",
            state=BeliefState.TENTATIVE,
            confidence=0.6,
            evidence_refs=["exp2"],
            belief_type="experiential",
            immutable=False,
            rationale="Creator 2",
            updated_by="creator2",
        )
        results.append(("creator2", success))

    await asyncio.gather(creator1(), creator2())

    # Verify results
    successes = [r for (_, r) in results if r]
    failures = [r for (_, r) in results if not r]

    assert len(successes) == 1, "Exactly one creator should succeed"
    assert len(failures) == 1, "Exactly one creator should fail"

    # Verify belief exists with first creator's data
    beliefs = store.get_current([belief_id])
    assert belief_id in beliefs
    assert beliefs[belief_id].ver == 1

    print("✓ Concurrent creation: one succeeded, one failed correctly")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_version_index_consistency(belief_store_dir: Path):
    """Test version consistency across multiple updates.

    Acceptance:
    - After many updates, index.json always matches current.json versions
    """
    store = BeliefStore(belief_store_dir)

    belief_id = "test.version-consistency"
    store.create_belief(
        belief_id=belief_id,
        statement="Initial statement",
        state=BeliefState.TENTATIVE,
        confidence=0.5,
        evidence_refs=["exp1"],
        belief_type="experiential",
        immutable=False,
        rationale="Test",
        updated_by="test",
    )

    # Apply 50 sequential deltas
    for i in range(50):
        beliefs = store.get_current([belief_id])
        current_ver = beliefs[belief_id].ver

        success = store.apply_delta(
            belief_id=belief_id,
            from_ver=current_ver,
            op=DeltaOp.REINFORCE,
            confidence_delta=0.02,
            updated_by="test",
            reason=f"Update {i}",
        )
        assert success, f"Update {i} should succeed"

        # Verify consistency after each update
        with open(belief_store_dir / "beliefs" / "current.json") as f:
            current = json.load(f)
        with open(belief_store_dir / "beliefs" / "index.json") as f:
            index = json.load(f)

        assert index[belief_id] == current[belief_id]["ver"], \
            f"Version mismatch at update {i}"

    # Final check
    beliefs = store.get_current([belief_id])
    assert beliefs[belief_id].ver == 51  # 1 create + 50 updates

    print(f"✓ Version consistency maintained across 50 updates")
