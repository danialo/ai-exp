"""Test ledger integrity and SHA chain (Phase 5.5)."""

import gzip
import json
import time
from pathlib import Path
import pytest
from freezegun import freeze_time
from src.services.identity_ledger import (
    append_event,
    LedgerEvent,
)


def get_day_stamp(ts: float) -> str:
    """Get YYYYMMDD stamp."""
    from datetime import datetime
    return datetime.utcfromtimestamp(ts).strftime("%Y%m%d")


def read_ledger_file(ledger_dir: Path, day: str) -> list:
    """Read ledger entries for a day."""
    ledger_file = ledger_dir / "identity" / f"ledger-{day}.ndjson.gz"
    entries = []

    if ledger_file.exists():
        with gzip.open(ledger_file, "rt") as f:
            for line in f:
                entries.append(json.loads(line))

    return entries


def verify_chain_integrity(ledger_path: Path, day: str) -> bool:
    """Verify SHA chain integrity for a given day.

    Args:
        ledger_path: Path to ledger directory
        day: Day stamp (YYYYMMDD)

    Returns:
        True if chain is valid, False otherwise
    """
    import hashlib

    ledger_file = ledger_path / f"ledger-{day}.ndjson.gz"

    if not ledger_file.exists():
        return True  # No file to verify

    try:
        with gzip.open(ledger_file, "rt") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            entry = json.loads(line)

            # Recompute SHA
            data_to_hash = {k: v for k, v in entry.items() if k != "sha"}
            canonical = json.dumps(data_to_hash, sort_keys=True, separators=(",", ":"))
            computed_sha = hashlib.sha256(canonical.encode()).hexdigest()

            # Compare with stored SHA
            if entry.get("sha") != computed_sha:
                return False

            # Check chain link
            if i > 0:
                prev_entry = json.loads(lines[i-1])
                if entry.get("prev_sha") != prev_entry.get("sha"):
                    return False

        return True

    except Exception:
        return False


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_sha_chain_day_rollover(ledger_dir: Path):
    """Test SHA chain across day rollover.

    Acceptance:
    - Append at T=23:59 and T=00:01
    - First entry of new file prev_sha equals final SHA of previous file
    """
    ledger_path = ledger_dir / "identity"
    ledger_path.mkdir(parents=True, exist_ok=True)

    # Append entry at 23:59
    with freeze_time("2025-10-31 23:59:00"):
        event1 = LedgerEvent(
            ts=time.time(),
            event="test_before_midnight",
            cause="test",
            data={"test": "entry1"},
        )
        append_event(event1, ledger_path)

        day1 = get_day_stamp(time.time())
        entries_day1 = read_ledger_file(ledger_dir, day1)
        assert len(entries_day1) > 0, "Should have entry before midnight"

        last_sha_day1 = entries_day1[-1]["sha"]

    # Append entry at 00:01 next day
    with freeze_time("2025-11-01 00:01:00"):
        event2 = LedgerEvent(
            ts=time.time(),
            event="test_after_midnight",
            cause="test",
            data={"test": "entry2"},
        )
        append_event(event2, ledger_path)

        day2 = get_day_stamp(time.time())
        entries_day2 = read_ledger_file(ledger_dir, day2)
        assert len(entries_day2) > 0, "Should have entry after midnight"

        first_prev_sha_day2 = entries_day2[0]["prev_sha"]

    # Verify chain links across days
    assert first_prev_sha_day2 == last_sha_day1, \
        f"Chain broken at day rollover: day1_last={last_sha_day1}, day2_first_prev={first_prev_sha_day2}"

    print(f"✓ SHA chain verified across day rollover: {day1} → {day2}")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_tamper_detection(ledger_dir: Path):
    """Test tamper detection by flipping one byte.

    Acceptance:
    - Flip one byte in historical entry
    - Integrity verifier must fail
    - healthz should report ledger_ok=false
    """
    ledger_path = ledger_dir / "identity"
    ledger_path.mkdir(parents=True, exist_ok=True)

    # Create several entries
    with freeze_time("2025-10-31 12:00:00"):
        for i in range(5):
            event = LedgerEvent(
                ts=time.time() + i,
                event=f"test_entry_{i}",
                cause="test",
                data={"index": i},
            )
            append_event(event, ledger_path)

        day = get_day_stamp(time.time())

    # Read entries
    entries = read_ledger_file(ledger_dir, day)
    assert len(entries) == 5, "Should have 5 entries"

    # Verify integrity before tampering
    is_valid_before = verify_chain_integrity(ledger_path, day)
    assert is_valid_before, "Chain should be valid before tampering"

    # Tamper with middle entry
    ledger_file = ledger_path / f"ledger-{day}.ndjson.gz"
    with gzip.open(ledger_file, "rt") as f:
        lines = f.readlines()

    # Modify middle entry (flip one byte in data)
    middle_idx = len(lines) // 2
    entry = json.loads(lines[middle_idx])
    entry["data"]["index"] = 999  # Tamper with data
    lines[middle_idx] = json.dumps(entry) + "\n"

    # Write back tampered file
    with gzip.open(ledger_file, "wt") as f:
        f.writelines(lines)

    # Verify integrity after tampering
    is_valid_after = verify_chain_integrity(ledger_path, day)
    assert not is_valid_after, "Chain should be invalid after tampering"

    print("✓ Tamper detection: integrity check failed after byte flip")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_genesis_entry(ledger_dir: Path):
    """Test first entry uses 'genesis' as prev_sha.

    Acceptance:
    - First entry of first day has prev_sha='genesis'
    """
    ledger_path = ledger_dir / "identity"
    ledger_path.mkdir(parents=True, exist_ok=True)

    with freeze_time("2025-10-31 12:00:00"):
        event = LedgerEvent(
            ts=time.time(),
            event="first_entry",
            cause="test",
            data={"test": "genesis"},
        )
        append_event(event, ledger_path)

        day = get_day_stamp(time.time())

    entries = read_ledger_file(ledger_dir, day)
    assert len(entries) > 0, "Should have at least one entry"

    first_entry = entries[0]
    assert first_entry["prev_sha"] == "genesis", \
        f"First entry should have prev_sha='genesis', got '{first_entry['prev_sha']}'"

    print("✓ Genesis entry: first prev_sha = 'genesis'")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_chain_verification_multiple_days(ledger_dir: Path):
    """Test chain verification across multiple days.

    Acceptance:
    - Append entries across 3 days
    - Verify chain integrity for each day
    """
    ledger_path = ledger_dir / "identity"
    ledger_path.mkdir(parents=True, exist_ok=True)

    days = []

    # Day 1
    with freeze_time("2025-10-31 12:00:00"):
        for i in range(3):
            event = LedgerEvent(
                ts=time.time() + i,
                event=f"day1_entry_{i}",
                cause="test",
                data={"day": 1, "index": i},
            )
            append_event(event, ledger_path)

        days.append(get_day_stamp(time.time()))

    # Day 2
    with freeze_time("2025-11-01 12:00:00"):
        for i in range(3):
            event = LedgerEvent(
                ts=time.time() + i,
                event=f"day2_entry_{i}",
                cause="test",
                data={"day": 2, "index": i},
            )
            append_event(event, ledger_path)

        days.append(get_day_stamp(time.time()))

    # Day 3
    with freeze_time("2025-11-02 12:00:00"):
        for i in range(3):
            event = LedgerEvent(
                ts=time.time() + i,
                event=f"day3_entry_{i}",
                cause="test",
                data={"day": 3, "index": i},
            )
            append_event(event, ledger_path)

        days.append(get_day_stamp(time.time()))

    # Verify each day
    for day in days:
        is_valid = verify_chain_integrity(ledger_path, day)
        assert is_valid, f"Chain should be valid for day {day}"

    print(f"✓ Chain integrity verified across {len(days)} days")


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_belief_versioning_logged_to_ledger(ledger_dir: Path, belief_store_dir: Path):
    """Test belief version changes are logged to ledger.

    Acceptance:
    - Belief creation and updates appear in ledger
    - Ledger entries reference belief_id and versions
    """
    from src.services.belief_store import BeliefStore, BeliefState, DeltaOp

    ledger_path = ledger_dir / "identity"
    ledger_path.mkdir(parents=True, exist_ok=True)

    store = BeliefStore(belief_store_dir)

    with freeze_time("2025-10-31 12:00:00"):
        # Create belief
        belief_id = "test.ledger-logging"
        store.create_belief(
            belief_id=belief_id,
            statement="Test statement",
            state=BeliefState.TENTATIVE,
            confidence=0.5,
            evidence_refs=["exp1"],
            belief_type="experiential",
            immutable=False,
            rationale="Test",
            updated_by="test",
        )

        # Update belief
        store.apply_delta(
            belief_id=belief_id,
            from_ver=1,
            op=DeltaOp.UPDATE,
            confidence_delta=0.05,
            updated_by="test",
            reason="Test update",
        )

        day = get_day_stamp(time.time())

    # Check ledger for belief events
    entries = read_ledger_file(ledger_dir, day)

    belief_events = [
        e for e in entries
        if e.get("event") == "belief_versioned" and e.get("data", {}).get("belief_id") == belief_id
    ]

    # Should have at least 2 events (create + update)
    assert len(belief_events) >= 2, \
        f"Expected at least 2 belief events in ledger, got {len(belief_events)}"

    print(f"✓ Belief versioning logged to ledger: {len(belief_events)} events")
