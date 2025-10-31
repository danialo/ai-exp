"""
Atomic persistence for awareness state with schema validation.

Implements write-temp-fsync-rename pattern for crash safety and maintains
daily NDJSON logs for history tracking.
"""

import asyncio
import gzip
import json
import os
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import base64
import numpy as np


SCHEMA_VERSION = 1
MAX_NOTES_IN_SNAPSHOT = 100


@dataclass
class AnchorData:
    """Serializable anchor vector."""
    dim: int
    vec_b64: str  # base64-encoded float32 bytes

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "AnchorData":
        """Convert numpy array to serializable form."""
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return cls(
            dim=len(arr),
            vec_b64=base64.b64encode(arr.tobytes()).decode('ascii')
        )

    def to_array(self) -> np.ndarray:
        """Convert back to numpy array."""
        data = base64.b64decode(self.vec_b64.encode('ascii'))
        return np.frombuffer(data, dtype=np.float32)


@dataclass
class NoteEntry:
    """Single awareness note with timestamp."""
    ts: int  # unix timestamp
    presence: float  # presence scalar at time of note
    crumb: str  # short text crumb


@dataclass
class AwarenessSnapshot:
    """Complete awareness state snapshot."""
    version: int
    session_id: str
    last_snapshot_ts: int
    anchors: Dict[str, AnchorData]  # name -> anchor
    notes: List[NoteEntry]

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "version": self.version,
            "session_id": self.session_id,
            "last_snapshot_ts": self.last_snapshot_ts,
            "anchors": {
                name: {"dim": anchor.dim, "vec_b64": anchor.vec_b64}
                for name, anchor in self.anchors.items()
            },
            "notes": [
                {"ts": note.ts, "presence": note.presence, "crumb": note.crumb}
                for note in self.notes
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AwarenessSnapshot":
        """Load from JSON dict with validation."""
        version = data.get("version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Schema version mismatch: expected {SCHEMA_VERSION}, got {version}"
            )

        return cls(
            version=version,
            session_id=data["session_id"],
            last_snapshot_ts=data["last_snapshot_ts"],
            anchors={
                name: AnchorData(**anchor_data)
                for name, anchor_data in data.get("anchors", {}).items()
            },
            notes=[
                NoteEntry(**note_data)
                for note_data in data.get("notes", [])
            ]
        )


class AwarenessPersistence:
    """
    Manages atomic persistence of awareness state.

    - Main snapshot: data/awareness_state.json
    - Daily logs: data/awareness_state-YYYYMMDD.ndjson.gz
    """

    def __init__(self, data_dir: Path):
        """
        Initialize persistence manager.

        Args:
            data_dir: Directory for storing awareness state
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.snapshot_path = self.data_dir / "awareness_state.json"
        self.temp_path = self.data_dir / "awareness_state.tmp"

    async def save_snapshot(self, snapshot: AwarenessSnapshot) -> None:
        """
        Atomically save snapshot to disk with fsync.

        Uses write-temp-fsync-rename pattern for crash safety.

        Args:
            snapshot: State to persist

        Raises:
            IOError: If write or fsync fails
        """
        try:
            # Serialize to JSON
            data = snapshot.to_dict()
            json_bytes = json.dumps(data, indent=2).encode('utf-8')

            # Write to temp file
            await asyncio.to_thread(self._write_atomic, json_bytes)

        except Exception as e:
            # Clean up temp file on error
            if self.temp_path.exists():
                self.temp_path.unlink()
            raise IOError(f"Failed to save snapshot: {e}") from e

    def _write_atomic(self, data: bytes) -> None:
        """
        Synchronous atomic write with fsync (runs in thread).

        Args:
            data: Bytes to write
        """
        # Write to temp
        with open(self.temp_path, 'wb') as f:
            f.write(data)
            f.flush()
            # Force write to disk
            os.fsync(f.fileno())

        # Atomic rename (on POSIX systems)
        self.temp_path.replace(self.snapshot_path)

    async def load_snapshot(self) -> Optional[AwarenessSnapshot]:
        """
        Load snapshot from disk with schema validation.

        Returns:
            Loaded snapshot, or None if file doesn't exist

        Raises:
            ValueError: If schema validation fails
            IOError: If read fails
        """
        if not self.snapshot_path.exists():
            return None

        try:
            data = await asyncio.to_thread(
                self.snapshot_path.read_text,
                encoding='utf-8'
            )
            json_data = json.loads(data)
            return AwarenessSnapshot.from_dict(json_data)

        except Exception as e:
            raise IOError(f"Failed to load snapshot: {e}") from e

    async def append_to_daily_log(
        self,
        session_id: str,
        presence: float,
        meta: dict,
        notes: List[str]
    ) -> None:
        """
        Append current state to daily NDJSON log.

        Creates new log file for each day, with automatic gzip compression
        for previous days.

        Args:
            session_id: Current session identifier
            presence: Current presence scalar
            meta: Presence metadata dict
            notes: Recent introspection notes
        """
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        log_path = self.data_dir / f"awareness_state-{today}.ndjson"

        entry = {
            "ts": int(datetime.now(timezone.utc).timestamp()),
            "session_id": session_id,
            "presence": presence,
            "meta": meta,
            "notes": notes[-5:] if notes else []  # Only last 5 notes
        }

        try:
            # Append line to log
            await asyncio.to_thread(
                self._append_line,
                log_path,
                json.dumps(entry) + "\n"
            )

            # Compress yesterday's log if it exists
            await self._compress_old_logs(today)

        except Exception as e:
            # Log append is best-effort, don't fail awareness loop
            pass

    def _append_line(self, path: Path, line: str) -> None:
        """Append line to file (thread-safe)."""
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line)
            f.flush()

    async def _compress_old_logs(self, today: str) -> None:
        """
        Compress NDJSON logs from previous days.

        Args:
            today: Today's date string (YYYYMMDD)
        """
        for log_file in self.data_dir.glob("awareness_state-*.ndjson"):
            # Extract date from filename
            date_str = log_file.stem.split('-', 2)[-1]

            if date_str < today:  # Older than today
                gz_path = log_file.with_suffix('.ndjson.gz')

                if not gz_path.exists():
                    try:
                        await asyncio.to_thread(
                            self._compress_file,
                            log_file,
                            gz_path
                        )
                    except Exception:
                        pass  # Best effort

    def _compress_file(self, src: Path, dst: Path) -> None:
        """Compress file with gzip."""
        with open(src, 'rb') as f_in:
            with gzip.open(dst, 'wb') as f_out:
                f_out.write(f_in.read())

        # Delete original after successful compression
        src.unlink()

    async def rotate_old_logs(self, keep_days: int = 30) -> int:
        """
        Delete compressed logs older than specified days.

        Args:
            keep_days: Number of days of logs to keep

        Returns:
            Number of files deleted
        """
        deleted = 0
        cutoff = datetime.now(timezone.utc).timestamp() - (keep_days * 86400)

        for gz_file in self.data_dir.glob("awareness_state-*.ndjson.gz"):
            try:
                if gz_file.stat().st_mtime < cutoff:
                    gz_file.unlink()
                    deleted += 1
            except Exception:
                pass  # Best effort

        return deleted

    def get_cold_start_anchor(self) -> Optional[np.ndarray]:
        """
        Extract self_anchor from last snapshot for cold start continuity.

        Returns:
            Self anchor vector, or None if not available
        """
        try:
            if not self.snapshot_path.exists():
                return None

            data = json.loads(self.snapshot_path.read_text(encoding='utf-8'))
            snapshot = AwarenessSnapshot.from_dict(data)

            if "self_anchor" in snapshot.anchors:
                return snapshot.anchors["self_anchor"].to_array()

            # Fallback: use latest note crumb
            if snapshot.notes:
                # Return None, let caller embed the crumb
                return None

        except Exception:
            return None

        return None
