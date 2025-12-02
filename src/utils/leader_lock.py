"""Leader lock for single-worker background job execution.

In multi-worker uvicorn deployments, background loops (gardener, introspection)
should only run in ONE worker to avoid version mismatch storms and duplicated work.

Uses fcntl.flock() for cross-process coordination.
"""

import os
import fcntl
import logging

logger = logging.getLogger(__name__)


class LeaderLock:
    """File-based leader election using flock().

    Only one process can hold the lock at a time. Non-leaders fail fast
    and can still serve HTTP requests.

    Usage:
        lock = LeaderLock()
        if lock.try_acquire():
            # This worker runs background loops
            start_background_threads()
        else:
            # This worker only serves HTTP
            logger.info("Not leader, skipping background loops")
    """

    def __init__(self, path: str = "/tmp/astra.leader.lock"):
        self.path = path
        self.fd = None
        self._is_leader = False

    def try_acquire(self) -> bool:
        """Attempt to acquire the leader lock (non-blocking).

        Returns:
            True if this process is now the leader, False otherwise.
        """
        # Ensure directory exists if path has one
        dir_path = os.path.dirname(self.path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Write our PID to the lock file for debugging
            os.ftruncate(self.fd, 0)
            os.write(self.fd, str(os.getpid()).encode("utf-8"))
            os.fsync(self.fd)
            self._is_leader = True
            logger.warning(f"[LeaderLock] Acquired leader lock: {self.path} pid={os.getpid()}")
            return True
        except BlockingIOError:
            # Another process holds the lock
            os.close(self.fd)
            self.fd = None
            self._is_leader = False
            logger.warning(f"[LeaderLock] Not leader (another worker is leader): {self.path} pid={os.getpid()}")
            return False

    def release(self) -> None:
        """Release the leader lock if held.

        Only releases if this process actually holds the lock (is_leader=True).
        Safe to call on non-leader workers - will no-op.
        """
        if not self._is_leader or self.fd is None:
            return
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            logger.warning(f"[LeaderLock] Released leader lock: {self.path} pid={os.getpid()}")
        except Exception as e:
            logger.warning(f"[LeaderLock] Error releasing lock: {e}")
        finally:
            self.fd = None
            self._is_leader = False

    @property
    def is_leader(self) -> bool:
        """Check if this process currently holds the leader lock."""
        return self._is_leader

    def __del__(self):
        """Release lock on garbage collection."""
        self.release()
