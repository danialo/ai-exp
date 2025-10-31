"""
Redis-based distributed lock for awareness loop single-runner guarantee.

Uses fencing token (timestamp) to detect lock ownership changes and prevent
split-brain scenarios in multi-worker deployments.
"""

import asyncio
import time
from typing import Optional
from redis.asyncio import Redis
from redis.exceptions import RedisError


class AwarenessLockError(Exception):
    """Raised when lock cannot be acquired or is lost."""
    pass


class AwarenessLock:
    """
    Distributed lock with heartbeat renewal for single-runner enforcement.

    The lock uses a fencing token (monotonic nanosecond timestamp) to detect
    ownership changes. If the token changes, the lock has been lost to another
    process.
    """

    LOCK_KEY = "awareness:lock"
    LOCK_TTL = 10  # seconds
    RENEW_INTERVAL = 5  # seconds

    def __init__(self, redis_client: Redis):
        """
        Initialize lock manager.

        Args:
            redis_client: Async Redis client instance
        """
        self.redis = redis_client
        self.token: Optional[int] = None
        self.running = False
        self._renew_task: Optional[asyncio.Task] = None

    async def acquire(self, timeout: float = 5.0) -> bool:
        """
        Attempt to acquire the lock with fencing token.

        Args:
            timeout: Maximum seconds to wait for lock acquisition

        Returns:
            True if lock acquired, False if timeout

        Raises:
            AwarenessLockError: If Redis operation fails
        """
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            try:
                # Generate fencing token from wall clock nanoseconds
                self.token = time.time_ns()

                # Try to acquire lock with NX (only if not exists)
                acquired = await self.redis.set(
                    self.LOCK_KEY,
                    str(self.token),
                    nx=True,
                    ex=self.LOCK_TTL
                )

                if acquired:
                    self.running = True
                    # Start heartbeat renewal task
                    self._renew_task = asyncio.create_task(self._renew_loop())
                    return True

                # Lock held by another process, wait briefly
                await asyncio.sleep(0.5)

            except RedisError as e:
                raise AwarenessLockError(f"Failed to acquire lock: {e}") from e

        return False

    async def _renew_loop(self):
        """
        Background task that renews lock TTL periodically.

        Validates fencing token on each renewal to detect ownership loss.
        """
        while self.running:
            try:
                await asyncio.sleep(self.RENEW_INTERVAL)

                if not self.running:
                    break

                # Check current token
                current_token_bytes = await self.redis.get(self.LOCK_KEY)

                if current_token_bytes is None:
                    raise AwarenessLockError("Lock disappeared during renewal")

                current_token = int(current_token_bytes.decode('utf-8'))

                if current_token != self.token:
                    raise AwarenessLockError(
                        f"Lost lock ownership: token mismatch "
                        f"(ours={self.token}, current={current_token})"
                    )

                # Renew TTL
                await self.redis.expire(self.LOCK_KEY, self.LOCK_TTL)

            except AwarenessLockError:
                # Lost ownership, stop immediately
                self.running = False
                raise
            except RedisError as e:
                # Redis communication error, stop to be safe
                self.running = False
                raise AwarenessLockError(f"Lock renewal failed: {e}") from e

    async def release(self):
        """
        Release the lock gracefully.

        Stops renewal and deletes lock key if we still own it.
        """
        self.running = False

        # Cancel renewal task
        if self._renew_task and not self._renew_task.done():
            self._renew_task.cancel()
            try:
                await self._renew_task
            except asyncio.CancelledError:
                pass

        # Delete lock if we still own it
        if self.token is not None:
            try:
                current_token_bytes = await self.redis.get(self.LOCK_KEY)
                if current_token_bytes is not None:
                    current_token = int(current_token_bytes.decode('utf-8'))
                    if current_token == self.token:
                        await self.redis.delete(self.LOCK_KEY)
            except RedisError:
                # Best effort, ignore errors during release
                pass
            finally:
                self.token = None

    def is_held(self) -> bool:
        """Check if we currently hold the lock."""
        return self.running and self.token is not None

    async def __aenter__(self):
        """Context manager entry."""
        if not await self.acquire():
            raise AwarenessLockError("Failed to acquire lock within timeout")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.release()
