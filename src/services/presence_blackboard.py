"""
Presence blackboard - shared state store for awareness loop.

Redis-backed state store that provides atomic operations and pub/sub
for cross-worker awareness synchronization.
"""

import asyncio
import json
from typing import Dict, List, Optional
import numpy as np
from redis.asyncio import Redis
from redis.exceptions import RedisError
from src.services import awareness_metrics


class PresenceBlackboard:
    """
    Shared state store for awareness presence.

    Provides atomic read/write operations backed by Redis for multi-worker
    deployments. State includes:
    - presence_scalar: Current presence magnitude [0, 1]
    - presence_vec: Presence vector embedding (float32, 64-dim)
    - meta: Presence metadata (entropy, novelty, etc.)
    - introspection_notes: Recent introspection text
    - dissonance_last: Last dissonance resolution
    """

    # Redis key prefixes
    PREFIX = "awareness:"
    KEY_SCALAR = f"{PREFIX}presence_scalar"
    KEY_VEC = f"{PREFIX}presence_vec"
    KEY_META = f"{PREFIX}meta"
    KEY_NOTES = f"{PREFIX}introspection_notes"
    KEY_DISSONANCE = f"{PREFIX}dissonance_last"

    # Pub/sub channels
    CHANNEL_SHIFT = f"{PREFIX}shift"

    def __init__(self, redis_client: Redis):
        """
        Initialize blackboard.

        Args:
            redis_client: Async Redis client
        """
        self.redis = redis_client
        self._lock = asyncio.Lock()

    async def update_presence(
        self,
        scalar: float,
        vec: np.ndarray,
        meta: dict
    ) -> None:
        """
        Atomically update presence state.

        Args:
            scalar: Presence scalar [0, 1]
            vec: Presence vector (numpy array, float32)
            meta: Metadata dict with keys: entropy, novelty, sim_prev, sim_self, etc.
        """
        async with self._lock:
            try:
                # Use pipeline for atomicity
                pipe = self.redis.pipeline()

                # Clamp scalar to [0, 1]
                scalar = float(np.clip(scalar, 0.0, 1.0))

                # Replace NaN with 0
                if np.isnan(scalar):
                    scalar = 0.0

                # Ensure vec is float32, length 64
                if vec.dtype != np.float32:
                    vec = vec.astype(np.float32)

                # Pad or truncate to 64
                if len(vec) < 64:
                    vec = np.pad(vec, (0, 64 - len(vec)), mode='constant')
                elif len(vec) > 64:
                    vec = vec[:64]

                # Replace NaN in vec
                vec = np.nan_to_num(vec, nan=0.0)

                # Store scalar (3 decimals)
                pipe.set(self.KEY_SCALAR, f"{scalar:.3f}")

                # Store vec as bytes
                pipe.set(self.KEY_VEC, vec.tobytes())

                # Store meta as JSON
                # Clean meta - remove NaN values
                clean_meta = {
                    k: (0.0 if isinstance(v, float) and np.isnan(v) else v)
                    for k, v in meta.items()
                }
                pipe.set(self.KEY_META, json.dumps(clean_meta))

                await pipe.execute()

                # Update metrics
                awareness_metrics.update_presence_gauges(
                    scalar=scalar,
                    novelty=clean_meta.get("novelty", 0.0),
                    sim_self_live=clean_meta.get("sim_self_live", 0.0),
                    sim_self_origin=clean_meta.get("sim_self_origin", 0.0),
                    sim_prev=clean_meta.get("sim_prev", 0.0),
                    entropy=clean_meta.get("entropy", 0.0),
                    coherence_drop=clean_meta.get("coherence_drop", 0.0)
                )

                awareness_metrics.increment_redis_ops(3)

                # Check if awareness shift should be published
                novelty = clean_meta.get("novelty", 0.0)
                coherence_drop = clean_meta.get("coherence_drop", 0.0)
                if novelty > 0.6 or coherence_drop > 0.4:
                    print(f"🚨 [SHIFT] Publishing shift! novelty={novelty:.3f}, coherence_drop={coherence_drop:.3f}")
                    await self.publish_shift(clean_meta)

            except RedisError as e:
                # Log but don't crash awareness loop
                pass

    async def get_presence_scalar(self) -> float:
        """
        Get current presence scalar.

        Returns:
            Presence scalar [0, 1]
        """
        try:
            value = await self.redis.get(self.KEY_SCALAR)
            awareness_metrics.increment_redis_ops()

            if value is None:
                return 0.0

            return float(value.decode('utf-8'))

        except (RedisError, ValueError):
            return 0.0

    async def get_presence_vec(self) -> Optional[np.ndarray]:
        """
        Get current presence vector.

        Returns:
            Presence vector (64-dim float32), or None if not available
        """
        try:
            data = await self.redis.get(self.KEY_VEC)
            awareness_metrics.increment_redis_ops()

            if data is None:
                return None

            # Expect 256 bytes (64 * 4 bytes for float32)
            return np.frombuffer(data, dtype=np.float32)

        except RedisError:
            return None

    async def get_meta(self) -> dict:
        """
        Get presence metadata.

        Returns:
            Metadata dict, or empty dict if not available
        """
        try:
            data = await self.redis.get(self.KEY_META)
            awareness_metrics.increment_redis_ops()

            if data is None:
                return {}

            return json.loads(data.decode('utf-8'))

        except (RedisError, json.JSONDecodeError):
            return {}

    async def add_introspection_note(self, note: str) -> None:
        """
        Add introspection note to rolling buffer.

        Args:
            note: Introspection text
        """
        if not note:
            return

        try:
            # LPUSH + LTRIM to maintain max 20 notes
            pipe = self.redis.pipeline()
            pipe.lpush(self.KEY_NOTES, note)
            pipe.ltrim(self.KEY_NOTES, 0, 19)  # Keep only 0-19 (20 items)
            await pipe.execute()

            awareness_metrics.increment_redis_ops(2)

        except RedisError:
            pass

    async def get_introspection_notes(self, limit: int = 20) -> List[str]:
        """
        Get recent introspection notes.

        Args:
            limit: Maximum number of notes to return

        Returns:
            List of note strings (newest first)
        """
        try:
            notes = await self.redis.lrange(self.KEY_NOTES, 0, limit - 1)
            awareness_metrics.increment_redis_ops()

            return [note.decode('utf-8') for note in notes]

        except RedisError:
            return []

    async def set_dissonance_last(self, dissonance: dict) -> None:
        """
        Store last dissonance resolution.

        Args:
            dissonance: Dict with keys: protected, revised, strategy
        """
        try:
            await self.redis.set(self.KEY_DISSONANCE, json.dumps(dissonance))
            awareness_metrics.increment_redis_ops()

        except RedisError:
            pass

    async def get_dissonance_last(self) -> Optional[dict]:
        """
        Get last dissonance resolution.

        Returns:
            Dissonance dict, or None if not available
        """
        try:
            data = await self.redis.get(self.KEY_DISSONANCE)
            awareness_metrics.increment_redis_ops()

            if data is None:
                return None

            return json.loads(data.decode('utf-8'))

        except (RedisError, json.JSONDecodeError):
            return None

    async def publish_shift(self, meta: dict) -> None:
        """
        Publish awareness shift event.

        Args:
            meta: Metadata to include in shift event
        """
        try:
            payload = json.dumps({
                "ts": int(asyncio.get_event_loop().time()),
                "meta": meta
            })

            await self.redis.publish(self.CHANNEL_SHIFT, payload)
            awareness_metrics.increment_redis_ops()
            awareness_metrics.get_metrics().increment_counter("awareness_shifts")

        except RedisError:
            pass

    async def subscribe_to_shifts(self, callback):
        """
        Subscribe to awareness shift events.

        Args:
            callback: Async function to call on shift event
        """
        pubsub = self.redis.pubsub()

        try:
            await pubsub.subscribe(self.CHANNEL_SHIFT)

            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'].decode('utf-8'))
                        await callback(data)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass

        except RedisError:
            pass
        finally:
            await pubsub.unsubscribe(self.CHANNEL_SHIFT)
            await pubsub.close()

    async def clear(self) -> None:
        """Clear all presence state (for testing)."""
        keys = [
            self.KEY_SCALAR,
            self.KEY_VEC,
            self.KEY_META,
            self.KEY_NOTES,
            self.KEY_DISSONANCE
        ]

        try:
            await self.redis.delete(*keys)
            awareness_metrics.increment_redis_ops(len(keys))
        except RedisError:
            pass
