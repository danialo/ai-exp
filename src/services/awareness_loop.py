"""
Core awareness loop - continuous presence maintenance.

Four-tier tick architecture:
- Fast (2 Hz): drain percept queue, compute cheap stats, publish
- Slow (0.1 Hz): re-embed if needed, compute novelty/similarity
- Introspection (30s ± jitter): LLM introspection with budget
- Snapshot (60s): atomic persistence
"""

import asyncio
import logging
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Deque
import numpy as np

logger = logging.getLogger(__name__)

from redis.asyncio import Redis
from src.services.awareness_lock import AwarenessLock, AwarenessLockError
from src.services.presence_blackboard import PresenceBlackboard
from src.services.awareness_persistence import (
    AwarenessPersistence,
    AwarenessSnapshot,
    AnchorData,
    NoteEntry
)
from src.services.embedding_cache import EmbeddingCache
from src.services.pii_redactor import redact_pii
from src.services import awareness_metrics
from src.memory.embedding import EmbeddingProvider


@dataclass
class Percept:
    """Single event/observation in awareness stream."""
    ts: float  # timestamp
    kind: str  # "token", "tool", "time", "sys", "user"
    payload: Dict[str, Any]


@dataclass
class AwarenessConfig:
    """Configuration for awareness loop."""
    enabled: bool = False
    tick_rate_fast: float = 2.0  # Hz
    tick_rate_slow: float = 0.1  # Hz
    introspection_interval: int = 30  # seconds
    introspection_jitter: int = 5  # seconds
    snapshot_interval: int = 60  # seconds
    buffer_size: int = 512
    queue_maxsize: int = 2048
    notes_max: int = 100
    embedding_dim: int = 64
    embedding_cache_ttl: int = 300
    watchdog_threshold_ms: float = 250.0
    watchdog_strikes: int = 3
    introspection_budget_per_min: int = 100


class AwarenessMode:
    """Operating modes for awareness loop."""
    FULL = "full"
    MODERATE = "moderate"
    MINIMAL = "minimal"


class AwarenessLoop:
    """
    Continuous awareness loop with multi-tier tick architecture.

    Maintains presence state and broadcasts to blackboard for consumption
    by other systems (dissonance checker, mood, response generation).
    """

    def __init__(
        self,
        redis_client: Redis,
        embedding_provider: EmbeddingProvider,
        data_dir: Path,
        config: AwarenessConfig,
        llm_service: Optional[Any] = None
    ):
        """
        Initialize awareness loop.

        Args:
            redis_client: Redis client for state and coordination
            embedding_provider: Provider for text embeddings
            data_dir: Directory for state persistence
            config: Configuration object
            llm_service: LLM service for introspection (optional)
        """
        self.config = config
        self.llm_service = llm_service

        # Components
        self.lock = AwarenessLock(redis_client)
        self.blackboard = PresenceBlackboard(redis_client)
        self.persistence = AwarenessPersistence(data_dir)
        self.embedding_cache = EmbeddingCache(
            embedding_provider,
            ttl_seconds=config.embedding_cache_ttl,
            model_version="sentence-transformers-v1"
        )

        # State
        self.session_id = uuid.uuid4().hex[:8]
        self.running = False
        self.mode = AwarenessMode.FULL

        # Percept buffer
        self.percepts: Deque[Percept] = deque(maxlen=config.buffer_size)
        self.percept_queue: asyncio.Queue[Percept] = asyncio.Queue(maxsize=config.queue_maxsize)

        # Tick tracking
        self.tick_id = 0
        self.last_slow_tick = 0.0
        self.last_introspection = 0.0
        self.last_snapshot = 0.0

        # Presence state
        self.last_presence_vec: Optional[np.ndarray] = None
        self.last_presence_scalar: float = 0.0
        self.anchors: Dict[str, np.ndarray] = {}
        self.notes: Deque[str] = deque(maxlen=config.notes_max)

        # Slow-loop computed metrics (preserved across fast ticks)
        self.last_novelty: float = 0.0
        self.last_sim_prev: float = 0.0
        self.last_sim_live: float = 0.0  # Similarity to live anchor (for coherence)
        self.last_sim_origin: float = 0.0  # Similarity to origin anchor (for drift)
        self.last_coherence_drop: float = 0.0

        # Watchdog
        self.slow_tick_count = 0

        # Introspection budget
        self.introspection_tokens_used = 0
        self.last_budget_reset = time.time()

        # Tasks
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """
        Start awareness loop.

        Acquires distributed lock and loads persisted state.
        """
        if not self.config.enabled:
            return

        # Acquire lock
        acquired = await self.lock.acquire(timeout=5.0)
        if not acquired:
            raise AwarenessLockError("Failed to acquire awareness lock")

        # Load snapshot
        snapshot = await self.persistence.load_snapshot()
        if snapshot:
            await self._restore_from_snapshot(snapshot)

        # Seed dual anchors if not present
        if "self_anchor_origin" not in self.anchors:
            cold_anchor = self.persistence.get_cold_start_anchor()
            if cold_anchor is not None:
                # Initialize both anchors to same starting point
                self.anchors["self_anchor_origin"] = cold_anchor.copy()
                self.anchors["self_anchor_live"] = cold_anchor.copy()

        self.running = True

    async def _restore_from_snapshot(self, snapshot: AwarenessSnapshot) -> None:
        """Restore state from snapshot."""
        # Restore anchors
        for name, anchor_data in snapshot.anchors.items():
            self.anchors[name] = anchor_data.to_array()

        # Restore notes
        for note_entry in snapshot.notes[-self.config.notes_max:]:
            self.notes.append(note_entry.crumb)

    async def run(self) -> None:
        """
        Main loop - spawns tick tasks.
        """
        if not self.running:
            return

        try:
            # Start time pacer (feeds time percepts)
            self._tasks.append(asyncio.create_task(self._time_pacer()))

            # Start tick loops
            self._tasks.append(asyncio.create_task(self._fast_loop()))
            self._tasks.append(asyncio.create_task(self._slow_loop()))
            self._tasks.append(asyncio.create_task(self._introspection_loop()))
            self._tasks.append(asyncio.create_task(self._snapshot_loop()))

            # Wait for tasks
            await asyncio.gather(*self._tasks)

        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop awareness loop gracefully."""
        self.running = False

        # Cancel tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for cancellation
        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Final snapshot
        await self.flush_snapshot()

        # Release lock
        await self.lock.release()

    async def observe(self, kind: str, payload: Dict[str, Any]) -> None:
        """
        Add observation to percept queue.

        Args:
            kind: Percept type
            payload: Percept data
        """
        percept = Percept(time.time(), kind, payload)

        try:
            self.percept_queue.put_nowait(percept)
        except asyncio.QueueFull:
            # Drop event, increment metric
            awareness_metrics.increment_events_dropped()

    async def _time_pacer(self) -> None:
        """Feed time percepts to keep awareness active in silence."""
        while self.running:
            await self.observe("time", {"tick": True})
            await asyncio.sleep(0.5)

    async def _fast_loop(self) -> None:
        """Fast loop (2 Hz): drain queue, cheap stats, publish."""
        period = 1.0 / self.config.tick_rate_fast

        while self.running:
            t0 = time.perf_counter()

            try:
                await self._fast_tick()
            except Exception:
                pass  # Never crash awareness loop

            dt = time.perf_counter() - t0
            awareness_metrics.record_tick_time(dt * 1000)

            # Watchdog
            if dt * 1000 > self.config.watchdog_threshold_ms:
                self.slow_tick_count += 1
                if self.slow_tick_count >= self.config.watchdog_strikes:
                    await self._degrade_to_minimal()
            else:
                self.slow_tick_count = 0

            await asyncio.sleep(max(0.0, period - dt))

    async def _fast_tick(self) -> None:
        """Execute fast tick."""
        self.tick_id += 1

        # Drain queue (non-blocking)
        percepts_added = 0
        try:
            while True:
                p = self.percept_queue.get_nowait()
                self.percepts.append(p)
                percepts_added += 1
        except asyncio.QueueEmpty:
            pass

        # Compute cheap stats
        entropy = self._compute_entropy()

        # Reuse last embedding if no significant change
        cur_vec = self.last_presence_vec if self.last_presence_vec is not None else np.zeros(self.config.embedding_dim, dtype=np.float32)

        # Compute scalar presence
        scalar = self._compute_presence_scalar(entropy)

        # Update blackboard (preserve slow-loop computed metrics)
        meta = {
            "mode": self.mode,
            "entropy": entropy,
            "novelty": self.last_novelty,  # Preserved from slow loop
            "sim_prev": self.last_sim_prev,
            "sim_self_live": self.last_sim_live,
            "sim_self_origin": self.last_sim_origin,
            "coherence_drop": self.last_coherence_drop,
            "tick": self.tick_id,
        }

        await self.blackboard.update_presence(scalar, cur_vec, meta)
        self.last_presence_scalar = scalar

    async def _slow_loop(self) -> None:
        """Slow loop (0.1 Hz): re-embed, compute novelty/similarity."""
        period = 1.0 / self.config.tick_rate_slow

        while self.running:
            await asyncio.sleep(period)

            if not self.running:
                break

            try:
                await self._slow_tick()
            except Exception as e:
                logger.error(f"[SLOW] Error in slow tick: {e}", exc_info=True)

    async def _slow_tick(self) -> None:
        """Execute slow tick."""
        # Extract recent text
        recent_text = self._extract_recent_text()

        if not recent_text:
            print(f"⚠️  [SLOW TICK] No text extracted from {len(self.percepts)} percepts")
            logger.debug(f"[SLOW] No recent text extracted from {len(self.percepts)} percepts")
            return

        print(f"✓ [SLOW TICK] Extracted {len(recent_text)} chars from {len(self.percepts)} percepts")
        logger.debug(f"[SLOW] Extracted {len(recent_text)} chars from {len(self.percepts)} percepts")

        # Get embedding (uses cache when possible)
        embedding = await self.embedding_cache.get_embedding(
            recent_text,
            len(self.percepts)
        )

        if embedding is None:
            logger.warning(f"[SLOW] Failed to get embedding for text: {recent_text[:100]}")
            return

        logger.debug(f"[SLOW] Got embedding, computing novelty (last_vec exists: {self.last_presence_vec is not None})")

        # Compute novelty (similarity to previous)
        novelty = 0.0
        if self.last_presence_vec is not None:
            sim_prev = self._cosine_sim(embedding, self.last_presence_vec)
            novelty = max(0.0, 1.0 - sim_prev)
            print(f"🔍 [NOVELTY] sim_prev={sim_prev:.4f}, novelty={novelty:.4f}")
        else:
            print(f"🔍 [NOVELTY] No previous vec, novelty=0.0 (first tick)")

        # Compute dual-anchor similarities
        sim_live = 0.0
        sim_origin = 0.0

        if "self_anchor_live" in self.anchors:
            sim_live = self._cosine_sim(embedding, self.anchors["self_anchor_live"])
            sim_origin = self._cosine_sim(embedding, self.anchors["self_anchor_origin"])
        else:
            # Initialize dual anchors (cold start)
            self.anchors["self_anchor_origin"] = embedding.copy()
            self.anchors["self_anchor_live"] = embedding.copy()
            sim_live = 1.0
            sim_origin = 1.0

        # Coherence drop (sudden deviation from live anchor)
        coherence_drop = 0.0
        if self.last_sim_live > 0.0:
            coherence_drop = max(0.0, self.last_sim_live - sim_live)

        # Update embedding
        self.last_presence_vec = embedding

        # Recompute presence scalar (use sim_live, not sim_origin)
        entropy = self._compute_entropy()
        scalar = self._compute_presence_scalar(entropy, novelty=novelty, sim_self=sim_live)

        logger.debug(f"[SLOW] Computed: novelty={novelty:.3f}, sim_live={sim_live:.3f}, sim_origin={sim_origin:.3f}, coherence_drop={coherence_drop:.3f}, scalar={scalar:.3f}")

        # Update blackboard
        meta = {
            "mode": self.mode,
            "entropy": entropy,
            "novelty": novelty,
            "sim_prev": 1.0 - novelty if self.last_presence_vec is not None else 0.0,
            "sim_self_live": sim_live,
            "sim_self_origin": sim_origin,
            "coherence_drop": coherence_drop,
            "tick": self.tick_id,
        }

        await self.blackboard.update_presence(scalar, embedding, meta)

        # Persist computed metrics for fast loop
        self.last_novelty = novelty
        self.last_sim_prev = 1.0 - novelty if self.last_presence_vec is not None else 0.0
        self.last_sim_live = sim_live
        self.last_sim_origin = sim_origin
        self.last_coherence_drop = coherence_drop

    async def _introspection_loop(self) -> None:
        """Introspection loop (30s ± jitter): LLM introspection."""
        while self.running:
            # Jitter
            interval = self.config.introspection_interval + random.uniform(
                -self.config.introspection_jitter,
                self.config.introspection_jitter
            )

            await asyncio.sleep(interval)

            if not self.running:
                break

            try:
                await self._introspection_tick()
            except Exception:
                pass

    async def _introspection_tick(self) -> None:
        """Execute introspection tick."""
        # Check budget
        if not self._check_introspection_budget():
            awareness_metrics.increment_counter("introspection_skipped")
            return

        if self.llm_service is None:
            return

        # Choose prompt
        prompts = [
            "What am I currently attending to and why does it matter to me?",
            "Name one tension visible right now and what I am protecting.",
            "What shift would I make if no one asked me a question?",
            "Which value is salient, which is quiet?",
        ]

        prompt = random.choice(prompts)

        t0 = time.perf_counter()

        try:
            # Call LLM (short response)
            response = await self._call_llm_for_introspection(prompt)

            if response:
                # Redact PII
                redacted = redact_pii(response)

                # Add to notes
                self.notes.append(redacted)

                # Add to blackboard
                await self.blackboard.add_introspection_note(redacted)

                # Track tokens (estimate)
                self.introspection_tokens_used += len(response.split())

        except Exception:
            pass  # Best effort

        dt = time.perf_counter() - t0
        awareness_metrics.record_introspection_time(dt * 1000)

    async def _snapshot_loop(self) -> None:
        """Snapshot loop (60s): atomic persistence."""
        while self.running:
            await asyncio.sleep(self.config.snapshot_interval)

            if not self.running:
                break

            try:
                await self.flush_snapshot()
            except Exception:
                awareness_metrics.increment_snapshot_errors()

    async def flush_snapshot(self) -> None:
        """Save current state to disk."""
        snapshot = AwarenessSnapshot(
            version=1,
            session_id=self.session_id,
            last_snapshot_ts=int(time.time()),
            anchors={
                name: AnchorData.from_array(vec)
                for name, vec in self.anchors.items()
            },
            notes=[
                NoteEntry(
                    ts=int(time.time()),
                    presence=self.last_presence_scalar,
                    crumb=note
                )
                for note in list(self.notes)[-20:]  # Last 20
            ]
        )

        await self.persistence.save_snapshot(snapshot)

        # Also append to daily log
        meta = await self.blackboard.get_meta()
        await self.persistence.append_to_daily_log(
            self.session_id,
            self.last_presence_scalar,
            meta,
            list(self.notes)
        )

    def _compute_entropy(self) -> float:
        """Compute entropy of recent text tokens."""
        texts = [
            p.payload.get("text", "")
            for p in list(self.percepts)[-128:]
            if p.kind in ("token", "user")
        ]

        if not texts:
            return 0.0

        # Simple word-level entropy
        words = " ".join(texts).split()
        if len(words) < 2:
            return 0.0

        from collections import Counter
        counts = Counter(words)
        probs = np.array(list(counts.values()), dtype=float)
        probs /= probs.sum()

        entropy = -np.sum(probs * np.log(probs + 1e-9))
        return float(entropy)

    def _compute_presence_scalar(
        self,
        entropy: float,
        novelty: float = 0.0,
        sim_self: float = 0.0
    ) -> float:
        """
        Compute presence scalar from features.

        Args:
            entropy: Text entropy
            novelty: Novelty score
            sim_self: Self-similarity

        Returns:
            Presence scalar [0, 1]
        """
        # Simple weighted combination
        scalar = (
            0.25 * min(entropy / 3.0, 1.0) +
            0.35 * novelty +
            0.25 * sim_self +
            0.15 * 0.5  # baseline
        )

        return float(np.clip(scalar, 0.0, 1.0))

    def _extract_recent_text(self, window: int = 32) -> str:
        """Extract recent text from percepts."""
        texts = [
            p.payload.get("text", "")
            for p in list(self.percepts)[-window:]
            if p.kind in ("token", "user")
        ]

        return " ".join(texts)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        a_norm = a / (np.linalg.norm(a) + 1e-9)
        b_norm = b / (np.linalg.norm(b) + 1e-9)
        return float(np.clip(a_norm @ b_norm, -1.0, 1.0))

    def _check_introspection_budget(self) -> bool:
        """Check if introspection budget allows execution."""
        now = time.time()

        # Reset budget every minute
        if now - self.last_budget_reset >= 60.0:
            self.introspection_tokens_used = 0
            self.last_budget_reset = now

        return self.introspection_tokens_used < self.config.introspection_budget_per_min

    async def _call_llm_for_introspection(self, prompt: str) -> Optional[str]:
        """Call LLM for introspection."""
        if not self.llm_service:
            logger.warning("[INTRO] No LLM service available for introspection")
            return None

        try:
            # Call LLM with short introspection prompt
            messages = [{"role": "user", "content": prompt}]

            # Use generate_with_tools but without tools for simple completion
            result = self.llm_service.generate_with_tools(
                messages=messages,
                tools=[],  # No tools needed
                temperature=0.7,
                max_tokens=150,  # Keep short to respect budget
            )

            response = result["message"].content

            # Track token usage (approximate)
            self.introspection_tokens_used += len(prompt.split()) + len(response.split())

            logger.debug(f"[INTRO] Generated note: {response[:100]}")
            return response

        except Exception as e:
            logger.error(f"[INTRO] LLM introspection failed: {e}")
            return None

    async def _degrade_to_minimal(self) -> None:
        """Degrade to minimal mode due to performance issues."""
        if self.mode != AwarenessMode.MINIMAL:
            self.mode = AwarenessMode.MINIMAL
            awareness_metrics.increment_counter("minimal_mode_activations")

    def update_live_anchor_on_resolution(
        self,
        strategy: str,
        beliefs_touched: list,
        beta_week_cap: float = 0.01
    ) -> None:
        """Update live anchor after dissonance resolution.

        Args:
            strategy: Resolution strategy ("Commit", "Reframe", etc.)
            beliefs_touched: List of belief statements involved
            beta_week_cap: Maximum beta per week (default 0.01)
        """
        from src.services.identity_anchor import update_live_anchor, Anchors

        # Only update on Commit or Reframe
        if strategy not in ["Commit", "Reframe"]:
            logger.info(f"Skipping anchor update for strategy: {strategy}")
            return

        # Need current presence vector
        if self.last_presence_vec is None:
            logger.warning("No presence vector available for anchor update")
            return

        # Need both anchors
        if "self_anchor_origin" not in self.anchors or "self_anchor_live" not in self.anchors:
            logger.warning("Anchors not initialized, skipping update")
            return

        # Create Anchors object
        anchors_obj = Anchors(
            origin=self.anchors["self_anchor_origin"],
            live=self.anchors["self_anchor_live"],
            last_update_ts=getattr(self, '_anchor_last_update_ts', 0.0)
        )

        # Update live anchor
        updated_anchors = update_live_anchor(
            anchors_obj,
            self.last_presence_vec,
            strategy,
            beliefs_touched,
            beta_week_cap
        )

        # Store updated anchor
        self.anchors["self_anchor_live"] = updated_anchors.live
        self._anchor_last_update_ts = updated_anchors.last_update_ts

        logger.info(f"Updated live anchor for strategy={strategy}, beliefs={beliefs_touched}")
