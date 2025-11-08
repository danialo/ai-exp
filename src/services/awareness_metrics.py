"""
Metrics tracking for awareness loop.

Provides observability into awareness loop performance and state.
Simple in-memory implementation that can be extended to Prometheus, StatsD, etc.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List


@dataclass
class HistogramStats:
    """Statistics for histogram data."""
    count: int = 0
    sum: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    values: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def record(self, value: float) -> None:
        """Record a value."""
        self.count += 1
        self.sum += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.values.append(value)

    def get_percentile(self, p: float) -> float:
        """
        Get percentile value.

        Args:
            p: Percentile (0-100)

        Returns:
            Percentile value
        """
        if not self.values:
            return 0.0

        sorted_values = sorted(self.values)
        index = int(len(sorted_values) * p / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def get_mean(self) -> float:
        """Get mean value."""
        return self.sum / self.count if self.count > 0 else 0.0


class AwarenessMetrics:
    """
    Metrics collector for awareness loop.

    Tracks histograms, gauges, and counters for observability.
    """

    def __init__(self):
        """Initialize metrics collector."""
        # Histograms (timing data)
        self.histograms: Dict[str, HistogramStats] = {
            "awareness_tick_ms": HistogramStats(),
            "introspection_latency_ms": HistogramStats(),
            "embedding_latency_ms": HistogramStats(),
        }

        # Gauges (current values)
        self.gauges: Dict[str, float] = {
            "presence_scalar": 0.0,
            "novelty": 0.0,
            "sim_self_live": 0.0,  # Similarity to live anchor (coherence)
            "sim_self_origin": 0.0,  # Similarity to origin anchor (drift)
            "sim_prev": 0.0,
            "entropy": 0.0,
            "coherence_drop": 0.0,
        }

        # Counters (cumulative)
        self.counters: Dict[str, int] = {
            "snapshot_errors": 0,
            "events_dropped": 0,
            "introspection_skipped": 0,
            "embedding_cache_hits": 0,
            "embedding_cache_misses": 0,
            "redis_ops": 0,
            "awareness_shifts": 0,
            "minimal_mode_activations": 0,
        }

        # Rate tracking (for ops/sec calculations)
        self._rate_windows: Dict[str, Deque[tuple[float, int]]] = {
            "redis_ops": deque(maxlen=60),  # 60 seconds of data
        }

    def record_histogram(self, name: str, value: float) -> None:
        """
        Record histogram value.

        Args:
            name: Histogram name
            value: Value to record
        """
        if name in self.histograms:
            self.histograms[name].record(value)

    def set_gauge(self, name: str, value: float) -> None:
        """
        Set gauge value.

        Args:
            name: Gauge name
            value: Current value
        """
        self.gauges[name] = value

    def increment_counter(self, name: str, delta: int = 1) -> None:
        """
        Increment counter.

        Args:
            name: Counter name
            delta: Amount to increment
        """
        if name in self.counters:
            self.counters[name] += delta

            # Track rates for specific counters
            if name in self._rate_windows:
                self._rate_windows[name].append((time.time(), delta))

    def get_histogram_stats(self, name: str) -> dict:
        """
        Get histogram statistics.

        Args:
            name: Histogram name

        Returns:
            Dict with stats (count, mean, p50, p95, p99, min, max)
        """
        if name not in self.histograms:
            return {}

        hist = self.histograms[name]

        return {
            "count": hist.count,
            "mean": hist.get_mean(),
            "p50": hist.get_percentile(50),
            "p95": hist.get_percentile(95),
            "p99": hist.get_percentile(99),
            "min": hist.min if hist.count > 0 else 0.0,
            "max": hist.max if hist.count > 0 else 0.0,
        }

    def get_rate(self, counter_name: str, window_seconds: int = 60) -> float:
        """
        Calculate rate (per second) for a counter.

        Args:
            counter_name: Counter name
            window_seconds: Time window for rate calculation

        Returns:
            Rate (events per second)
        """
        if counter_name not in self._rate_windows:
            return 0.0

        window = self._rate_windows[counter_name]
        if not window:
            return 0.0

        now = time.time()
        cutoff = now - window_seconds

        # Count events in window
        events_in_window = sum(
            delta for ts, delta in window
            if ts >= cutoff
        )

        # Calculate actual window size
        oldest_ts = min(ts for ts, _ in window if ts >= cutoff) if events_in_window > 0 else now
        actual_window = now - oldest_ts

        if actual_window == 0:
            return 0.0

        return events_in_window / actual_window

    def get_all_metrics(self) -> dict:
        """
        Get all metrics as dict.

        Returns:
            Dict with histograms, gauges, counters, and rates
        """
        return {
            "histograms": {
                name: self.get_histogram_stats(name)
                for name in self.histograms
            },
            "gauges": dict(self.gauges),
            "counters": dict(self.counters),
            "rates": {
                "redis_ops_per_sec": self.get_rate("redis_ops"),
            },
        }

    def reset_counters(self) -> None:
        """Reset all counters to zero."""
        for key in self.counters:
            self.counters[key] = 0

    def get_summary(self) -> dict:
        """
        Get human-readable summary of key metrics.

        Returns:
            Dict with selected key metrics
        """
        tick_stats = self.get_histogram_stats("awareness_tick_ms")

        return {
            "presence": round(self.gauges["presence_scalar"], 3),
            "novelty": round(self.gauges["novelty"], 3),
            "tick_ms": {
                "mean": round(tick_stats.get("mean", 0), 1),
                "p95": round(tick_stats.get("p95", 0), 1),
            },
            "events_dropped": self.counters["events_dropped"],
            "cache_hit_rate": self._calculate_cache_hit_rate(),
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate embedding cache hit rate."""
        hits = self.counters["embedding_cache_hits"]
        misses = self.counters["embedding_cache_misses"]
        total = hits + misses

        if total == 0:
            return 0.0

        return hits / total


# Global metrics instance
_metrics = AwarenessMetrics()


def get_metrics() -> AwarenessMetrics:
    """Get global metrics instance."""
    return _metrics


# Convenience functions
def record_tick_time(duration_ms: float) -> None:
    """Record awareness tick duration."""
    _metrics.record_histogram("awareness_tick_ms", duration_ms)


def record_introspection_time(duration_ms: float) -> None:
    """Record introspection latency."""
    _metrics.record_histogram("introspection_latency_ms", duration_ms)


def update_presence_gauges(
    scalar: float,
    novelty: float,
    sim_self_live: float,
    sim_self_origin: float,
    sim_prev: float,
    entropy: float,
    coherence_drop: float = 0.0
) -> None:
    """Update all presence-related gauges."""
    _metrics.set_gauge("presence_scalar", scalar)
    _metrics.set_gauge("novelty", novelty)
    _metrics.set_gauge("sim_self_live", sim_self_live)
    _metrics.set_gauge("sim_self_origin", sim_self_origin)
    _metrics.set_gauge("sim_prev", sim_prev)
    _metrics.set_gauge("entropy", entropy)
    _metrics.set_gauge("coherence_drop", coherence_drop)


def increment_events_dropped(count: int = 1) -> None:
    """Increment dropped events counter."""
    _metrics.increment_counter("events_dropped", count)


def increment_snapshot_errors() -> None:
    """Increment snapshot error counter."""
    _metrics.increment_counter("snapshot_errors")


def increment_redis_ops(count: int = 1) -> None:
    """Increment Redis operations counter."""
    _metrics.increment_counter("redis_ops", count)
