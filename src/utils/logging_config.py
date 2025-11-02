"""Centralized logging configuration with separate log files for different event types."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
from datetime import datetime


class MultiFileLogger:
    """Manages multiple rotating log files for different event types."""

    def __init__(self, base_dir: str = "logs"):
        self.base_dir = Path(base_dir)
        self.loggers = {}
        self._setup_loggers()

    def _create_rotating_handler(
        self,
        log_file: Path,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB default
        backup_count: int = 5
    ) -> RotatingFileHandler:
        """Create a rotating file handler."""
        handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        return handler

    def _setup_loggers(self):
        """Setup all specialized loggers."""

        # 1. Application logs (general system events)
        app_dir = self.base_dir / "app"
        app_dir.mkdir(parents=True, exist_ok=True)

        app_logger = logging.getLogger("astra.app")
        app_logger.setLevel(logging.INFO)
        app_logger.propagate = False
        app_logger.addHandler(self._create_rotating_handler(app_dir / "application.log", max_bytes=20*1024*1024))
        self.loggers['app'] = app_logger

        # 2. Conversation logs (user-Astra chat)
        conv_dir = self.base_dir / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)

        conv_logger = logging.getLogger("astra.conversation")
        conv_logger.setLevel(logging.INFO)
        conv_logger.propagate = False
        conv_logger.addHandler(self._create_rotating_handler(conv_dir / "conversations.log", max_bytes=50*1024*1024, backup_count=10))
        self.loggers['conversation'] = conv_logger

        # 3. Error logs (errors only, from all systems)
        error_dir = self.base_dir / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)

        error_logger = logging.getLogger("astra.error")
        error_logger.setLevel(logging.ERROR)
        error_logger.propagate = False
        error_logger.addHandler(self._create_rotating_handler(error_dir / "errors.log", max_bytes=10*1024*1024))
        self.loggers['error'] = error_logger

        # 4. Tool calls (when Astra uses tools like search, file ops, etc)
        tools_dir = self.base_dir / "tools"
        tools_dir.mkdir(parents=True, exist_ok=True)

        tools_logger = logging.getLogger("astra.tools")
        tools_logger.setLevel(logging.INFO)
        tools_logger.propagate = False
        tools_logger.addHandler(self._create_rotating_handler(tools_dir / "tool_calls.log", max_bytes=20*1024*1024))
        self.loggers['tools'] = tools_logger

        # 5. Memory retrieval (memory searches and fetches)
        memory_dir = self.base_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        memory_logger = logging.getLogger("astra.memory")
        memory_logger.setLevel(logging.INFO)
        memory_logger.propagate = False
        memory_logger.addHandler(self._create_rotating_handler(memory_dir / "memory_retrieval.log", max_bytes=30*1024*1024))
        self.loggers['memory'] = memory_logger

        # 6. Belief system (belief queries, updates, dissonance)
        belief_dir = self.base_dir / "beliefs"
        belief_dir.mkdir(parents=True, exist_ok=True)

        belief_logger = logging.getLogger("astra.beliefs")
        belief_logger.setLevel(logging.INFO)
        belief_logger.propagate = False
        belief_logger.addHandler(self._create_rotating_handler(belief_dir / "belief_system.log", max_bytes=20*1024*1024))
        self.loggers['beliefs'] = belief_logger

        # 7. Awareness loop (introspection, percepts, observations)
        awareness_dir = self.base_dir / "awareness"
        awareness_dir.mkdir(parents=True, exist_ok=True)

        awareness_logger = logging.getLogger("astra.awareness")
        awareness_logger.setLevel(logging.INFO)
        awareness_logger.propagate = False
        awareness_logger.addHandler(self._create_rotating_handler(awareness_dir / "awareness_loop.log", max_bytes=15*1024*1024))
        self.loggers['awareness'] = awareness_logger

        # 8. Performance metrics (timing, token usage, costs)
        perf_dir = self.base_dir / "performance"
        perf_dir.mkdir(parents=True, exist_ok=True)

        perf_logger = logging.getLogger("astra.performance")
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False
        perf_logger.addHandler(self._create_rotating_handler(perf_dir / "performance.log", max_bytes=10*1024*1024))
        self.loggers['performance'] = perf_logger

    def get_logger(self, log_type: str) -> logging.Logger:
        """Get a specialized logger by type."""
        return self.loggers.get(log_type, self.loggers['app'])

    # Convenience methods for common logging operations

    def log_conversation(self, role: str, message: str, metadata: Optional[dict] = None):
        """Log a conversation message."""
        logger = self.loggers['conversation']
        msg = f"[{role.upper()}] {message}"
        if metadata:
            msg += f" | Metadata: {metadata}"
        logger.info(msg)

    def log_tool_call(self, tool_name: str, arguments: dict, result: Optional[str] = None, duration_ms: Optional[float] = None):
        """Log a tool call."""
        logger = self.loggers['tools']
        msg = f"Tool: {tool_name} | Args: {arguments}"
        if duration_ms:
            msg += f" | Duration: {duration_ms:.2f}ms"
        logger.info(msg)

        if result:
            result_preview = result[:500] + "..." if len(result) > 500 else result
            logger.info(f"  Result: {result_preview}")

    def log_memory_retrieval(self, query: str, count: int, retrieval_type: str = "similar", duration_ms: Optional[float] = None):
        """Log memory retrieval."""
        logger = self.loggers['memory']
        msg = f"Query: {query} | Type: {retrieval_type} | Count: {count}"
        if duration_ms:
            msg += f" | Duration: {duration_ms:.2f}ms"
        logger.info(msg)

    def log_belief_event(self, event_type: str, belief_statement: Optional[str] = None, details: Optional[dict] = None):
        """Log belief system events."""
        logger = self.loggers['beliefs']
        msg = f"Event: {event_type}"
        if belief_statement:
            msg += f" | Belief: {belief_statement}"
        if details:
            msg += f" | Details: {details}"
        logger.info(msg)

    def log_awareness_event(self, event_type: str, data: Optional[dict] = None):
        """Log awareness loop events."""
        logger = self.loggers['awareness']
        msg = f"Event: {event_type}"
        if data:
            msg += f" | Data: {data}"
        logger.info(msg)

    def log_performance(self, operation: str, duration_ms: float, tokens_used: Optional[int] = None, cost: Optional[float] = None):
        """Log performance metrics."""
        logger = self.loggers['performance']
        msg = f"Op: {operation} | Duration: {duration_ms:.2f}ms"
        if tokens_used:
            msg += f" | Tokens: {tokens_used}"
        if cost:
            msg += f" | Cost: ${cost:.6f}"
        logger.info(msg)

    def log_error(self, error_msg: str, exception: Optional[Exception] = None, context: Optional[dict] = None):
        """Log an error."""
        logger = self.loggers['error']
        msg = f"Error: {error_msg}"
        if context:
            msg += f" | Context: {context}"
        logger.error(msg)

        if exception:
            logger.exception(exception)


# Global instance
_multi_logger = None


def get_multi_logger() -> MultiFileLogger:
    """Get or create the global multi-file logger instance."""
    global _multi_logger
    if _multi_logger is None:
        _multi_logger = MultiFileLogger()
    return _multi_logger
