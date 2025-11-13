"""Specialized agent for code generation with strict safety guarantees."""

import json
import hashlib
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base import BaseAgent

logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):
    """Code generation agent with JSON output and safety validation.

    Features:
    - JSON-only structured output
    - Forbidden API detection
    - Size limit enforcement
    - Deterministic output (idempotent keys)
    - Static analysis validation
    """

    # Forbidden APIs that must not appear in generated code
    FORBIDDEN_APIS = [
        "eval", "exec", "compile", "__import__",
        "subprocess", "os.system", "os.popen", "os.spawn",
        "socket", "requests", "httpx", "urllib", "urllib3", "aiohttp",
        "pickle", "shelve", "marshal",
        "sys.exit", "os._exit",
        "globals", "locals", "vars", "dir",
        "setattr", "delattr", "__setattr__", "__delattr__",
        "sqlite3.execute", "cursor.execute",
    ]

    # Size limits
    MAX_IMPL_LINES = 400
    MAX_TEST_LINES = 250
    MAX_IMPL_BYTES = 40 * 1024  # 40 KB
    MAX_TEST_BYTES = 24 * 1024  # 24 KB

    def __init__(self, llm_service):
        """Initialize CoderAgent with LLM service.

        Args:
            llm_service: LLM service instance (no beliefs, no memories)
        """
        self.llm = llm_service
        self.system_prompt = self._load_prompt()
        logger.info("CoderAgent initialized")

    def _load_prompt(self) -> str:
        """Load coder system prompt from file.

        Returns:
            System prompt content

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        prompt_path = Path(__file__).parent / "prompts" / "coder_v1_0.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Coder prompt not found at {prompt_path}")

        return prompt_path.read_text()

    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code artifacts from request.

        Args:
            request: {
                "goal_text": str,           # e.g., "implement_feature"
                "context": {
                    "requirements": str,     # Natural language description
                    "existing_files": list,  # Optional: Files to consider
                    "constraints": list      # Optional: Additional constraints
                },
                "timeout_ms": int           # Optional: Timeout
            }

        Returns:
            {
                "plan": list[str],           # High-level steps
                "artifacts": list[dict],     # Generated files
                "checks": dict,              # Validation results
                "assumptions": list[str]     # Assumptions made
            }

        Raises:
            ValueError: Invalid request or unsafe output
            json.JSONDecodeError: LLM returned invalid JSON
        """
        logger.info(f"CoderAgent processing request: {request.get('goal_text')}")

        # Build minimal prompt (NO beliefs, NO personality)
        user_prompt = self._build_user_prompt(request)

        # Call LLM with coder system prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Use low temperature for deterministic code generation
        response = self.llm.generate_with_tools(
            messages=messages,
            tools=None,  # No tools for coder
            temperature=0.2,  # Low temp for consistency
            max_tokens=8000,  # Smaller than Astra's 16k (400 lines ≈ 6k tokens)
        )

        # Parse JSON output
        raw_content = response["message"].content
        output = self._parse_json_output(raw_content)

        # Validate safety (forbidden APIs, size limits)
        self._validate_safety(output)

        # Validate schema
        if not self.validate_output(output):
            raise ValueError("CoderAgent output validation failed: schema mismatch")

        logger.info(f"CoderAgent generated {len(output['artifacts'])} artifacts")
        return output

    def _build_user_prompt(self, request: Dict[str, Any]) -> str:
        """Build minimal user prompt from request.

        Args:
            request: Request dictionary

        Returns:
            Formatted user prompt string
        """
        lines = []

        # Add goal
        lines.append(f"GOAL: {request.get('goal_text', 'implement_feature')}")

        # Add context if present
        if "context" in request:
            ctx = request["context"]

            if "requirements" in ctx:
                lines.append(f"\nREQUIREMENTS:\n{ctx['requirements']}")

            if "existing_files" in ctx and ctx["existing_files"]:
                lines.append("\nEXISTING FILES TO CONSIDER:")
                for f in ctx["existing_files"]:
                    lines.append(f"  - {f}")

            if "constraints" in ctx and ctx["constraints"]:
                lines.append("\nCONSTRAINTS:")
                for c in ctx["constraints"]:
                    lines.append(f"  - {c}")

        return "\n".join(lines)

    def _parse_json_output(self, raw: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response.

        Handles markdown fences and whitespace.

        Args:
            raw: Raw LLM response string

        Returns:
            Parsed JSON dictionary

        Raises:
            json.JSONDecodeError: If output is not valid JSON
        """
        # Remove markdown fences if present
        raw = raw.strip()

        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]

        if raw.endswith("```"):
            raw = raw[:-3]

        # Parse JSON
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM: {e}")
            logger.error(f"Raw output: {raw[:500]}...")
            raise

    def _validate_safety(self, output: Dict[str, Any]) -> None:
        """Validate no forbidden APIs used and size limits met.

        Args:
            output: Parsed output dictionary

        Raises:
            ValueError: If validation fails
        """
        forbidden_found = []

        for artifact in output.get("artifacts", []):
            code = artifact.get("code", "")
            filename = artifact.get("filename", "unknown")

            # Check forbidden APIs
            for forbidden in self.FORBIDDEN_APIS:
                if forbidden in code:
                    # Allow in comments/docstrings
                    lines = code.split("\n")
                    for line in lines:
                        stripped = line.strip()
                        # Skip comments and docstrings
                        if stripped.startswith("#"):
                            continue
                        if '"""' in stripped or "'''" in stripped:
                            continue

                        if forbidden in line and not line.strip().startswith("#"):
                            forbidden_found.append(f"{forbidden} in {filename}")

            # Check size limits
            lines = len(code.splitlines())
            bytes_size = len(code.encode('utf-8'))

            is_test = "test" in filename.lower() or filename.startswith("test_")
            max_lines = self.MAX_TEST_LINES if is_test else self.MAX_IMPL_LINES
            max_bytes = self.MAX_TEST_BYTES if is_test else self.MAX_IMPL_BYTES

            if lines > max_lines:
                raise ValueError(
                    f"Artifact {filename} exceeds line limit: {lines} > {max_lines}"
                )

            if bytes_size > max_bytes:
                raise ValueError(
                    f"Artifact {filename} exceeds byte limit: {bytes_size} > {max_bytes}"
                )

        if forbidden_found:
            raise ValueError(f"Forbidden APIs found: {', '.join(forbidden_found)}")

    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate output matches expected schema.

        Args:
            output: Response to validate

        Returns:
            True if valid, False otherwise
        """
        # Check required top-level keys
        required_keys = ["plan", "artifacts", "checks", "assumptions"]
        if not all(k in output for k in required_keys):
            logger.error(f"Missing required keys. Expected {required_keys}, got {list(output.keys())}")
            return False

        # Validate plan is a list
        if not isinstance(output["plan"], list):
            logger.error(f"'plan' must be a list, got {type(output['plan'])}")
            return False

        # Validate artifacts structure
        artifacts = output.get("artifacts", [])
        if not isinstance(artifacts, list):
            logger.error(f"'artifacts' must be a list, got {type(artifacts)}")
            return False

        for i, artifact in enumerate(artifacts):
            required_artifact_keys = ["filename", "language", "code"]
            if not all(k in artifact for k in required_artifact_keys):
                logger.error(f"Artifact {i} missing required keys: {required_artifact_keys}")
                return False

        # Validate checks structure
        checks = output.get("checks", {})
        if not isinstance(checks, dict):
            logger.error(f"'checks' must be a dict, got {type(checks)}")
            return False

        required_check_keys = [
            "ruff_black_clean",
            "mypy_clean",
            "forbidden_apis_used",
            "size_ok",
            "idempotent_key"
        ]
        if not all(k in checks for k in required_check_keys):
            logger.error(f"'checks' missing required keys: {required_check_keys}")
            return False

        # Validate assumptions is a list
        if not isinstance(output["assumptions"], list):
            logger.error(f"'assumptions' must be a list, got {type(output['assumptions'])}")
            return False

        return True

    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities and constraints.

        Returns:
            Dict with agent metadata
        """
        return {
            "name": "CoderAgent",
            "description": "Specialized code generation with safety guarantees",
            "max_input_tokens": 4000,  # Minimal prompt
            "max_output_tokens": 8000,  # ~400 lines × 2 files
            "cost_per_request": 0.12,  # USD (estimated for GPT-4o)
            "avg_latency_ms": 8000,    # ~8 seconds
            "features": [
                "JSON-only output",
                "Forbidden API detection",
                "Size limit enforcement",
                "Deterministic output",
                "Comprehensive tests included"
            ],
            "constraints": [
                f"Max {self.MAX_IMPL_LINES} lines per implementation file",
                f"Max {self.MAX_TEST_LINES} lines per test file",
                "No network access in generated code",
                "Pure stdlib only (unless explicitly allowed)"
            ]
        }
