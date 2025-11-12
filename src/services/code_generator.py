"""Code generation service with caching, validation, and policy enforcement."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol
import hashlib
import json
import os
import tempfile
import subprocess
import logging

logger = logging.getLogger(__name__)

# Prompt versions for cache invalidation
PROMPT_VERSION = "codeagent5_v2.0"  # Updated to CodeAgent-5 template
TEST_PROMPT_VERSION = "codeagent5_test_v2.0"

# Policy: forbidden imports and builtins
FORBIDDEN_IMPORTS = {
    "subprocess", "os.system", "os.popen", "pickle", "dill", "marshal", "ctypes", "cffi",
    "socket", "requests", "httpx", "urllib", "aiohttp", "ftplib", "paramiko",
    "pexpect", "resource", "signal", "multiprocessing", "pty", "importlib",
}
FORBIDDEN_BUILTINS = {"eval", "exec", "compile", "__import__"}

# Policy: size limits
MAX_BYTES = {"implementation": 40_000, "test": 24_000}
MAX_LINES = {"implementation": 400, "test": 250}


@dataclass
class GenRequest:
    """Code generation request."""
    goal_text: str
    context: Dict[str, Any]
    file_path: str
    role: str  # "implementation" or "test"
    language: str = "python"


@dataclass
class GenResult:
    """Code generation result."""
    code: str
    cache_hit: bool
    diagnostics: Dict[str, Any]


class LLMClient(Protocol):
    """Protocol for LLM clients."""
    async def complete(self, prompt: str, system: str, temperature: float = 0.2) -> str: ...


class LLMServiceWrapper:
    """Wrapper to adapt existing LLMService to async LLMClient protocol."""

    def __init__(self, llm_service):
        """Initialize wrapper.

        Args:
            llm_service: LLMService instance (from src.services.llm)
        """
        self.llm_service = llm_service

    async def complete(self, prompt: str, system: str, temperature: float = 0.2) -> str:
        """Generate completion using LLMService.

        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature (note: LLMService uses its own temp)

        Returns:
            Generated code
        """
        # LLMService is synchronous, just call it directly
        # (The OpenAI client it uses is sync, not async)
        return self.llm_service.generate_response(
            prompt=prompt,
            system_prompt=system,
            memories=None,
            include_self_awareness=False  # Don't include persona for code gen
        )


class CodeGenerator:
    """Generates code using LLM with caching, validation, and policy enforcement."""

    def __init__(self, llm: LLMClient, cache_dir: str = ".cache/codegen"):
        """Initialize code generator.

        Args:
            llm: LLM client for code generation
            cache_dir: Directory for content-addressable code cache
        """
        self.llm = llm
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Initialized CodeGenerator with cache at {cache_dir}")

    def _key(self, req: GenRequest) -> str:
        """Generate idempotency key for request.

        Key includes: goal, file path, role, context version, and prompt version.
        """
        pv = PROMPT_VERSION if req.role == "implementation" else TEST_PROMPT_VERSION
        payload = {
            "goal": req.goal_text,
            "file": req.file_path,
            "role": req.role,
            "ctx": req.context.get("version", 0),
            "prompt_version": pv,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _cache_path(self, key: str) -> str:
        """Get cache file path for key."""
        return os.path.join(self.cache_dir, f"{key}.code")

    def _sha256(self, text: str) -> str:
        """Compute SHA256 hash of text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _too_large(self, code: str, role: str) -> bool:
        """Check if code exceeds size limits."""
        return len(code.encode()) > MAX_BYTES[role] or code.count("\n") + 1 > MAX_LINES[role]

    def _forbidden_used(self, code: str) -> list[str]:
        """Check for forbidden imports and builtins in code.

        Returns:
            List of forbidden items found in code
        """
        hits = []
        # Check for actual import statements, not just substring presence
        lines = code.split('\n')
        for item in FORBIDDEN_IMPORTS:
            for line in lines:
                stripped = line.strip()
                # Match: "import foo" or "from foo import" or "import foo.bar"
                if (stripped.startswith(f"import {item}") or
                    stripped.startswith(f"from {item} ") or
                    f"import {item}." in stripped):
                    hits.append(item)
                    break

        # Check for forbidden builtins (keep simple check)
        for b in FORBIDDEN_BUILTINS:
            if f"{b}(" in code:
                hits.append(b)
        return sorted(set(hits))

    async def generate(self, req: GenRequest) -> GenResult:
        """Generate code for request with caching and validation.

        Args:
            req: Generation request

        Returns:
            GenResult with generated code and diagnostics

        Raises:
            RuntimeError: If code violates policy after retry
        """
        key = self._key(req)
        cp = self._cache_path(key)

        # Check cache
        if os.path.exists(cp):
            logger.info(f"Cache hit for {req.role} {req.file_path} (key={key[:16]}...)")
            with open(cp, "r", encoding="utf-8") as f:
                return GenResult(code=f.read(), cache_hit=True, diagnostics={"key": key})

        logger.info(f"Generating {req.role} for {req.file_path} (key={key[:16]}...)")

        # Generate code
        prompt, system = self._build_prompts(req)
        raw = await self.llm.complete(prompt=prompt, system=system, temperature=0.1)
        code = self._extract_code(raw)

        # Policy gate 1: size
        if self._too_large(code, req.role):
            logger.warning(f"Code too large ({len(code)} bytes, {code.count('\n')+1} lines), retrying")
            hint = f"Output exceeded size limits ({MAX_BYTES[req.role]} bytes or {MAX_LINES[req.role]} lines)."
            raw = await self.llm.complete(
                prompt=f"{prompt}\n\n{hint}\nRegenerate smaller, adhere to limits.",
                system=system,
                temperature=0.1
            )
            code = self._extract_code(raw)

        # Policy gate 2: forbidden APIs
        bad = self._forbidden_used(code)
        if bad:
            logger.warning(f"Forbidden APIs detected: {bad}, retrying")
            raw = await self.llm.complete(
                prompt=f"{prompt}\n\nRemove forbidden APIs: {bad}. Regenerate compliant code only.",
                system=system,
                temperature=0.1
            )
            code = self._extract_code(raw)

        # Policy gate 3: syntax and style
        ok, diags = self._lint_and_typecheck(code, req.language)
        if not ok:
            logger.warning(f"Syntax/type check failed: {diags}, retrying")
            retry_prompt = f"{prompt}\n\nPrevious attempt failed checks:\n{diags}\nRegenerate corrected code only."
            raw = await self.llm.complete(prompt=retry_prompt, system=system, temperature=0.1)
            code = self._extract_code(raw)

        # Final validation: if still bad, raise error
        if self._too_large(code, req.role) or self._forbidden_used(code):
            logger.error("Code generation policy violation after retry")
            raise RuntimeError("Codegen policy violation after retry")

        # Cache and return
        with open(cp, "w", encoding="utf-8") as f:
            f.write(code)

        return GenResult(
            code=code,
            cache_hit=False,
            diagnostics={"key": key, "checks": diags if not ok else {}}
        )

    def _build_prompts(self, req: GenRequest) -> tuple[str, str]:
        """Build prompt and system message using CodeAgent-5 template.

        Args:
            req: Generation request

        Returns:
            (prompt, system) tuple
        """
        if req.role == "implementation":
            return self._build_implementation_prompt(req)
        else:
            return self._build_test_prompt(req)

    def _build_implementation_prompt(self, req: GenRequest) -> tuple[str, str]:
        """Build CodeAgent-5 implementation prompt."""
        # Extract context
        codebase_context = req.context.get("codebase_context", "No additional context")
        similar_patterns = req.context.get("similar_patterns", "")
        dependencies = req.context.get("dependencies", [])

        system = "You are CodeAgent-5, a senior software engineer. Produce production-grade code only."

        prompt = f"""<ROLE>
You are CodeAgent-5, a senior software engineer specializing in Python. Your objective is:
Implement {req.goal_text}
</ROLE>

<CONTEXT>
Target file: {req.file_path}
Language: {req.language}
Runtime: Python 3.11
Build system: pytest for tests

Codebase context:
{codebase_context}

{f"Similar patterns in codebase:\\n{similar_patterns}" if similar_patterns else ""}

Available dependencies: {', '.join(dependencies) if dependencies else 'Standard library only'}
</CONTEXT>

<DELIVERABLES>
- Working implementation of: {req.goal_text}
- Production-ready with type hints, docstrings, and error handling
- Must return JSON with keys: filename, language, code
</DELIVERABLES>

<REQUIREMENTS>
<SCOPE>
Functional requirements:
- Implement {req.goal_text}
- Include comprehensive docstrings with Args, Returns, Raises sections
- Add type hints for all function signatures
- Handle errors gracefully with specific exception types
- Keep implementation focused and minimal (under 150 lines)

Performance requirements:
- O(n) or better time complexity where applicable
- No blocking operations unless explicitly required
- Memory efficient (no large in-memory caches without bounds)
</SCOPE>

<CONSTRAINTS>
- Language: Python {req.language}
- Max size: {MAX_LINES['implementation']} lines, {MAX_BYTES['implementation']} bytes
- FORBIDDEN imports: {', '.join(sorted(list(FORBIDDEN_IMPORTS)[:10]))}... (and others)
- FORBIDDEN builtins: {', '.join(sorted(FORBIDDEN_BUILTINS))}
- No network calls unless explicitly required by goal
- No file system writes unless explicitly required by goal
</CONSTRAINTS>

<FORMAT>
Return JSON with exactly these keys:
{{
  "filename": "{os.path.basename(req.file_path)}",
  "language": "{req.language}",
  "code": "...full implementation as string with actual newlines..."
}}

Do NOT use escaped newlines like \\n - use actual newlines in the JSON string value.
</FORMAT>

<EVALUATION_CRITERIA>
- Correctness: Implements the goal specification exactly
- Completeness: Handles edge cases and errors
- Security: Validates inputs, no injection risks
- Performance: Efficient algorithms and data structures
- Readability: Clear names, good structure, helpful comments
- Pythonic: Follows PEP 8, uses standard library idioms
</EVALUATION_CRITERIA>
</REQUIREMENTS>

<QUALITY_GUARDS>
- Security: Validate all inputs, use type checking, avoid injection
- Errors: Never swallow exceptions silently - log or re-raise
- Performance: Use generators for large sequences, avoid O(n²) unless necessary
- Logging: Include informative log messages at appropriate levels
- Documentation: Every public function needs a docstring
</QUALITY_GUARDS>

<OUTPUT>
Return ONLY the JSON object. No markdown fences, no explanations.

Example format:
{{
  "filename": "example.py",
  "language": "python",
  "code": "import logging\\n\\nlogger = logging.getLogger(__name__)\\n\\ndef process_data(items: list[str]) -> dict[str, int]:\\n    \\"\\"\\"Process items and return counts.\\n    \\n    Args:\\n        items: List of items to process\\n        \\n    Returns:\\n        Dictionary mapping items to counts\\n    \\"\\"\\"\\n    return {{item: items.count(item) for item in set(items)}}"
}}
</OUTPUT>"""

        return prompt.strip(), system

    def _build_test_prompt(self, req: GenRequest) -> tuple[str, str]:
        """Build CodeAgent-5 test prompt."""
        impl_code = req.context.get("implementation_code", "")
        impl_preview = impl_code[:1000] if impl_code else "No implementation provided"

        system = "You are CodeAgent-5, a senior software engineer specializing in test engineering. Produce comprehensive test code."

        prompt = f"""<ROLE>
You are CodeAgent-5, a test engineering specialist. Your objective is:
Create comprehensive tests for {req.file_path}
</ROLE>

<CONTEXT>
Target test file: {req.file_path}
Language: Python 3.11
Test framework: pytest
Coverage target: Critical paths and edge cases

Implementation being tested:
```python
{impl_preview}
```
</CONTEXT>

<DELIVERABLES>
- pytest test suite covering critical functionality
- Tests for edge cases, errors, and boundary conditions
- Must return JSON with keys: filename, language, code
</DELIVERABLES>

<REQUIREMENTS>
<SCOPE>
Test requirements:
- Test normal operation (happy path)
- Test edge cases (empty inputs, None, large values)
- Test error conditions (invalid inputs, exceptions)
- Test boundary conditions (min/max values, limits)
- Use pytest fixtures where appropriate
- Keep under 120 lines

Test structure:
- One test class or multiple test functions
- Clear, descriptive test names (test_function_name_condition_expected_behavior)
- Use assert statements with helpful messages
- Mock external dependencies if any
</SCOPE>

<CONSTRAINTS>
- Framework: pytest only
- Max size: {MAX_LINES['test']} lines, {MAX_BYTES['test']} bytes
- No forbidden imports or builtins
- No network calls or file system access (use mocks)
</CONSTRAINTS>

<FORMAT>
Return JSON with exactly these keys:
{{
  "filename": "{os.path.basename(req.file_path)}",
  "language": "{req.language}",
  "code": "...full test code as string with actual newlines..."
}}

Do NOT use escaped newlines like \\n - use actual newlines in the JSON string value.
</FORMAT>

<EVALUATION_CRITERIA>
- Coverage: All critical paths tested
- Quality: Tests are clear, focused, and deterministic
- Robustness: Tests catch real bugs, not just exercise code
- Maintainability: Easy to understand what's being tested
</EVALUATION_CRITERIA>
</REQUIREMENTS>

<OUTPUT>
Return ONLY the JSON object. No markdown fences, no explanations.

Example format:
{{
  "filename": "test_example.py",
  "language": "python",
  "code": "import pytest\\nfrom src.example import process_data\\n\\ndef test_process_data_empty_list():\\n    \\"\\"\\"Test processing empty list.\\"\\"\\"\\n    result = process_data([])\\n    assert result == {{}}, \\"Expected empty dict for empty list\\""
}}
</OUTPUT>"""

        return prompt.strip(), system

    def _extract_code(self, raw: str) -> str:
        """Extract code from LLM response.

        Args:
            raw: Raw LLM response

        Returns:
            Extracted code string
        """
        try:
            obj = json.loads(raw)
            code = obj.get("code", "")
            # Handle escaped newlines if LLM returned them as literal strings
            # This happens when the LLM returns "import foo\\n\\ndef bar():" instead of actual newlines
            if "\\n" in code and "\n" not in code:
                code = code.replace("\\n", "\n").replace("\\t", "\t")
            return code
        except Exception:
            # Fallback: try to find JSON in response
            import re
            match = re.search(r'\{.*"code"\s*:\s*"(.*?)"\s*\}', raw, re.DOTALL)
            if match:
                code = match.group(1)
                # Handle escaped newlines
                if "\\n" in code and "\n" not in code:
                    code = code.replace("\\n", "\n").replace("\\t", "\t")
                return code
            # Last resort: return raw
            logger.warning("Failed to extract JSON, returning raw response")
            return raw

    def _lint_and_typecheck(self, code: str, language: str) -> tuple[bool, Dict[str, Any]]:
        """Lint and typecheck generated code.

        Args:
            code: Code to check
            language: Programming language

        Returns:
            (ok, diagnostics) tuple
        """
        if language != "python":
            return True, {}

        import sys

        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "snippet.py")
            with open(fp, "w", encoding="utf-8") as f:
                f.write(code)

            diags = {}

            # Use current Python interpreter (works in venv)
            python_exe = sys.executable

            def run(cmd):
                p = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=10
                )
                return p.returncode, p.stdout

            # Syntax check with pyflakes
            try:
                _, out1 = run([python_exe, "-m", "pyflakes", fp])
                diags["pyflakes"] = out1
            except Exception as e:
                logger.warning(f"pyflakes check failed: {e}")
                diags["pyflakes"] = str(e)

            # Format with black
            try:
                _, out2 = run([python_exe, "-m", "black", "--quiet", fp])
                # Read formatted code back
                with open(fp, "r", encoding="utf-8") as f:
                    formatted = f.read()
            except Exception as e:
                logger.warning(f"black format failed: {e}")

            # Type check with mypy
            try:
                rc3, out3 = run([python_exe, "-m", "mypy", "--ignore-missing-imports", fp])
                diags["mypy"] = out3
                diags["mypy_rc"] = rc3
            except Exception as e:
                logger.warning(f"mypy check failed: {e}")
                diags["mypy"] = str(e)
                rc3 = 1

            # Check passes if no critical syntax errors
            # Allow minor mypy warnings and unused imports - they're not critical
            pyflakes_output = diags.get("pyflakes", "")
            has_critical_pyflakes = any(
                keyword in pyflakes_output
                for keyword in ["SyntaxError", "IndentationError", "invalid syntax"]
            )

            # Consider code OK if:
            # - No critical syntax errors from pyflakes
            # - Mypy can at least parse it (rc != 2 which is syntax error)
            ok = not has_critical_pyflakes and rc3 != 2

            return ok, diags
