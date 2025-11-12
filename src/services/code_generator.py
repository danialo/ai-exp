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
PROMPT_VERSION = "impl_v1.0"
TEST_PROMPT_VERSION = "test_v1.0"

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
        # Cheap string-based checks (AST parse could be added later for precision)
        for item in FORBIDDEN_IMPORTS:
            if item in code:
                hits.append(item)
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
        """Build prompt and system message for generation.

        Args:
            req: Generation request

        Returns:
            (prompt, system) tuple
        """
        system = "You produce minimal, production-grade code. Respond with a JSON object containing fields filename, language, and code. No prose."

        if req.role == "implementation":
            prompt = f"""
Return JSON with exactly these keys: filename, language, code.

Context:
- Goal: {req.goal_text}
- Target file: {req.file_path}
- Language: {req.language}

Requirements:
- Pure function or small module with docstring, type hints, and errors handled
- No network calls, no file writes unless explicitly required
- Keep under 150 lines
- No forbidden imports: {', '.join(sorted(FORBIDDEN_IMPORTS))}
- No forbidden builtins: {', '.join(sorted(FORBIDDEN_BUILTINS))}
"""
        else:  # test
            impl_hint = req.context.get("implementation_code", "")
            prompt = f"""
Return JSON with exactly these keys: filename, language, code.

Generate tests for the implementation at {req.file_path}.

Requirements:
- Use pytest
- Cover edge cases and error paths
- Keep under 120 lines
- No forbidden imports or builtins
"""
            if impl_hint:
                prompt += f"\n\nImplementation to test:\n{impl_hint[:500]}..."

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

            # Check passes if no pyflakes errors and mypy passes
            ok = rc3 == 0 and not diags.get("pyflakes", "").strip()

            return ok, diags
