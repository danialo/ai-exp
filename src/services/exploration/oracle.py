"""Oracle system for verifying exploration job acceptance criteria."""

import importlib
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

from .models import OracleConfig, OracleResult, OracleType

logger = logging.getLogger(__name__)


class Oracle:
    """Acceptance test oracle for exploration jobs."""

    def __init__(self, config: OracleConfig, workspace_dir: Path):
        """Initialize oracle.

        Args:
            config: Oracle configuration
            workspace_dir: Path to job workspace
        """
        self.config = config
        self.workspace_dir = workspace_dir
        self.llm_service = None  # Will be injected if needed

    def set_llm_service(self, llm_service):
        """Inject LLM service for LLM-based oracles.

        Args:
            llm_service: LLM service instance
        """
        self.llm_service = llm_service

    async def check(self) -> OracleResult:
        """Run the oracle check.

        Returns:
            OracleResult with pass/fail status and details
        """
        start_time = time.time()

        try:
            if self.config.type == OracleType.SCRIPT:
                result = await self._check_script()
            elif self.config.type == OracleType.PYTHON:
                result = await self._check_python()
            elif self.config.type == OracleType.LLM:
                result = await self._check_llm()
            else:
                result = OracleResult(
                    ok=False,
                    detail={"error": f"Unknown oracle type: {self.config.type}"}
                )

            result.duration_ms = int((time.time() - start_time) * 1000)
            return result

        except Exception as e:
            logger.error(f"Oracle check failed: {e}")
            return OracleResult(
                ok=False,
                detail={"error": str(e)},
                duration_ms=int((time.time() - start_time) * 1000)
            )

    async def _check_script(self) -> OracleResult:
        """Run script-based oracle (exit code verification).

        Returns:
            OracleResult
        """
        if not self.config.cmd:
            return OracleResult(
                ok=False,
                detail={"error": "Script oracle requires 'cmd' parameter"}
            )

        try:
            # Run script in workspace directory
            process = subprocess.run(
                self.config.cmd,
                shell=True,
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_sec
            )

            # Truncate output to reasonable size
            max_output = 10000
            stdout = process.stdout[:max_output]
            stderr = process.stderr[:max_output]

            return OracleResult(
                ok=(process.returncode == 0),
                detail={
                    "exit_code": process.returncode,
                    "command": self.config.cmd
                },
                stdout=stdout,
                stderr=stderr,
                exit_code=process.returncode
            )

        except subprocess.TimeoutExpired:
            return OracleResult(
                ok=False,
                detail={
                    "error": f"Script timeout after {self.config.timeout_sec}s"
                }
            )
        except Exception as e:
            return OracleResult(
                ok=False,
                detail={"error": f"Script execution failed: {str(e)}"}
            )

    async def _check_python(self) -> OracleResult:
        """Run Python callable oracle.

        Returns:
            OracleResult
        """
        if not self.config.callable:
            return OracleResult(
                ok=False,
                detail={"error": "Python oracle requires 'callable' parameter"}
            )

        try:
            # Parse dotted path (e.g., "verifiers.jokes:check")
            if ":" not in self.config.callable:
                return OracleResult(
                    ok=False,
                    detail={"error": "Callable must be in format 'module:function'"}
                )

            module_path, func_name = self.config.callable.split(":", 1)

            # Import module
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)

            # Call function with workspace_dir as argument
            result = func(str(self.workspace_dir))

            if isinstance(result, bool):
                return OracleResult(
                    ok=result,
                    detail={"callable": self.config.callable}
                )
            elif isinstance(result, dict):
                # Function can return dict with ok + details
                return OracleResult(
                    ok=result.get("ok", False),
                    detail=result.get("detail", {}),
                    stdout=result.get("stdout", ""),
                    stderr=result.get("stderr", "")
                )
            else:
                return OracleResult(
                    ok=False,
                    detail={"error": f"Callable must return bool or dict, got {type(result)}"}
                )

        except ImportError as e:
            return OracleResult(
                ok=False,
                detail={"error": f"Failed to import module: {str(e)}"}
            )
        except AttributeError as e:
            return OracleResult(
                ok=False,
                detail={"error": f"Function not found: {str(e)}"}
            )
        except Exception as e:
            return OracleResult(
                ok=False,
                detail={"error": f"Callable execution failed: {str(e)}"}
            )

    async def _check_llm(self) -> OracleResult:
        """Run LLM-based oracle verification.

        Returns:
            OracleResult
        """
        if not self.config.rubric:
            return OracleResult(
                ok=False,
                detail={"error": "LLM oracle requires 'rubric' parameter"}
            )

        if not self.llm_service:
            return OracleResult(
                ok=False,
                detail={"error": "LLM oracle requires LLM service"}
            )

        try:
            # Collect workspace state
            workspace_files = []
            for file_path in self.workspace_dir.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    rel_path = file_path.relative_to(self.workspace_dir)
                    workspace_files.append(str(rel_path))

            # Build verification prompt
            prompt = f"""Verify if the workspace meets the acceptance criteria.

Workspace directory: {self.workspace_dir}
Files present: {', '.join(workspace_files)}

Acceptance criteria:
{chr(10).join(f"- {criterion}" for criterion in self.config.rubric)}

For each criterion, check if it's met. Respond with:
- "PASS" if all criteria are met
- "FAIL" if any criterion is not met

Include a brief explanation for each criterion."""

            # Call LLM
            response = self.llm_service.generate(
                prompt=prompt,
                max_tokens=self.config.max_tokens or 1000
            )

            # Parse response
            response_lower = response.lower()
            passed = "pass" in response_lower and "fail" not in response_lower

            return OracleResult(
                ok=passed,
                detail={
                    "rubric": self.config.rubric,
                    "llm_verdict": response
                },
                stdout=response
            )

        except Exception as e:
            return OracleResult(
                ok=False,
                detail={"error": f"LLM oracle failed: {str(e)}"}
            )


def create_oracle(config: OracleConfig, workspace_dir: Path) -> Oracle:
    """Factory function to create an Oracle instance.

    Args:
        config: Oracle configuration
        workspace_dir: Path to job workspace

    Returns:
        Configured Oracle instance
    """
    return Oracle(config, workspace_dir)
