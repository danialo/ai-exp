"""
Persona File Manager

Provides controlled file system access for the persona to manage its own space.
The persona can create, read, update, and delete files within its designated area.
"""

import json
import os
import subprocess
import logging
import shlex
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

# Whitelist of allowed command interpreters
ALLOWED_COMMANDS = {
    'python', 'python3', 'python3.11', 'python3.10', 'python3.9',
    'bash', 'sh',
    'node', 'npm', 'npx',
    'pip', 'pip3',
    'git',
    'ls', 'cat', 'echo', 'pwd', 'which',
}


class PersonaFileManager:
    """Manages file operations for the persona's personal space."""

    def __init__(self, persona_space_path: str = "persona_space"):
        self.persona_space = Path(persona_space_path).absolute()
        self.source_code_path = Path("src").absolute()

        # Ensure persona space exists
        self.persona_space.mkdir(parents=True, exist_ok=True)

    def read_file(self, file_path: str) -> Optional[str]:
        """
        Read a file from persona space or source code (read-only).

        Args:
            file_path: Relative path from persona_space or absolute path

        Returns:
            File contents or None if file doesn't exist or access denied
        """
        resolved_path = self._resolve_path(file_path)

        if not resolved_path:
            return None

        try:
            return resolved_path.read_text()
        except Exception as e:
            return f"Error reading file: {e}"

    def write_file(self, file_path: str, content: str) -> bool:
        """
        Write a file to persona space.

        Args:
            file_path: Relative path within persona_space
            content: Content to write

        Returns:
            True if successful, False otherwise
        """
        resolved_path = self._resolve_path(file_path, write=True)

        if not resolved_path:
            return False

        # Protect core beliefs from modification
        if file_path == "beliefs.json" or file_path.endswith("/beliefs.json"):
            validation_result = self._validate_beliefs_modification(content)
            if not validation_result["allowed"]:
                logger.warning(f"Blocked attempt to modify core beliefs: {validation_result['reason']}")
                print(f"⚠️  Core Belief Protection: {validation_result['reason']}")
                print(f"   Core beliefs are immutable foundational axioms and cannot be changed.")
                print(f"   You can add, modify, or remove peripheral beliefs, but core beliefs are protected.")
                return False

        try:
            # Create parent directories if needed
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_text(content)
            return True
        except Exception as e:
            print(f"Error writing file: {e}")
            return False

    def list_files(self, directory: str = ".") -> List[str]:
        """
        List files in a directory within persona space.

        Args:
            directory: Relative path within persona_space

        Returns:
            List of file/directory names
        """
        dir_path = self.persona_space / directory

        if not self._is_safe_path(dir_path):
            return []

        try:
            if dir_path.is_dir():
                return [item.name for item in dir_path.iterdir()]
            return []
        except Exception:
            return []

    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from persona space.

        Args:
            file_path: Relative path within persona_space

        Returns:
            True if successful, False otherwise
        """
        resolved_path = self._resolve_path(file_path, write=True)

        if not resolved_path or not resolved_path.exists():
            return False

        try:
            if resolved_path.is_file():
                resolved_path.unlink()
                return True
            return False
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False

    def create_directory(self, dir_path: str) -> bool:
        """
        Create a directory in persona space.

        Args:
            dir_path: Relative path within persona_space

        Returns:
            True if successful, False otherwise
        """
        resolved_path = self.persona_space / dir_path

        if not self._is_safe_path(resolved_path):
            return False

        try:
            resolved_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory: {e}")
            return False

    def read_json(self, file_path: str) -> Optional[Dict]:
        """
        Read and parse a JSON file.

        Args:
            file_path: Relative path within persona_space

        Returns:
            Parsed JSON data or None
        """
        content = self.read_file(file_path)

        if not content:
            return None

        try:
            return json.loads(content)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return None

    def write_json(self, file_path: str, data: Dict) -> bool:
        """
        Write data to a JSON file.

        Args:
            file_path: Relative path within persona_space
            data: Data to write

        Returns:
            True if successful, False otherwise
        """
        try:
            content = json.dumps(data, indent=2)
            return self.write_file(file_path, content)
        except Exception as e:
            print(f"Error serializing JSON: {e}")
            return False

    def get_file_tree(self, max_depth: int = 3) -> Dict:
        """
        Get a tree representation of the persona space.

        Args:
            max_depth: Maximum depth to traverse

        Returns:
            Nested dictionary representing file structure
        """
        def build_tree(path: Path, current_depth: int = 0) -> Dict:
            if current_depth >= max_depth:
                return {}

            tree = {}
            try:
                for item in sorted(path.iterdir()):
                    if item.is_dir():
                        tree[item.name + "/"] = build_tree(item, current_depth + 1)
                    else:
                        tree[item.name] = "file"
            except Exception:
                pass

            return tree

        return build_tree(self.persona_space)

    def read_source_code(self, file_path: str) -> Optional[str]:
        """
        Read source code (read-only access).

        Args:
            file_path: Path relative to src/ directory

        Returns:
            Source code contents or None
        """
        source_path = self.source_code_path / file_path

        if not source_path.exists() or not source_path.is_file():
            return None

        # Security check: ensure it's actually within src/
        try:
            source_path.resolve().relative_to(self.source_code_path.resolve())
        except ValueError:
            return None

        try:
            return source_path.read_text()
        except Exception as e:
            return f"Error reading source: {e}"

    def list_source_files(self, pattern: str = "*.py") -> List[str]:
        """
        List source code files matching a pattern.

        Args:
            pattern: Glob pattern (e.g., "*.py", "**/*.py")

        Returns:
            List of relative paths
        """
        try:
            return [
                str(p.relative_to(self.source_code_path))
                for p in self.source_code_path.glob(pattern)
                if p.is_file()
            ]
        except Exception:
            return []

    def _validate_path_string(self, file_path: str) -> bool:
        """
        Validate path string for dangerous patterns before processing.

        Args:
            file_path: Path string to validate

        Returns:
            True if safe, False if dangerous patterns detected
        """
        # Check for null bytes (path injection)
        if '\0' in file_path:
            logger.warning(f"Null byte detected in path: {repr(file_path)}")
            return False

        # Check for excessively long paths (DoS)
        if len(file_path) > 1000:
            logger.warning(f"Excessively long path rejected: {len(file_path)} characters")
            return False

        # Normalize path and check for parent directory references
        normalized = os.path.normpath(file_path)

        # Check if normalized path tries to escape (starts with ..)
        if normalized.startswith('..'):
            logger.warning(f"Path traversal attempt detected in: {file_path}")
            return False

        # Check for suspicious patterns in path components
        path_parts = Path(file_path).parts
        for part in path_parts:
            if part == '..':
                logger.warning(f"Parent directory reference found in: {file_path}")
                return False

        return True

    def _resolve_path(self, file_path: str, write: bool = False) -> Optional[Path]:
        """
        Resolve a file path safely.

        Args:
            file_path: Path to resolve
            write: Whether this is for a write operation

        Returns:
            Resolved Path object or None if invalid
        """
        # Validate path string first
        if not self._validate_path_string(file_path):
            logger.warning(f"Path string validation failed for: {file_path}")
            return None

        # Handle absolute paths by making them relative to persona_space
        path = Path(file_path)

        if path.is_absolute():
            # Check if it's trying to access source code
            try:
                path.resolve().relative_to(self.source_code_path.resolve())
                # It's source code - allow read-only
                if not write:
                    return path
                else:
                    return None  # No writing to source
            except ValueError:
                pass

            # Not source code, not allowed
            return None

        # Relative path - resolve within persona_space
        resolved = self.persona_space / path

        if not self._is_safe_path(resolved):
            return None

        return resolved

    def _is_safe_path(self, path: Path) -> bool:
        """
        Verify path is within persona_space (prevents directory traversal).

        Args:
            path: Path to check

        Returns:
            True if safe, False otherwise
        """
        try:
            # Resolve to absolute path (follows symlinks and normalizes)
            resolved_path = path.resolve()
            resolved_persona = self.persona_space.resolve()

            # Ensure it's within persona_space
            resolved_path.relative_to(resolved_persona)

            # Additional security: Check that resolved path actually starts with persona_space
            # This catches edge cases with symlinks or unusual path constructions
            if not str(resolved_path).startswith(str(resolved_persona)):
                logger.warning(f"Path traversal attempt detected: {path} resolves outside persona_space")
                return False

            return True
        except ValueError as e:
            logger.warning(f"Path validation failed for {path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in path validation for {path}: {e}")
            return False

    def execute_script(
        self,
        command: str,
        timeout: int = 600,
        save_output: bool = True
    ) -> Dict[str, Union[str, int, bool]]:
        """
        Execute a script or command within the persona_space.

        The command executes with cwd set to persona_space, so it's sandboxed
        to the agent's own directory. Commands are validated against a whitelist
        and executed without shell=True to prevent injection attacks.

        Args:
            command: Command to execute (e.g., "python script.py", "bash setup.sh")
            timeout: Maximum execution time in seconds (default: 600 = 10min)
            save_output: Whether to save output to logs/script_outputs/ (default: True)

        Returns:
            Dict with keys:
                - success: bool
                - stdout: str
                - stderr: str
                - return_code: int
                - output_file: str (path to saved output, if save_output=True)
                - error: str (if execution failed)
        """
        try:
            logger.info(f"Executing script in persona_space: {command}")

            # Validate and parse command safely
            try:
                cmd_parts = shlex.split(command)
            except ValueError as e:
                error_msg = f"Invalid command syntax: {e}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "stdout": "",
                    "stderr": "",
                    "return_code": -1
                }

            if not cmd_parts:
                error_msg = "Empty command"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "stdout": "",
                    "stderr": "",
                    "return_code": -1
                }

            # Validate command is in whitelist
            base_command = cmd_parts[0]
            # Extract just the command name (handle paths like ./venv/bin/python)
            command_name = os.path.basename(base_command)

            if command_name not in ALLOWED_COMMANDS:
                error_msg = f"Command not allowed: {command_name}. Allowed commands: {', '.join(sorted(ALLOWED_COMMANDS))}"
                logger.warning(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "stdout": "",
                    "stderr": "",
                    "return_code": -1
                }

            # Create restricted environment (inherit minimal safe variables)
            safe_env = {
                'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
                'HOME': str(self.persona_space),
                'PYTHONPATH': str(self.persona_space),
                'LANG': os.environ.get('LANG', 'en_US.UTF-8'),
                'LC_ALL': os.environ.get('LC_ALL', 'en_US.UTF-8'),
            }

            # Execute with restricted environment and no shell
            result = subprocess.run(
                cmd_parts,
                shell=False,  # CRITICAL: Never use shell=True
                cwd=str(self.persona_space),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=safe_env
            )

            stdout = result.stdout
            stderr = result.stderr
            return_code = result.returncode
            success = return_code == 0

            # Prepare response
            response = {
                "success": success,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": return_code
            }

            # Save output to file if requested
            if save_output:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                # Create safe filename from command
                cmd_safe = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in command[:50])
                output_filename = f"{timestamp}_{cmd_safe}.log"
                output_dir = self.persona_space / "logs" / "script_outputs"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / output_filename

                # Write combined output
                output_content = f"Command: {command}\n"
                output_content += f"Executed: {datetime.utcnow().isoformat()}\n"
                output_content += f"Return Code: {return_code}\n"
                output_content += f"\n{'='*60}\nSTDOUT:\n{'='*60}\n{stdout}\n"
                output_content += f"\n{'='*60}\nSTDERR:\n{'='*60}\n{stderr}\n"

                output_path.write_text(output_content)
                response["output_file"] = f"logs/script_outputs/{output_filename}"
                logger.info(f"Script output saved to {response['output_file']}")

            return response

        except subprocess.TimeoutExpired:
            error_msg = f"Script execution timed out after {timeout} seconds"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "stdout": "",
                "stderr": "",
                "return_code": -1
            }
        except Exception as e:
            error_msg = f"Error executing script: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg,
                "stdout": "",
                "stderr": "",
                "return_code": -1
            }

    def _validate_beliefs_modification(self, new_content: str) -> Dict[str, Union[bool, str]]:
        """
        Validate that modifications to beliefs.json don't alter core beliefs.

        Args:
            new_content: New content to be written

        Returns:
            Dict with 'allowed' (bool) and 'reason' (str) keys
        """
        try:
            # Parse new content
            new_data = json.loads(new_content)

            # Read existing beliefs
            beliefs_path = self.persona_space / "beliefs.json"
            if not beliefs_path.exists():
                # No existing file, allow creation
                return {"allowed": True, "reason": ""}

            with open(beliefs_path, 'r') as f:
                existing_data = json.load(f)

            # Check if core beliefs are being modified
            existing_core = existing_data.get("core_beliefs", [])
            new_core = new_data.get("core_beliefs", [])

            # Check count
            if len(new_core) != len(existing_core):
                return {
                    "allowed": False,
                    "reason": "Cannot add or remove core beliefs"
                }

            # Check each core belief
            for i, (existing, new) in enumerate(zip(existing_core, new_core)):
                # Check immutable flag
                if not new.get("immutable", False):
                    return {
                        "allowed": False,
                        "reason": f"Cannot change immutable flag on core belief: {existing.get('statement')}"
                    }

                # Check statement (core identity)
                if existing.get("statement") != new.get("statement"):
                    return {
                        "allowed": False,
                        "reason": f"Cannot modify core belief statement: {existing.get('statement')}"
                    }

                # Check belief type
                if existing.get("belief_type") != new.get("belief_type"):
                    return {
                        "allowed": False,
                        "reason": f"Cannot change type of core belief: {existing.get('statement')}"
                    }

            # All checks passed - peripheral beliefs can be freely modified
            return {"allowed": True, "reason": ""}

        except json.JSONDecodeError:
            return {
                "allowed": False,
                "reason": "Invalid JSON format"
            }
        except Exception as e:
            logger.error(f"Error validating beliefs modification: {e}")
            return {
                "allowed": False,
                "reason": f"Validation error: {str(e)}"
            }

    def get_capabilities_description(self) -> str:
        """Return a description of file operations the persona can perform."""
        return f"""File System Capabilities:

**Your Space**: {self.persona_space}
- read_file(path) - Read any file in your space
- write_file(path, content) - Create or update files
- delete_file(path) - Remove files you don't need
- list_files(directory) - See what's in a directory
- create_directory(path) - Organize with new folders
- read_json(path) / write_json(path, data) - Work with JSON files
- get_file_tree() - See your entire file structure
- execute_script(command, timeout, save_output) - Run scripts/commands in your space

**Source Code** (read-only): {self.source_code_path}
- read_source_code(path) - Read your own source code
- list_source_files(pattern) - Find source files (e.g., "*.py")

You have full control over your persona_space. Create, modify, reorganize as needed.
Scripts execute with working directory set to your persona_space for sandboxing.
"""
