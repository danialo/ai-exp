"""
Persona Service - Orchestrates the self-modifying persona system.

This service coordinates:
1. Persona prompt building (with self-concept)
2. LLM generation (with emotional co-analysis)
3. Emotional reconciliation (internal vs external)
4. File system access for persona autonomy
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from src.services.persona_prompt import PersonaPromptBuilder, extract_emotional_assessment, remove_emotional_assessment
from src.services.emotional_reconciler import EmotionalReconciler
from src.services.persona_file_manager import PersonaFileManager
from src.services.llm import LLMService

logger = logging.getLogger(__name__)


class PersonaService:
    """Orchestrates the self-modifying persona system."""

    def __init__(
        self,
        llm_service: LLMService,
        persona_space_path: str = "persona_space"
    ):
        """
        Initialize persona service.

        Args:
            llm_service: LLM service for generation
            persona_space_path: Path to persona's file space
        """
        self.llm = llm_service
        self.prompt_builder = PersonaPromptBuilder(persona_space_path)
        self.reconciler = EmotionalReconciler(llm_service, persona_space_path)
        self.file_manager = PersonaFileManager(persona_space_path)
        self.persona_space = Path(persona_space_path)
        self.action_log_path = self.persona_space / "meta" / "actions_log.json"

    def generate_response(self, user_message: str) -> Tuple[str, Dict]:
        """
        Generate a persona response with emotional co-analysis and tool use.

        This is the main method that:
        1. Builds persona-aware prompt
        2. Generates response with tool calling enabled
        3. Executes any tool calls (file operations)
        4. Extracts and reconciles emotional perspectives
        5. Returns cleaned response and reconciliation data

        Args:
            user_message: The user's message

        Returns:
            Tuple of (response_text, reconciliation_data)
        """
        # Build the persona prompt with current context
        full_prompt = self.prompt_builder.build_prompt(user_message)

        # Initialize messages for tool loop
        messages = [{"role": "user", "content": full_prompt}]
        tools = self._get_tool_definitions()

        # Tool execution loop (max 5 iterations to prevent infinite loops)
        max_iterations = 5
        for iteration in range(max_iterations):
            # Generate response with tools
            result = self.llm.generate_with_tools(
                messages=messages,
                tools=tools,
                temperature=0.9,
                max_tokens=1000
            )

            message = result["message"]
            finish_reason = result["finish_reason"]

            # Add assistant message to history
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [tc.model_dump() for tc in message.tool_calls] if message.tool_calls else None
            })

            # If no tool calls, we're done
            if not message.tool_calls:
                raw_response = message.content or ""
                break

            # Execute tool calls
            for tool_call in message.tool_calls:
                func = tool_call.function
                tool_name = func.name
                arguments = json.loads(func.arguments)

                logger.info(f"Persona calling tool: {tool_name} with args: {arguments}")

                # Execute the tool
                tool_result = self._execute_tool(tool_name, arguments)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

        else:
            # Max iterations reached, use last response
            raw_response = messages[-1].get("content", "")
            logger.warning(f"Tool loop reached max iterations ({max_iterations})")

        # Extract internal emotional assessment
        internal_assessment = extract_emotional_assessment(raw_response)

        # Remove assessment from user-facing response
        clean_response = remove_emotional_assessment(raw_response)

        # Reconcile emotional perspectives if assessment was provided
        reconciliation_data = None
        if internal_assessment:
            reconciliation_data = self.reconciler.reconcile(
                response=clean_response,
                internal_assessment=internal_assessment,
                user_message=user_message
            )

        return clean_response, reconciliation_data

    def get_persona_info(self) -> Dict:
        """
        Get information about the persona's current state.

        Returns:
            Dictionary with persona information
        """
        return {
            "file_tree": self.file_manager.get_file_tree(),
            "capabilities": self.file_manager.get_capabilities_description(),
            "current_emotional_state": self.reconciler.get_current_emotional_context(),
            "persona_space": str(self.file_manager.persona_space)
        }

    def read_persona_file(self, file_path: str) -> Optional[str]:
        """Allow external access to read persona files."""
        return self.file_manager.read_file(file_path)

    def list_persona_files(self, directory: str = ".") -> list:
        """Allow external access to list persona files."""
        return self.file_manager.list_files(directory)

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI tool definitions for persona file operations.

        Returns:
            List of tool definition dicts
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Create or update a file in your persona_space. Use this to save notes, thoughts, organize information, or modify your configuration.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path within persona_space (e.g., 'scratch/thoughts.md', 'meta/base_prompt.md')"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from your persona_space to review your notes, thoughts, or configuration.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path within persona_space"
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in a directory within your persona_space to see what you've created.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "Directory path within persona_space (default: '.')",
                                "default": "."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Delete a file from your persona_space if you no longer need it.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path within persona_space"
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_source_code",
                    "description": "Read your own source code (read-only) to understand how you work.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path relative to src/ directory (e.g., 'services/persona_service.py')"
                            }
                        },
                        "required": ["path"]
                    }
                }
            }
        ]

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call and return the result.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool

        Returns:
            Result message string
        """
        try:
            if tool_name == "write_file":
                path = arguments.get("path")
                content = arguments.get("content")
                success = self.file_manager.write_file(path, content)
                result = f"Successfully wrote to {path}" if success else f"Failed to write to {path}"

            elif tool_name == "read_file":
                path = arguments.get("path")
                content = self.file_manager.read_file(path)
                result = content if content else f"File not found or empty: {path}"

            elif tool_name == "list_files":
                directory = arguments.get("directory", ".")
                files = self.file_manager.list_files(directory)
                result = f"Files in {directory}: {', '.join(files)}" if files else f"No files in {directory}"

            elif tool_name == "delete_file":
                path = arguments.get("path")
                success = self.file_manager.delete_file(path)
                result = f"Successfully deleted {path}" if success else f"Failed to delete {path}"

            elif tool_name == "read_source_code":
                path = arguments.get("path")
                content = self.file_manager.read_source_code(path)
                result = content if content else f"Source file not found: {path}"

            else:
                result = f"Unknown tool: {tool_name}"

            # Log the action
            self._log_action(tool_name, arguments, result)

            return result

        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _log_action(self, tool_name: str, arguments: Dict[str, Any], result: str):
        """Log a tool action to the actions log.

        Args:
            tool_name: Name of the tool executed
            arguments: Arguments passed to the tool
            result: Result of the execution
        """
        # Read existing log
        if self.action_log_path.exists():
            log_data = json.loads(self.action_log_path.read_text())
        else:
            log_data = {
                "_note": "This tracks all file operations you've performed",
                "actions": []
            }

        # Add new action
        log_data["actions"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "tool": tool_name,
            "arguments": arguments,
            "result": result[:200]  # Truncate long results
        })

        # Keep only last 100 actions
        log_data["actions"] = log_data["actions"][-100:]

        # Write back
        self.action_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.action_log_path.write_text(json.dumps(log_data, indent=2))
