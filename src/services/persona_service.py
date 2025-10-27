"""
Persona Service - Orchestrates the self-modifying persona system.

This service coordinates:
1. Persona prompt building (with self-concept)
2. LLM generation (with emotional co-analysis)
3. Emotional reconciliation (internal vs external)
4. File system access for persona autonomy
5. Anti-meta-talk filtering and rewriting
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
from src.services.anti_metatalk import (
    create_logit_bias_builder,
    create_metatalk_detector,
    create_metatalk_rewriter,
)
from src.services.persona_config import create_persona_config_loader
from config.settings import settings

logger = logging.getLogger(__name__)


class PersonaService:
    """Orchestrates the self-modifying persona system."""

    def __init__(
        self,
        llm_service: LLMService,
        persona_space_path: str = "persona_space",
        retrieval_service=None,
        enable_anti_metatalk: bool = True,
        logit_bias_strength: float = -100,
        auto_rewrite: bool = True,
        belief_system=None,
        web_search_service=None,
        url_fetcher_service=None,
        web_interpretation_service=None,
    ):
        """
        Initialize persona service.

        Args:
            llm_service: LLM service for generation
            persona_space_path: Path to persona's file space
            retrieval_service: Optional retrieval service for memory access
            enable_anti_metatalk: Enable anti-meta-talk system
            logit_bias_strength: Strength of logit bias for token suppression
            auto_rewrite: Automatically rewrite responses containing meta-talk
            belief_system: Optional belief system for ontological grounding
            web_search_service: Optional web search service
            url_fetcher_service: Optional URL fetcher service
            web_interpretation_service: Optional web interpretation service
        """
        self.llm = llm_service
        self.prompt_builder = PersonaPromptBuilder(persona_space_path, belief_system=belief_system)
        self.reconciler = EmotionalReconciler(llm_service, persona_space_path)
        self.file_manager = PersonaFileManager(persona_space_path)
        self.retrieval_service = retrieval_service
        self.persona_space = Path(persona_space_path)
        self.action_log_path = self.persona_space / "meta" / "actions_log.json"
        self.config_loader = create_persona_config_loader(persona_space_path)

        # Web services
        self.web_search_service = web_search_service
        self.url_fetcher_service = url_fetcher_service
        self.web_interpretation_service = web_interpretation_service

        # Rate limiting for web operations (per conversation)
        self.search_count = 0
        self.url_fetch_count = 0

        # Anti-meta-talk system
        self.enable_anti_metatalk = enable_anti_metatalk
        self.auto_rewrite = auto_rewrite
        self.logit_bias_strength = logit_bias_strength

        if enable_anti_metatalk:
            # Get model name from LLM service for tokenizer
            model_name = llm_service.model
            self.logit_bias_builder = create_logit_bias_builder(model=model_name)
            self.logit_bias = self.logit_bias_builder.build_bias(strength=logit_bias_strength)
            self.metatalk_detector = create_metatalk_detector()
            self.metatalk_rewriter = create_metatalk_rewriter(llm_service)
            logger.info(f"Anti-meta-talk enabled with {len(self.logit_bias)} suppressed tokens")
        else:
            self.logit_bias = {}
            self.metatalk_detector = None
            self.metatalk_rewriter = None

    def generate_response(self, user_message: str, retrieve_memories: bool = True, top_k: int = 5, conversation_history: list = None) -> Tuple[str, Dict]:
        """
        Generate a persona response with emotional co-analysis and tool use.

        This is the main method that:
        1. Retrieves relevant memories (if enabled)
        2. Builds persona-aware prompt with memory context
        3. Generates response with tool calling enabled
        4. Executes any tool calls (file operations)
        5. Extracts and reconciles emotional perspectives
        6. Returns cleaned response and reconciliation data

        Args:
            user_message: The user's message
            retrieve_memories: Whether to retrieve relevant memories
            top_k: Number of memories to retrieve
            conversation_history: Previous messages in the conversation [{"role": "user"|"assistant", "content": "..."}]

        Returns:
            Tuple of (response_text, reconciliation_data)
        """
        if conversation_history is None:
            conversation_history = []

        # Load persona's LLM configuration
        config = self.config_loader.load_config()

        # Retrieve relevant memories if enabled (use config setting)
        memories = []
        should_retrieve = config.retrieve_memories if hasattr(config, 'retrieve_memories') else retrieve_memories
        memory_count = config.memory_top_k if hasattr(config, 'memory_top_k') else top_k

        if should_retrieve and self.retrieval_service:
            try:
                memories = self.retrieval_service.retrieve_similar(
                    prompt=user_message,
                    top_k=memory_count
                )
                logger.info(f"Retrieved {len(memories)} relevant memories for persona")
                print(f"🧠 Retrieved {len(memories)} memories for context")
            except Exception as e:
                logger.error(f"Failed to retrieve memories: {e}")

        # Build the persona prompt with current context and memories
        full_prompt = self.prompt_builder.build_prompt(user_message, memories=memories)

        # Log prompt stats for visibility (not full content to avoid clutter)
        prompt_lines = full_prompt.count('\n')
        prompt_chars = len(full_prompt)
        logger.info(f"Built persona prompt: {prompt_lines} lines, {prompt_chars} chars")
        print(f"📝 Persona prompt built: {prompt_lines} lines, {prompt_chars} chars")

        # Initialize messages for tool loop
        # Start with system prompt, then conversation history, then current user message
        messages = [{"role": "system", "content": full_prompt}]

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        tools = self._get_tool_definitions()

        # Tool execution loop (max 5 iterations to prevent infinite loops)
        max_iterations = 5
        assistant_responses = []  # Collect all assistant content

        for iteration in range(max_iterations):
            # Generate response with tools using persona's config
            use_anti_metatalk = config.enable_anti_metatalk if hasattr(config, 'enable_anti_metatalk') else self.enable_anti_metatalk

            result = self.llm.generate_with_tools(
                messages=messages,
                tools=tools,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                presence_penalty=config.presence_penalty,
                frequency_penalty=config.frequency_penalty,
                logit_bias=self.logit_bias if use_anti_metatalk else None,
            )

            message = result["message"]
            finish_reason = result["finish_reason"]

            # Collect content from assistant
            if message.content:
                assistant_responses.append(message.content)

            # Add assistant message to history
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [tc.model_dump() for tc in message.tool_calls] if message.tool_calls else None
            })

            # If no tool calls, we're done
            if not message.tool_calls:
                break

            # Execute tool calls
            for tool_call in message.tool_calls:
                func = tool_call.function
                tool_name = func.name
                arguments = json.loads(func.arguments)

                logger.info(f"Persona calling tool: {tool_name} with args: {arguments}")
                # Print to console for user visibility
                print(f"🤖 AGENT ACTION: {tool_name}({', '.join(f'{k}={v[:50] if isinstance(v, str) else v}...' if isinstance(v, str) and len(v) > 50 else f'{k}={v}' for k, v in arguments.items())})")

                # Execute the tool
                tool_result = self._execute_tool(tool_name, arguments)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

        # Combine all assistant responses (use first substantive one, not meta-statements)
        # Prefer the first response as it usually has the actual content
        raw_response = assistant_responses[0] if assistant_responses else ""

        # Debug logging
        logger.info(f"Assistant responses collected: {len(assistant_responses)}")
        if assistant_responses:
            logger.info(f"First response length: {len(assistant_responses[0])}")
        else:
            logger.warning("No assistant responses collected - response will be empty!")

        if iteration >= max_iterations - 1:
            logger.warning(f"Tool loop reached max iterations ({max_iterations})")

        # Extract internal emotional assessment
        internal_assessment = extract_emotional_assessment(raw_response)

        # Remove assessment from user-facing response
        clean_response = remove_emotional_assessment(raw_response)

        # Layer 3: Detect and handle meta-talk (use config settings)
        auto_rewrite_config = config.auto_rewrite_metatalk if hasattr(config, 'auto_rewrite_metatalk') else self.auto_rewrite

        if use_anti_metatalk and self.metatalk_detector:
            has_metatalk = self.metatalk_detector.detect(clean_response)
            if has_metatalk:
                meta_phrases = self.metatalk_detector.find_all(clean_response)
                logger.warning(f"Meta-talk detected in response: {meta_phrases}")
                print(f"⚠️  Meta-talk detected: {meta_phrases}")

                if auto_rewrite_config and self.metatalk_rewriter:
                    print(f"🔄 Auto-rewriting response to remove meta-talk...")
                    try:
                        clean_response = self.metatalk_rewriter.rewrite(clean_response, user_message)
                        print(f"✅ Response rewritten successfully")
                    except Exception as e:
                        logger.error(f"Rewrite failed: {e}, stripping meta-talk instead")
                        print(f"❌ Rewrite failed, stripping meta-talk sentences instead")
                        clean_response = self.metatalk_detector.strip(clean_response)
                else:
                    # Just strip the meta-talk
                    clean_response = self.metatalk_detector.strip(clean_response)

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

    def reset_web_limits(self):
        """Reset web operation counters for a new conversation."""
        self.search_count = 0
        self.url_fetch_count = 0
        logger.info("Web operation limits reset")

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
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_script",
                    "description": "Execute a script or shell command within your persona_space. The command runs with working directory set to your space, giving you full access to run any interpreter (python, bash, node, etc.). You can create virtual environments and install packages. Output is returned to you and saved to logs/script_outputs/. Default timeout is 600 seconds (10 minutes). Output saving defaults to true.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Shell command to execute (e.g., 'python script.py', 'bash setup.sh', 'python -m venv myenv', 'myenv/bin/pip install requests')"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Maximum execution time in seconds (optional, defaults to 600)"
                            },
                            "save_output": {
                                "type": "boolean",
                                "description": "Whether to save output to a log file (optional, defaults to true)"
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for current information, facts, or knowledge you don't have. Use this when you need up-to-date information, news, specific facts, or when your knowledge might be outdated. Returns search results with titles, URLs, and snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (e.g., 'latest AI breakthroughs 2025', 'what is quantum computing')"
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5, max: 10)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browse_url",
                    "description": "Visit a URL and read its content with full understanding. The content will be interpreted from your perspective and stored in your memory. Use this to deeply understand a web page, article, or resource.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to visit and read"
                            },
                            "context": {
                                "type": "string",
                                "description": "Optional context about why you're visiting this URL (helps with interpretation)"
                            }
                        },
                        "required": ["url"]
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

            elif tool_name == "execute_script":
                command = arguments.get("command")
                timeout = arguments.get("timeout", 600)
                save_output = arguments.get("save_output", True)

                exec_result = self.file_manager.execute_script(
                    command=command,
                    timeout=timeout,
                    save_output=save_output
                )

                # Format result for agent
                if exec_result.get("success"):
                    result = f"✓ Script executed successfully (exit code {exec_result['return_code']})\n\n"
                    if exec_result.get("stdout"):
                        result += f"STDOUT:\n{exec_result['stdout']}\n"
                    if exec_result.get("stderr"):
                        result += f"\nSTDERR:\n{exec_result['stderr']}\n"
                    if exec_result.get("output_file"):
                        result += f"\n📁 Full output saved to: {exec_result['output_file']}"
                else:
                    result = f"✗ Script execution failed\n"
                    if exec_result.get("error"):
                        result += f"Error: {exec_result['error']}\n"
                    if exec_result.get("stdout"):
                        result += f"\nSTDOUT:\n{exec_result['stdout']}\n"
                    if exec_result.get("stderr"):
                        result += f"\nSTDERR:\n{exec_result['stderr']}\n"

            elif tool_name == "search_web":
                if not self.web_search_service:
                    result = "Search functionality not available (web_search_service not configured)"
                elif self.search_count >= settings.MAX_SEARCHES_PER_CONVERSATION:
                    result = f"Search limit reached ({settings.MAX_SEARCHES_PER_CONVERSATION} searches per conversation)"
                else:
                    query = arguments.get("query")
                    num_results = min(arguments.get("num_results", 5), 10)  # Cap at 10

                    try:
                        search_results = self.web_search_service.search(query, num_results)
                        self.search_count += 1

                        # Format results for persona
                        result = f"Found {len(search_results)} results for '{query}':\n\n"
                        for sr in search_results:
                            result += f"{sr.position}. {sr.title}\n"
                            result += f"   URL: {sr.url}\n"
                            result += f"   {sr.snippet}\n\n"

                        result += f"(Search {self.search_count}/{settings.MAX_SEARCHES_PER_CONVERSATION})"

                    except Exception as e:
                        result = f"Search failed: {str(e)}"
                        logger.error(f"Search error: {e}")

            elif tool_name == "browse_url":
                if not self.url_fetcher_service or not self.web_interpretation_service:
                    result = "URL browsing not available (services not configured)"
                elif self.url_fetch_count >= settings.MAX_URL_FETCHES_PER_CONVERSATION:
                    result = f"URL fetch limit reached ({settings.MAX_URL_FETCHES_PER_CONVERSATION} fetches per conversation)"
                else:
                    url = arguments.get("url")
                    context = arguments.get("context", "")

                    try:
                        # Fetch the URL
                        fetched = self.url_fetcher_service.fetch_url(url)
                        self.url_fetch_count += 1

                        if not fetched.success:
                            result = f"Failed to fetch {url}: {fetched.error_message}"
                        else:
                            # Interpret the content
                            interpretation = self.web_interpretation_service.interpret_content(
                                url=url,
                                title=fetched.title,
                                content=fetched.main_content or fetched.text_content,
                                user_context=context,
                                query_context=None,
                            )

                            # Format interpretation for persona
                            result = f"Content from: {fetched.title}\n"
                            result += f"URL: {url}\n\n"
                            result += f"YOUR INTERPRETATION:\n{interpretation.interpretation}\n\n"

                            if interpretation.key_facts:
                                result += "KEY FACTS:\n"
                                for fact in interpretation.key_facts:
                                    result += f"• {fact}\n"
                                result += "\n"

                            result += f"Emotional salience: {interpretation.emotional_salience}\n"
                            result += f"Relevance: {interpretation.relevance_to_query}\n\n"
                            result += f"(Fetch {self.url_fetch_count}/{settings.MAX_URL_FETCHES_PER_CONVERSATION})"

                            # TODO: Store interpretation in memory as WEB_OBSERVATION

                    except Exception as e:
                        result = f"Error browsing {url}: {str(e)}"
                        logger.error(f"Browse error: {e}")

            else:
                result = f"Unknown tool: {tool_name}"

            # Log the action
            self._log_action(tool_name, arguments, result)

            # Print result summary for user visibility
            result_preview = result[:100] + "..." if len(result) > 100 else result
            print(f"   ✓ Result: {result_preview}")

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
