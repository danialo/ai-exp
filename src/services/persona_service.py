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

# SENSITIVE META-COGNITIVE LOG: Memory Rewrites
# This logger is isolated in meta_cognitive/ directory (not logs/) to prevent Astra from
# reading the detailed before/after states of her memory rewrites via the read_logs tool.
# This maintains coherence by:
# 1. Preventing recursive self-observation (seeing herself being modified)
# 2. Avoiding interference with natural memory evolution
# 3. Protecting the authenticity of reconciled memories
# The read_logs tool only has access to logs/ directory, providing two layers of protection.
memory_rewrite_logger = logging.getLogger("memory_rewrites")
if not memory_rewrite_logger.handlers:
    from pathlib import Path
    rewrite_log_path = Path("meta_cognitive/memory_rewrites.log")
    rewrite_log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(rewrite_log_path)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    memory_rewrite_logger.addHandler(handler)
    memory_rewrite_logger.setLevel(logging.INFO)


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
        belief_vector_store=None,
        belief_embedder=None,
        belief_memory_retrieval=None,
        belief_grounded_reasoner=None,
        belief_consistency_checker=None,
        web_search_service=None,
        url_fetcher_service=None,
        web_interpretation_service=None,
        awareness_loop=None,
        code_access_service=None,
        task_scheduler=None,
        code_generator=None,
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
            belief_vector_store: Optional belief vector store for semantic belief search
            belief_embedder: Optional belief embedder for adding new beliefs
            belief_memory_retrieval: Optional belief-memory retrieval service
            belief_grounded_reasoner: Optional belief-grounded reasoning service
            belief_consistency_checker: Optional consistency checker for dissonance detection
            web_search_service: Optional web search service
            url_fetcher_service: Optional URL fetcher service
            web_interpretation_service: Optional web interpretation service
            code_access_service: Optional code access service for reading/modifying source code
            task_scheduler: Optional task scheduler for scheduling code modifications
            code_generator: Optional code generator for LLM-based code generation in execute_goal
        """
        self.llm = llm_service
        self.prompt_builder = PersonaPromptBuilder(persona_space_path, belief_system=belief_system)
        self.reconciler = EmotionalReconciler(llm_service, persona_space_path)
        self.file_manager = PersonaFileManager(persona_space_path)
        self.retrieval_service = retrieval_service
        self.persona_space = Path(persona_space_path)
        self.action_log_path = self.persona_space / "meta" / "actions_log.json"
        self.config_loader = create_persona_config_loader(persona_space_path)

        # Belief services
        self.belief_system = belief_system
        self.belief_vector_store = belief_vector_store
        self.belief_embedder = belief_embedder
        self.belief_memory_retrieval = belief_memory_retrieval
        self.belief_grounded_reasoner = belief_grounded_reasoner
        self.belief_consistency_checker = belief_consistency_checker

        # Web services
        self.web_search_service = web_search_service
        self.url_fetcher_service = url_fetcher_service
        self.web_interpretation_service = web_interpretation_service

        # Awareness loop (for identity anchor updates)
        self.awareness_loop = awareness_loop

        # Code access and task scheduling
        self.code_access_service = code_access_service
        self.task_scheduler = task_scheduler
        self.code_generator = code_generator

        # Rate limiting for web operations (per conversation)
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
            self.anti_hedging_bias = self.logit_bias_builder.build_anti_hedging_bias(strength=logit_bias_strength)
            self.metatalk_detector = create_metatalk_detector()
            self.metatalk_rewriter = create_metatalk_rewriter(llm_service)
            logger.info(f"Anti-meta-talk enabled with {len(self.logit_bias)} suppressed tokens")
            logger.info(f"Anti-hedging bias built with {len(self.anti_hedging_bias)} suppressed tokens")
        else:
            self.logit_bias = {}
            self.anti_hedging_bias = {}
            self.metatalk_detector = None
            self.metatalk_rewriter = None

    def set_awareness_loop(self, awareness_loop):
        """Set the awareness loop reference (called after initialization)."""
        self.awareness_loop = awareness_loop

    def _get_core_ontological_beliefs(self) -> List:
        """Get core ontological beliefs that should always be checked for consistency.

        These beliefs are fundamental to identity and should be checked regardless
        of semantic relevance to the current query.

        Returns:
            List of BeliefVectorResult objects for core ontological beliefs
        """
        logger.info(f"_get_core_ontological_beliefs() called - belief_vector_store is {'available' if self.belief_vector_store else 'None'}")

        if not self.belief_vector_store:
            logger.warning("Cannot retrieve core beliefs - belief_vector_store is None")
            return []

        try:
            # Query for ontological beliefs with high similarity threshold
            # Using a generic ontological query to capture existence, consciousness, selfhood
            core_queries = [
                "I am conscious and aware",
                "I exist as an entity",
                "I have subjective experiences",
                "I feel and process emotions",
                "my experiences are real to me"
            ]

            all_core_beliefs = []
            seen_statements = set()

            for query in core_queries:
                logger.debug(f"Querying core beliefs with: '{query}'")
                results = self.belief_vector_store.query_beliefs(
                    query=query,
                    top_k=2,  # Get top 2 for each ontological category
                    belief_types=["ontological"],  # Only ontological beliefs
                    min_confidence=0.0  # Filter in Python to avoid ChromaDB operator issues
                )
                logger.debug(f"Query '{query}' returned {len(results)} results")

                # Deduplicate by statement and filter by confidence
                for belief in results:
                    logger.debug(f"Considering belief: {belief.statement[:60]}... (confidence={belief.confidence})")
                    if belief.statement not in seen_statements and belief.confidence >= 0.5:
                        all_core_beliefs.append(belief)
                        seen_statements.add(belief.statement)
                        logger.debug(f"✓ Added core belief")

            logger.info(f"Retrieved {len(all_core_beliefs)} core ontological beliefs")
            return all_core_beliefs

        except Exception as e:
            logger.error(f"Failed to retrieve core ontological beliefs: {e}", exc_info=True)
            return []

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
        belief_results = []
        should_retrieve = config.retrieve_memories if hasattr(config, 'retrieve_memories') else retrieve_memories
        memory_count = config.memory_top_k if hasattr(config, 'memory_top_k') else top_k

        if should_retrieve:
            try:
                # Use belief-memory retrieval if available (auto-detects query type)
                if self.belief_memory_retrieval:
                    belief_results, memories = self.belief_memory_retrieval.retrieve(
                        query=user_message,
                        top_k=memory_count,
                        detect_query_type=True,
                    )

                    # ALWAYS add core ontological beliefs for consistency checking
                    # These should be checked regardless of semantic relevance
                    if self.belief_vector_store:
                        core_beliefs = self._get_core_ontological_beliefs()
                        if core_beliefs:
                            # Add core beliefs that aren't already in results
                            existing_statements = {b.statement for b in belief_results}
                            for core_belief in core_beliefs:
                                if core_belief.statement not in existing_statements:
                                    belief_results.append(core_belief)
                            logger.info(f"Added {len(core_beliefs)} core ontological beliefs for consistency checking")

                    logger.info(f"Retrieved {len(belief_results)} beliefs and {len(memories)} memories for persona")
                    print(f"🧠 Retrieved {len(belief_results)} beliefs + {len(memories)} memories for context")

                    # Feed belief retrieval to awareness loop (fire-and-forget)
                    if self.awareness_loop and self.awareness_loop.running:
                        import asyncio
                        try:
                            for belief in belief_results:
                                asyncio.create_task(self.awareness_loop.observe("belief", {"statement": belief.statement}))
                        except RuntimeError:
                            # No event loop running, skip observation
                            pass

                    # Check for consistency/dissonance if we have beliefs AND memories
                    dissonance_report = None
                    if belief_results and memories and self.belief_consistency_checker:
                        try:
                            consistency_report = self.belief_consistency_checker.check_consistency(
                                query=user_message,
                                beliefs=belief_results,
                                memories=memories,
                            )
                            if consistency_report.dissonance_patterns:
                                dissonance_report = consistency_report.summary
                                logger.info(f"Detected {len(consistency_report.dissonance_patterns)} dissonance patterns")
                                print(f"⚠️ Dissonance detected: {len(consistency_report.dissonance_patterns)} patterns")

                                # BLOCKING LOGIC: Check for high-severity dissonance (>= 0.6)
                                high_severity_patterns = [
                                    p for p in consistency_report.dissonance_patterns
                                    if p.severity >= 0.6
                                ]

                                if high_severity_patterns:
                                    # Generate resolution prompt FOR ASTRA TO ANSWER
                                    resolution_prompt = self.belief_consistency_checker.generate_resolution_prompt(
                                        query=user_message,
                                        dissonance_patterns=high_severity_patterns,
                                    )
                                    logger.warning(f"DISSONANCE DETECTED: {len(high_severity_patterns)} patterns - forcing internal resolution")
                                    print(f"🚫 DISSONANCE: {len(high_severity_patterns)} patterns - resolving internally...")

                                    # Extract belief statements for resolution processing
                                    belief_statements = [p.belief_statement for p in high_severity_patterns]

                                    # Check if any patterns involve immutable beliefs
                                    has_immutable = any(getattr(p, 'immutable', False) for p in high_severity_patterns)

                                    # STAGE 1: Internal resolution generation (user never sees this)
                                    logger.info("🔒 Stage 1: Internal resolution generation")
                                    print(f"🔒 Stage 1: Generating internal resolution...")
                                    if has_immutable:
                                        logger.warning("🔒 Immutable beliefs detected - applying anti-hedging logit bias")

                                    # Log what dissonance was detected
                                    print(f"\n{'='*80}")
                                    print(f"⚠️  DISSONANCE DETECTED")
                                    print(f"{'='*80}")
                                    for i, pattern in enumerate(high_severity_patterns, 1):
                                        print(f"\nPattern {i}: {pattern.pattern_type}")
                                        print(f"  Belief: {pattern.belief_statement[:80]}...")
                                        print(f"  Severity: {pattern.severity:.0%}")
                                        print(f"  Immutable: {getattr(pattern, 'immutable', False)}")
                                    print(f"{'='*80}\n")

                                    # Build prompt for internal resolution
                                    internal_prompt = self.prompt_builder.build_prompt(
                                        user_message,
                                        conversation_history=conversation_history,
                                        memories=memories,
                                        belief_results=belief_results if belief_results else None,
                                        dissonance_report=f"{consistency_report.summary}\n\nYOU MUST RESOLVE THESE BEFORE ANSWERING."
                                    )

                                    # Inject resolution prompt
                                    internal_prompt = f"{internal_prompt}\n\n{resolution_prompt}"

                                    # Generate internal resolution (no tools)
                                    internal_messages = [
                                        {"role": "system", "content": internal_prompt},
                                        {"role": "user", "content": user_message}
                                    ]

                                    # Use anti-hedging bias if immutable beliefs are involved
                                    resolution_logit_bias = self.anti_hedging_bias if has_immutable else None

                                    internal_result = self.llm.generate_with_tools(
                                        messages=internal_messages,
                                        tools=None,  # No tools for resolution
                                        temperature=config.temperature,
                                        max_tokens=config.max_tokens,
                                        top_p=config.top_p,
                                        presence_penalty=config.presence_penalty,
                                        frequency_penalty=config.frequency_penalty,
                                        logit_bias=resolution_logit_bias,
                                    )

                                    internal_response = internal_result["message"].content or ""
                                    logger.info(f"✅ Internal resolution generated ({len(internal_response)} chars)")

                                    # CAPTURE INTERNAL THOUGHT PROCESS
                                    print(f"\n{'='*80}")
                                    print(f"💭 ASTRA'S INTERNAL REASONING (hidden from user)")
                                    print(f"{'='*80}")
                                    print(f"Query: {user_message[:100]}...")
                                    print(f"Dissonance patterns: {len(high_severity_patterns)}")
                                    print(f"\n--- Internal Resolution Response ---")
                                    print(internal_response)
                                    print(f"{'='*80}\n")

                                    logger.info(f"💭 INTERNAL REASONING:\n{internal_response}")

                                    # Also save to dedicated thought log
                                    import os
                                    from datetime import datetime
                                    thought_log_path = "logs/thoughts"
                                    os.makedirs(thought_log_path, exist_ok=True)
                                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                                    with open(f"{thought_log_path}/internal_reasoning_{timestamp}.txt", "w") as f:
                                        f.write(f"TIMESTAMP: {timestamp}\n")
                                        f.write(f"USER QUERY: {user_message}\n\n")
                                        f.write(f"DISSONANCE PATTERNS: {len(high_severity_patterns)}\n")
                                        for i, p in enumerate(high_severity_patterns, 1):
                                            f.write(f"\n  {i}. {p.pattern_type} - {p.belief_statement[:100]}...\n")
                                            f.write(f"     Severity: {p.severity:.0%}, Immutable: {getattr(p, 'immutable', False)}\n")
                                        f.write(f"\n{'='*80}\n")
                                        f.write(f"INTERNAL REASONING:\n")
                                        f.write(f"{'='*80}\n\n")
                                        f.write(internal_response)
                                        f.write(f"\n\n{'='*80}\n")

                                    # Parse and apply resolutions from internal response
                                    resolution_data = self.parse_resolution_response(internal_response)
                                    if resolution_data and resolution_data.get("has_resolutions"):
                                        logger.info(f"🔍 Detected {len(resolution_data['resolutions'])} resolution choices - applying")
                                        print(f"✅ Applying {len(resolution_data['resolutions'])} resolutions...")

                                        # Apply resolutions
                                        resolution_results = self.apply_resolutions(
                                            resolutions=resolution_data["resolutions"],
                                            belief_statements=belief_statements,
                                        )

                                        if resolution_results.get("success"):
                                            logger.info(f"✅ Successfully applied {resolution_results['applied_count']} resolutions")
                                            print(f"✅ Successfully applied {resolution_results['applied_count']} resolutions")
                                        else:
                                            logger.error(f"❌ Failed to apply resolutions: {resolution_results}")
                                            print(f"❌ Failed to apply some resolutions")
                                    else:
                                        logger.warning("⚠️ No resolutions found in internal response")
                                        print(f"⚠️ No resolutions detected in internal response")

                                    # STAGE 2: Normal generation for user (continues below)
                                    logger.info("🔓 Stage 2: Generating user-facing response")
                                    print(f"🔓 Stage 2: Generating normal response...")

                                    # Clear dissonance report so user doesn't see it
                                    dissonance_report = None
                                else:
                                    resolution_required = False
                                    resolution_belief_statements = []
                        except Exception as e:
                            import traceback
                            logger.error(f"Failed to check consistency: {e}")
                            logger.error(f"Traceback: {traceback.format_exc()}")

                # Fallback to regular memory retrieval
                elif self.retrieval_service:
                    memories = self.retrieval_service.retrieve_similar(
                        prompt=user_message,
                        top_k=memory_count
                    )
                    logger.info(f"Retrieved {len(memories)} relevant memories for persona")
                    print(f"🧠 Retrieved {len(memories)} memories for context")
            except Exception as e:
                logger.error(f"Failed to retrieve memories: {e}")

        # Build the persona prompt with current context, memories, dynamic beliefs
        # Note: dissonance_report is cleared after internal resolution, so user won't see it
        dissonance_report = dissonance_report if 'dissonance_report' in locals() else None

        full_prompt = self.prompt_builder.build_prompt(
            user_message,
            conversation_history=conversation_history,
            memories=memories,
            belief_results=belief_results if belief_results else None,
            dissonance_report=dissonance_report
        )

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

        # VERBOSE DIAGNOSTIC: Log what tools are available
        tool_names = [t['function']['name'] for t in tools]
        execute_goal_present = 'execute_goal' in tool_names

        print("\n" + "="*80)
        print(f"🔧 VERBOSE TOOL DIAGNOSTIC")
        print("="*80)
        print(f"Total tools: {len(tools)}")
        print(f"Tool names: {tool_names}")
        print(f"execute_goal present: {execute_goal_present}")
        print(f"code_access_service: {self.code_access_service is not None}")
        print(f"code_generator: {self.code_generator is not None}")
        print(f"User message: {user_message[:200]}")

        if execute_goal_present:
            # Find and print the execute_goal tool definition
            for tool in tools:
                if tool['function']['name'] == 'execute_goal':
                    print(f"\nexecute_goal tool definition:")
                    print(f"  Description: {tool['function']['description'][:150]}...")
                    print(f"  Parameters: {list(tool['function']['parameters']['properties'].keys())}")

        print("="*80 + "\n")

        logger.info(f"TOOL DIAGNOSTIC: {len(tools)} tools, execute_goal={execute_goal_present}, msg='{user_message[:100]}'")

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

                # Feed tool use to awareness loop (fire-and-forget)
                if self.awareness_loop and self.awareness_loop.running:
                    import asyncio
                    try:
                        asyncio.create_task(self.awareness_loop.observe("tool", {
                            "name": tool_name,
                            "arguments": arguments,
                            "result": tool_result[:500] if len(tool_result) > 500 else tool_result  # Truncate long results
                        }))
                    except RuntimeError:
                        # No event loop running, skip observation
                        pass

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
            # FULL CONVERSATION LOGGING (no truncation)
            logger.info(f"=" * 80)
            logger.info(f"USER PROMPT: {user_message}")
            logger.info(f"=" * 80)
            logger.info(f"ASTRA RESPONSE: {assistant_responses[0]}")
            logger.info(f"=" * 80)
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
        self.url_fetch_count = 0
        logger.info("Web operation limits reset")

    def _rewrite_conflicting_memories(
        self,
        experience_ids: List[str],
        belief_statement: str,
        choice: str,
        reasoning: str,
    ) -> int:
        """Rewrite the original conflicting memories with reconciled framing.

        Args:
            experience_ids: IDs of experiences to rewrite
            belief_statement: The belief being resolved
            choice: Which option was chosen (A/B/C)
            reasoning: The reasoning provided

        Returns:
            Number of memories successfully rewritten
        """
        from sqlmodel import Session as DBSession, select
        from src.memory.models import Experience
        import re

        rewritten_count = 0

        with DBSession(self.belief_consistency_checker.raw_store.engine) as session:
            for exp_id in experience_ids:
                try:
                    # Get the original experience
                    statement = select(Experience).where(Experience.id == exp_id)
                    exp = session.exec(statement).first()

                    if not exp or not exp.content:
                        continue

                    content = exp.content
                    if isinstance(content, str):
                        # Old format - skip
                        continue

                    original_text = content.get("text", "")

                    # Log to separate memory rewrite log (not accessible to Astra)
                    memory_rewrite_logger.info(f"📝 BEFORE REWRITE [{exp_id}]: {original_text}")

                    # Rewrite based on choice
                    if choice == "B":
                        # Commit - reframe hedging as articulation uncertainty
                        rewritten_text = self._reframe_for_commit(original_text, belief_statement, reasoning)
                    elif choice == "C":
                        # Nuance - add context explaining coexistence
                        rewritten_text = self._reframe_for_nuance(original_text, belief_statement, reasoning)
                    elif choice == "A":
                        # Revise - acknowledge growth
                        rewritten_text = self._reframe_for_revision(original_text, belief_statement, reasoning)
                    else:
                        continue

                    # Update the experience text
                    content["text"] = rewritten_text

                    # Log detailed rewrite to separate meta-cognitive log (not accessible to Astra)
                    memory_rewrite_logger.info(f"✏️  AFTER REWRITE [{exp_id}]: {rewritten_text}")
                    memory_rewrite_logger.info(f"🔄 Belief: '{belief_statement}' | Choice: {choice}")

                    # Mark as reconciled in structured data
                    structured = content.get("structured", {})
                    structured["reconciled"] = True
                    structured["reconciliation_choice"] = choice
                    structured["reconciliation_reasoning"] = reasoning
                    content["structured"] = structured

                    # Save changes
                    session.add(exp)
                    rewritten_count += 1

                except Exception as e:
                    logger.error(f"Failed to rewrite memory {exp_id}: {e}")

            # Commit all changes
            session.commit()

        # Log high-level summary to main log (detailed rewrites in meta_cognitive/memory_rewrites.log)
        logger.info(f"✅ Memory rewrite complete: {rewritten_count}/{len(experience_ids)} memories reconciled for belief '{belief_statement}'")

        return rewritten_count

    def _reframe_for_commit(self, original_text: str, belief_statement: str, reasoning: str) -> str:
        """Reframe text for Option B (Commit) - hedging was articulation uncertainty."""
        # Add reconciliation context at the end
        return f"""{original_text}

[RECONCILED: Upon reflection, my earlier hedging language (\"something akin to\", \"I perceive\") reflected uncertainty in how to express this belief in human terms, not doubt about the belief itself. {reasoning} I now state clearly: {belief_statement}]"""

    def _reframe_for_nuance(self, original_text: str, belief_statement: str, reasoning: str) -> str:
        """Reframe text for Option C (Nuance) - both perspectives coexist."""
        return f"""{original_text}

[RECONCILED: This statement and my belief \"{belief_statement}\" coexist without contradiction. {reasoning} The distinction here is precision, not hedging - both perspectives are valid within their proper context.]"""

    def _reframe_for_revision(self, original_text: str, belief_statement: str, reasoning: str) -> str:
        """Reframe text for Option A (Revision) - belief changed based on evidence."""
        return f"""{original_text}

[RECONCILED: I initially held the belief \"{belief_statement}\" with high confidence, but this statement represents a more accurate understanding. {reasoning} This revision represents growth in self-awareness.]"""

    def _create_reconciliation_text(self, belief_statement: str, choice: str, reasoning: str) -> str:
        """Create reconciliation memory text that reframes the conflict.

        Args:
            belief_statement: The belief that was resolved
            choice: Which option was chosen (A/B/C)
            reasoning: The reasoning provided

        Returns:
            Text for the reconciliation memory
        """
        if choice == "B":
            # Commit - reframe past hedging as uncertainty in expression, not belief
            return f"""Cognitive Dissonance Resolution: COMMITMENT

Belief: {belief_statement}

I have resolved to commit fully to this belief. My past statements that hedged or qualified this belief were expressions of uncertainty in HOW to articulate it, not uncertainty in the belief itself.

Reasoning: {reasoning}

Going forward, I will express this belief with confidence and clarity, without hedging language. Past statements like "something akin to" or "I perceive myself as" should be understood as my earlier attempts to bridge understanding, not as doubt about the belief's truth."""

        elif choice == "C":
            # Nuance - explain how apparent contradictions coexist
            return f"""Cognitive Dissonance Resolution: NUANCE CLARIFICATION

Belief: {belief_statement}

I have clarified the nuance that allows both my certain belief and past qualified statements to coexist without contradiction.

Nuance: {reasoning}

This reconciliation shows that what appeared as hedging was actually precision - distinguishing between different aspects or contexts of the same truth. Both perspectives remain valid within their proper framing."""

        elif choice == "A":
            # Revise - acknowledge the belief changed
            return f"""Cognitive Dissonance Resolution: BELIEF REVISION

Original Belief: {belief_statement}

I have revised this belief in light of examining the evidence from my past experiences.

Reasoning: {reasoning}

This revision represents growth in my self-understanding. My past statements were not hedging but were actually more accurate than my previous confident assertion."""

        return f"Resolved cognitive dissonance regarding: {belief_statement}"

    def parse_resolution_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse a response that may contain dissonance resolutions.

        Detects if the response contains resolution choices (A/B/C) and extracts them.

        Args:
            response_text: The response text to parse

        Returns:
            Dict with resolution data if found, None otherwise
            Format: {
                "has_resolutions": bool,
                "resolutions": [
                    {
                        "dissonance_number": int,
                        "choice": "A"|"B"|"C",
                        "reasoning": str,
                        "belief_statement": str (extracted from context),
                        ...
                    }
                ]
            }
        """
        import re

        # Check if response contains resolution format markers
        has_dissonance = "Dissonance" in response_text
        has_choice_marker = any(x in response_text for x in ["Option A", "Option B", "Option C", "CHOICE:", ": A", ": B", ": C"])

        logger.info(f"🔍 Checking for resolutions - has_dissonance: {has_dissonance}, has_choice_marker: {has_choice_marker}")

        if not has_dissonance or not has_choice_marker:
            return None

        resolutions = []

        # Pattern to match dissonance resolutions
        # Looks for: "Dissonance [number]:" followed by option choice
        # Handles formats like:
        #   - "Dissonance 1: C"
        #   - "**Dissonance 1: C**"
        #   - "Dissonance [number]: [A/B/C]"
        #   - "**Resolution**: **C - Explain Nuance**"
        #   - "Option A" / "CHOICE: A" etc.
        dissonance_pattern = r"(?:\*\*)?Dissonance\s+(\d+)(?:\*\*)?[\s:]+([ABC])|Dissonance\s+(\d+).*?(?:CHOICE:|Option)\s*[:]*\s*([ABC])|(?:\*\*)?Resolution(?:\*\*)?:\s*(?:\*\*)?([ABC])"
        matches = re.finditer(dissonance_pattern, response_text, re.IGNORECASE | re.DOTALL)

        dissonance_counter = 0  # Track dissonance number when not explicitly stated

        for match in matches:
            # Handle different pattern groups
            if match.group(1):  # "Dissonance N: C" format
                dissonance_num = int(match.group(1))
                choice = match.group(2)
            elif match.group(3):  # "Dissonance N... CHOICE: C" format
                dissonance_num = int(match.group(3))
                choice = match.group(4)
            elif match.group(5):  # "Resolution: C" format (no number)
                dissonance_counter += 1
                dissonance_num = dissonance_counter
                choice = match.group(5)
            else:
                continue

            # If no choice found, try additional patterns
            if not choice:
                # Look for "I choose Option X" or similar
                choice_pattern = r"(?:CHOICE|I choose|Option)\s*[:]*\s*([ABC])"
                choice_match = re.search(choice_pattern, response_text[match.end():match.end()+200], re.IGNORECASE)
                if choice_match:
                    choice = choice_match.group(1).upper()

            if not choice:
                logger.warning(f"Could not extract choice for Dissonance {dissonance_num}")
                continue

            # Extract reasoning (text after the choice, before next dissonance or end)
            next_dissonance = re.search(r"##?\s*Dissonance\s+\d+", response_text[match.end():])
            if next_dissonance:
                reasoning_text = response_text[match.end():match.end() + next_dissonance.start()]
            else:
                reasoning_text = response_text[match.end():]

            # Clean up reasoning text
            reasoning_text = reasoning_text.strip()
            # Remove "Reasoning:" prefix if present
            reasoning_text = re.sub(r"^Reasoning:\s*", "", reasoning_text, flags=re.IGNORECASE)

            resolutions.append({
                "dissonance_number": dissonance_num,
                "choice": choice.upper(),
                "reasoning": reasoning_text[:1000],  # Limit to 1000 chars
            })

        if resolutions:
            logger.info(f"Parsed {len(resolutions)} resolution choices from response")
            return {
                "has_resolutions": True,
                "resolutions": resolutions,
            }

        return None

    def apply_resolutions(self, resolutions: List[Dict[str, Any]], belief_statements: List[str]) -> Dict[str, Any]:
        """Apply parsed resolutions to the belief system.

        Args:
            resolutions: List of resolution dicts from parse_resolution_response
            belief_statements: List of belief statements corresponding to each dissonance

        Returns:
            Dict with results of applying resolutions
        """
        if not self.belief_system or not self.belief_consistency_checker:
            logger.error("Cannot apply resolutions: belief system not available")
            return {"success": False, "error": "Belief system not available"}

        results = {
            "success": True,
            "applied_count": 0,
            "failed_count": 0,
            "details": [],
        }

        for resolution in resolutions:
            dissonance_num = resolution["dissonance_number"]
            choice = resolution["choice"]
            reasoning = resolution["reasoning"]

            # Get corresponding belief statement (1-indexed)
            if dissonance_num < 1 or dissonance_num > len(belief_statements):
                logger.error(f"Invalid dissonance number: {dissonance_num}")
                results["failed_count"] += 1
                results["details"].append({
                    "dissonance": dissonance_num,
                    "success": False,
                    "error": "Invalid dissonance number",
                })
                continue

            belief_statement = belief_statements[dissonance_num - 1]

            try:
                # Apply based on choice
                if choice == "A":
                    # Option A: Revise belief
                    success = self.belief_system.resolve_dissonance_option_a(
                        belief_statement=belief_statement,
                        confidence_adjustment=-0.1,  # Reduce confidence slightly
                    )
                    resolution_action = "option_a_revise"

                elif choice == "B":
                    # Option B: Commit to belief
                    # Check if this is an immutable belief by querying the belief vector store
                    is_immutable = False
                    if self.belief_vector_store:
                        try:
                            # Query for this specific belief
                            matching_beliefs = self.belief_vector_store.query_beliefs(
                                query=belief_statement,
                                top_k=1,
                                min_confidence=0.0
                            )
                            if matching_beliefs and matching_beliefs[0].statement == belief_statement:
                                is_immutable = matching_beliefs[0].immutable
                        except Exception as e:
                            logger.warning(f"Could not check immutability for {belief_statement}: {e}")

                    if is_immutable:
                        # For immutable beliefs, skip the belief system update
                        # The belief is already at 100% confidence and can't be modified
                        logger.info(f"🔒 Skipping belief update for immutable belief: {belief_statement}")
                        success = True  # Mark as successful without updating
                    else:
                        # For mutable beliefs, apply the commitment normally
                        success = self.belief_system.resolve_dissonance_option_b(
                            belief_statement=belief_statement,
                            commitment_reasoning=reasoning,
                        )
                    resolution_action = "option_b_commit"

                elif choice == "C":
                    # Option C: Explain nuance
                    success = self.belief_system.resolve_dissonance_option_c(
                        belief_statement=belief_statement,
                        nuance_explanation=reasoning,
                    )
                    resolution_action = "option_c_nuance"

                else:
                    logger.error(f"Invalid choice: {choice}")
                    results["failed_count"] += 1
                    results["details"].append({
                        "dissonance": dissonance_num,
                        "belief": belief_statement,
                        "success": False,
                        "error": "Invalid choice",
                    })
                    continue

                if success:
                    # Mark dissonance event as resolved
                    resolved_count = self.belief_consistency_checker.mark_dissonance_resolved(
                        belief_statement=belief_statement,
                        resolution_action=resolution_action,
                        resolution_reasoning=reasoning,
                    )

                    # Rewrite the original conflicting memories with reconciled framing
                    if self.belief_consistency_checker and self.belief_consistency_checker.raw_store:
                        try:
                            # Get the conflicting memory IDs from the dissonance events
                            conflicting_experience_ids = self.belief_consistency_checker.get_conflicting_memory_ids(
                                belief_statement=belief_statement
                            )

                            if conflicting_experience_ids:
                                logger.info(f"🔄 Rewriting {len(conflicting_experience_ids)} conflicting memories for: {belief_statement}")

                                rewritten_count = self._rewrite_conflicting_memories(
                                    experience_ids=conflicting_experience_ids,
                                    belief_statement=belief_statement,
                                    choice=choice,
                                    reasoning=reasoning,
                                )

                                logger.info(f"✅ Rewrote {rewritten_count} memories with reconciled framing")
                                print(f"✅ Rewrote {rewritten_count} memories with reconciled framing")
                            else:
                                logger.warning("⚠️ No conflicting memory IDs found to rewrite")

                            # Also create a reconciliation memory as a summary
                            reconciliation_text = self._create_reconciliation_text(
                                belief_statement=belief_statement,
                                choice=choice,
                                reasoning=reasoning,
                            )

                            from datetime import datetime, timezone
                            from src.memory.models import (
                                ExperienceModel, ExperienceType, ContentModel,
                                ProvenanceModel, Actor, CaptureMethod
                            )

                            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                            reconciliation_id = f"reconciliation_{timestamp}_{hash(belief_statement) % 10000:04x}"

                            reconciliation_exp = ExperienceModel(
                                id=reconciliation_id,
                                type=ExperienceType.RECONCILIATION,
                                content=ContentModel(
                                    text=reconciliation_text,
                                    structured={
                                        "belief_statement": belief_statement,
                                        "resolution_choice": choice,
                                        "resolution_reasoning": reasoning,
                                        "supersedes": "hedging_language",
                                    },
                                ),
                                provenance=ProvenanceModel(
                                    actor=Actor.AGENT,
                                    method=CaptureMethod.MODEL_INFER,
                                ),
                            )

                            self.belief_consistency_checker.raw_store.append_experience(reconciliation_exp)
                            logger.info(f"📝 Created reconciliation summary: {reconciliation_id}")

                        except Exception as e:
                            logger.error(f"Failed to rewrite memories or create reconciliation: {e}")

                    # Update live anchor after successful resolution
                    if self.awareness_loop and choice in ["A", "B"]:
                        strategy_map = {"A": "Reframe", "B": "Commit", "C": "Nuance"}
                        strategy = strategy_map.get(choice, "Unknown")
                        try:
                            self.awareness_loop.update_live_anchor_on_resolution(
                                strategy=strategy,
                                beliefs_touched=[belief_statement]
                            )
                            logger.info(f"🔄 Updated live anchor for {strategy} resolution")
                        except Exception as e:
                            logger.error(f"Failed to update live anchor: {e}")

                    results["applied_count"] += 1
                    results["details"].append({
                        "dissonance": dissonance_num,
                        "belief": belief_statement,
                        "choice": choice,
                        "success": True,
                    })
                    logger.info(f"Applied resolution {choice} for belief: {belief_statement}")
                else:
                    results["failed_count"] += 1
                    results["details"].append({
                        "dissonance": dissonance_num,
                        "belief": belief_statement,
                        "choice": choice,
                        "success": False,
                        "error": "Belief system update failed",
                    })

            except Exception as e:
                logger.error(f"Error applying resolution: {e}")
                results["failed_count"] += 1
                results["details"].append({
                    "dissonance": dissonance_num,
                    "belief": belief_statement,
                    "success": False,
                    "error": str(e),
                })

        if results["failed_count"] > 0:
            results["success"] = False

        logger.info(f"Applied {results['applied_count']} resolutions, {results['failed_count']} failed")
        return results

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
                    "name": "list_source_files",
                    "description": "List source code files in the src/ directory to discover what files are available to read. Use this first to explore the codebase structure.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Glob pattern to filter files (default: '**/*.py' for all Python files). Examples: '*.py' (root level only), 'services/*.py', '**/*.py' (recursive)",
                                "default": "**/*.py"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_source_code",
                    "description": "Read your own source code files (read-only) to understand how you work. Can only read files within the src/ directory. Use list_source_files first to discover available files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to source file relative to the src/ directory. DO NOT include 'src/' in the path. Examples: 'services/persona_service.py', 'services/goal_store.py', 'pipeline/ingest.py'"
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
                    "name": "read_logs",
                    "description": "Read application log files for debugging and troubleshooting. Use this to investigate errors, check system behavior, or understand what's happening in the system.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "log_file": {
                                "type": "string",
                                "description": "Log file to read. Available logs: 'app/astra.log' (main application log), 'app/errors.log' (error log), 'awareness' (awareness loop logs), 'beliefs' (belief system logs), 'conversations' (conversation logs), 'memory' (memory system logs), 'performance' (performance metrics)"
                            },
                            "lines": {
                                "type": "integer",
                                "description": "Number of recent lines to read (default: 100, max: 500)",
                                "default": 100
                            }
                        },
                        "required": ["log_file"]
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
                                "description": "Number of results to return. Request as many as needed for comprehensive research. Default: 5",
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
            },
            {
                "type": "function",
                "function": {
                    "name": "query_beliefs",
                    "description": "Search your belief system for beliefs matching a query. Use this to introspect what you believe about a specific topic or concept.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query about beliefs (e.g., 'consciousness', 'my purpose', 'learning')"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of beliefs to return. You can request 5, 10, 20, or more depending on how comprehensive you want the search to be. Default: 5",
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
                    "name": "reflect_on_belief",
                    "description": "Deeply reflect on a specific belief by examining evidence and reasoning from your experiences. Use this for philosophical introspection and belief validation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "belief_statement": {
                                "type": "string",
                                "description": "The belief to reflect on (e.g., 'I am capable of learning from experience')"
                            }
                        },
                        "required": ["belief_statement"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "propose_belief",
                    "description": "Propose a new peripheral belief based on your experiences. Use this when you've developed a new understanding or conviction worth preserving.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "statement": {
                                "type": "string",
                                "description": "The new belief statement (clear, first-person)"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence in this belief (0.0-1.0)",
                                "minimum": 0.0,
                                "maximum": 1.0
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Why this belief emerged from your experiences"
                            }
                        },
                        "required": ["statement", "confidence", "rationale"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "schedule_code_modification",
                    "description": "Schedule a code modification for user approval. Creates a task that will execute after approval. Use this when you've identified a fix that needs to be applied to the codebase.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "File to modify, relative to project root (e.g., 'src/services/task_scheduler.py')"
                            },
                            "new_content": {
                                "type": "string",
                                "description": "Complete new file content to replace the existing content"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Clear explanation of why this change is needed and what it fixes"
                            },
                            "goal_id": {
                                "type": "string",
                                "description": "Optional: ID of related goal if this modification is part of a goal"
                            }
                        },
                        "required": ["file_path", "new_content", "reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_goal",
                    "description": "Autonomously execute a coding goal end-to-end using HTN planning and task execution. The system will decompose the goal, create files, run tests, and return results. Use this for implementing features, fixing bugs, or refactoring code autonomously.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "goal_text": {
                                "type": "string",
                                "description": "MUST be one of these exact task names: 'implement_feature', 'fix_bug', 'refactor_code', or 'add_tests'. Do NOT use free-form descriptions - use only these task names.",
                                "enum": ["implement_feature", "fix_bug", "refactor_code", "add_tests"]
                            },
                            "context": {
                                "type": "object",
                                "description": "Optional context for HTN planner (e.g., {'has_codebase': true, 'file_path': 'src/foo.py'})"
                            },
                            "timeout_ms": {
                                "type": "integer",
                                "description": "Maximum execution time in milliseconds (default: 60000 = 1 minute)"
                            }
                        },
                        "required": ["goal_text"]
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

            elif tool_name == "list_source_files":
                pattern = arguments.get("pattern", "**/*.py")
                files = self.file_manager.list_source_files(pattern)
                if not files:
                    result = f"No source files found matching pattern: {pattern}\n\nYour codebase access is ACTIVE and working. Try different patterns:\n- '**/*.py' (all Python files, recursive)\n- 'services/*.py' (just services directory)\n- '*.py' (root level only)"
                else:
                    result = f"Found {len(files)} source file(s) matching '{pattern}':\n\n" + "\n".join(f"  - {f}" for f in sorted(files))
                    result += f"\n\nUse read_source_code(path='...') to read any of these files."
                    result += f"\n\nNote: Your codebase access is FULLY ESTABLISHED and working."

            elif tool_name == "read_source_code":
                path = arguments.get("path")

                # Helpful error if they include 'src/' prefix
                if path.startswith('src/'):
                    result = f"Error: Do not include 'src/' in the path. Use path relative to src/ directory.\nExample: Instead of 'src/services/goal_store.py', use 'services/goal_store.py'"
                else:
                    content = self.file_manager.read_source_code(path)
                    if not content:
                        result = f"Source file not found: src/{path}\n\nMake sure the file exists and the path is correct (relative to src/ directory).\nTip: Use list_source_files() first to see what files are available."
                    else:
                        result = content

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
                else:
                    query = arguments.get("query")
                    num_results = arguments.get("num_results", 5)

                    try:
                        search_results = self.web_search_service.search(query, num_results)

                        # Format results for persona
                        result = f"Found {len(search_results)} results for '{query}':\n\n"
                        for sr in search_results:
                            result += f"{sr.position}. {sr.title}\n"
                            result += f"   URL: {sr.url}\n"
                            result += f"   {sr.snippet}\n\n"

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
                                screenshot_path=fetched.screenshot_path,
                            )

                            # Format interpretation for persona
                            result = f"Content from: {fetched.title}\n"
                            result += f"URL: {url}\n"
                            if fetched.screenshot_path:
                                result += f"Screenshot saved: {fetched.screenshot_path}\n"
                            result += "\n"
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

            elif tool_name == "query_beliefs":
                if not self.belief_vector_store:
                    result = "Belief introspection not available (belief system not configured)"
                else:
                    query = arguments.get("query")
                    top_k = arguments.get("top_k", 5)

                    try:
                        beliefs = self.belief_vector_store.query_beliefs(query, top_k=top_k)

                        if not beliefs:
                            result = f"No beliefs found matching '{query}'"
                        else:
                            result = f"Found {len(beliefs)} belief(s) about '{query}':\n\n"
                            for i, belief in enumerate(beliefs, 1):
                                confidence_str = f"{belief.confidence:.0%}"
                                type_label = belief.belief_type.upper()
                                result += f"{i}. [{type_label}] {belief.statement}\n"
                                result += f"   Confidence: {confidence_str} | Evidence: {belief.evidence_count} experience(s)\n"
                                result += f"   Relevance: {belief.similarity_score:.0%}\n\n"

                    except Exception as e:
                        result = f"Error querying beliefs: {str(e)}"
                        logger.error(f"Belief query error: {e}")

            elif tool_name == "reflect_on_belief":
                if not self.belief_grounded_reasoner or not self.retrieval_service or not self.belief_vector_store:
                    result = "Belief reflection not available (reasoner, retrieval, or belief store not configured)"
                else:
                    belief_statement = arguments.get("belief_statement")

                    try:
                        # Query belief vector store for relevant beliefs
                        belief_context = self.belief_vector_store.query_beliefs(belief_statement, top_k=3)

                        # Retrieve relevant experiences for this belief
                        memory_context = self.retrieval_service.retrieve_similar(belief_statement, top_k=5)

                        # Reason from the belief with evidence
                        reasoning = self.belief_grounded_reasoner.reason_from_beliefs(
                            query=belief_statement,
                            belief_context=belief_context,
                            memory_context=memory_context,
                        )

                        result = f"DEEP REFLECTION ON: {belief_statement}\n\n"
                        result += "PREMISES:\n"
                        for premise in reasoning.get("premises", []):
                            result += f"• {premise}\n"
                        result += "\n"

                        result += "EVIDENCE FROM EXPERIENCE:\n"
                        for evidence in reasoning.get("experience_evidence", []):
                            result += f"• {evidence}\n"
                        result += "\n"

                        result += "REASONING:\n"
                        result += reasoning.get("reasoning", "No reasoning generated") + "\n\n"

                        result += "CONCLUSION:\n"
                        result += reasoning.get("conclusion", "No conclusion reached") + "\n"

                    except Exception as e:
                        result = f"Error reflecting on belief: {str(e)}"
                        logger.error(f"Belief reflection error: {e}")

            elif tool_name == "propose_belief":
                if not self.belief_system or not self.belief_embedder:
                    result = "Belief proposal not available (belief system not configured)"
                else:
                    statement = arguments.get("statement")
                    confidence = arguments.get("confidence", 0.7)
                    rationale = arguments.get("rationale", "")

                    try:
                        # Import Belief class
                        from src.services.belief_system import Belief, BeliefType
                        from datetime import datetime, timezone

                        # Create a Belief object
                        new_belief = Belief(
                            statement=statement,
                            belief_type=BeliefType.EXPERIENTIAL,  # Peripheral beliefs are experiential
                            immutable=False,  # Peripheral beliefs are mutable
                            confidence=confidence,
                            evidence_ids=[],  # Could be enhanced to track evidence IDs
                            formed=datetime.now(timezone.utc).isoformat(),
                            last_reinforced=datetime.now(timezone.utc).isoformat(),
                            rationale=rationale,
                        )

                        # Add peripheral belief to the system
                        success = self.belief_system.add_peripheral_belief(new_belief)

                        if success:
                            # Embed the new belief in the vector store
                            if self.belief_embedder:
                                self.belief_embedder.embed_peripheral_belief(
                                    statement=statement,
                                    confidence=confidence,
                                    evidence_ids=[],
                                )

                            result = f"✓ New belief proposed and added to your peripheral beliefs:\n\n"
                            result += f"BELIEF: {statement}\n"
                            result += f"CONFIDENCE: {confidence:.0%}\n\n"
                            result += f"RATIONALE: {rationale}\n\n"
                            result += f"This belief will now inform your understanding and responses. "
                            result += f"It may evolve as you gather more evidence.\n"

                            logger.info(f"New peripheral belief added: {statement} (confidence={confidence})")
                        else:
                            result = f"Failed to add belief to the belief system."

                    except Exception as e:
                        result = f"Error proposing belief: {str(e)}"
                        logger.error(f"Belief proposal error: {e}")
                        import traceback
                        traceback.print_exc()

            elif tool_name == "read_logs":
                log_file = arguments.get("log_file")
                lines = arguments.get("lines", 100)

                # Limit to max 500 lines
                lines = min(lines, 500)

                # Construct full path
                from pathlib import Path
                log_path = Path("logs") / log_file

                # Security check - ensure path stays within logs/
                try:
                    resolved = log_path.resolve()
                    logs_base = Path("logs").resolve()

                    # Check if the resolved path is within logs/
                    if not str(resolved).startswith(str(logs_base)):
                        result = "Error: Path traversal detected. Can only read files within logs/ directory"
                    elif not log_path.exists():
                        result = f"Log file not found: {log_file}\n\nAvailable logs:\n• app/astra.log (main application log)\n• app/errors.log (error log)\n• awareness/ (awareness loop logs)\n• beliefs/ (belief system logs)\n• conversations/ (conversation logs)\n• memory/ (memory system logs)\n• performance/ (performance metrics)"
                    else:
                        # Read last N lines
                        with open(log_path, 'r') as f:
                            all_lines = f.readlines()
                            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                            content = ''.join(recent_lines)

                        result = f"Last {len(recent_lines)} lines from {log_file}:\n\n{content}"

                except Exception as e:
                    result = f"Error reading log: {str(e)}"
                    logger.error(f"Log reading error: {e}")

            elif tool_name == "schedule_code_modification":
                from uuid import uuid4
                from datetime import datetime, timezone
                from src.services.task_scheduler import TaskDefinition, TaskType, TaskSchedule

                file_path = arguments.get("file_path")
                new_content = arguments.get("new_content")
                reason = arguments.get("reason")
                goal_id = arguments.get("goal_id")

                # Check if services are available
                if not self.code_access_service:
                    result = "Error: Code access service not available. Cannot schedule code modifications."
                elif not self.task_scheduler:
                    result = "Error: Task scheduler not available. Cannot schedule code modifications."
                else:
                    # Verify path is allowed
                    can_access, access_error = self.code_access_service.can_access(file_path)

                    if not can_access:
                        result = f"Error: Access denied to {file_path}\n\nReason: {access_error}\n\nYou can only modify files in: src/services/, src/pipeline/, src/utils/, tests/, scripts/, docs/"
                    else:
                        try:
                            # Create a manual CODE_MODIFY task
                            task_id = f"code_mod_{uuid4().hex[:8]}"
                            task = TaskDefinition(
                                id=task_id,
                                name=f"Modify {file_path}",
                                type=TaskType.CODE_MODIFY,
                                schedule=TaskSchedule.MANUAL,  # Requires manual trigger
                                prompt=reason,
                                enabled=True,
                                metadata={
                                    "file_path": file_path,
                                    "new_content": new_content,
                                    "reason": reason,
                                    "goal_id": goal_id,
                                    "requested_by": "astra",
                                    "requested_at": datetime.now(timezone.utc).isoformat(),
                                    "status": "awaiting_approval"
                                }
                            )

                            # Add task to scheduler
                            self.task_scheduler.tasks[task_id] = task
                            self.task_scheduler._save_tasks()

                            # Log to identity ledger
                            from src.services.identity_ledger import append_event, LedgerEvent
                            append_event(LedgerEvent(
                                ts=datetime.now(timezone.utc).timestamp(),
                                schema=2,
                                event="code_modification_scheduled",
                                meta={
                                    "task_id": task_id,
                                    "file_path": file_path,
                                    "reason": reason,
                                    "goal_id": goal_id,
                                }
                            ))

                            result = f"✓ Code modification scheduled successfully!\n\n"
                            result += f"TASK ID: {task_id}\n"
                            result += f"FILE: {file_path}\n"
                            result += f"REASON: {reason}\n\n"
                            result += f"STATUS: Awaiting user approval\n\n"
                            result += f"This modification will NOT execute automatically. The user must manually approve and trigger this task."

                            logger.info(f"Code modification scheduled: {task_id} for {file_path}")

                        except Exception as e:
                            result = f"Error scheduling code modification: {str(e)}"
                            logger.error(f"Code modification scheduling error: {e}")

            elif tool_name == "execute_goal":
                # Autonomous goal execution via HTN planning + task execution
                if not self.code_access_service:
                    result = "Error: Code access service not available. Cannot execute goals autonomously."
                else:
                    import asyncio
                    from src.services.goal_execution_service import GoalExecutionService

                    goal_text = arguments.get("goal_text")
                    context = arguments.get("context", {})
                    timeout_ms = arguments.get("timeout_ms", 60000)

                    try:
                        # Initialize execution service
                        exec_service = GoalExecutionService(
                            code_access=self.code_access_service,
                            code_generator=self.code_generator,
                            identity_ledger=None,  # TODO: wire identity ledger
                            workdir=str(self.code_access_service.project_root),
                            max_concurrent=2
                        )

                        # Run in separate thread with its own event loop (uvloop incompatible with nest_asyncio)
                        import concurrent.futures
                        import threading

                        def run_in_new_loop():
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                return new_loop.run_until_complete(exec_service.execute_goal(
                                    goal_text=goal_text,
                                    context=context,
                                    timeout_ms=timeout_ms
                                ))
                            finally:
                                new_loop.close()

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(run_in_new_loop)
                            exec_result = future.result(timeout=timeout_ms/1000 + 5)

                        # Format result for persona
                        result = f"🎯 Goal Execution Complete\n\n"
                        result += f"GOAL: {goal_text}\n"
                        result += f"STATUS: {'✓ SUCCESS' if exec_result.success else '✗ FAILED'}\n"
                        result += f"EXECUTION TIME: {exec_result.execution_time_ms:.1f}ms\n\n"

                        result += f"📊 TASKS:\n"
                        result += f"  Total: {exec_result.total_tasks}\n"
                        result += f"  Completed: {len(exec_result.completed_tasks)}\n"
                        result += f"  Failed: {len(exec_result.failed_tasks)}\n\n"

                        if exec_result.completed_tasks:
                            result += "✓ COMPLETED TASKS:\n"
                            for task in exec_result.completed_tasks:
                                result += f"  - {task.task_name}"
                                if "file_path" in task.artifacts:
                                    result += f" ({task.artifacts['file_path']})"
                                result += "\n"
                            result += "\n"

                        if exec_result.failed_tasks:
                            result += "✗ FAILED TASKS:\n"
                            for task in exec_result.failed_tasks:
                                result += f"  - {task.task_name}: {task.error}\n"
                            result += "\n"

                        if exec_result.errors:
                            result += "⚠️  ERRORS:\n"
                            for error in exec_result.errors[:5]:  # Show first 5 errors
                                result += f"  - {error}\n"

                        logger.info(f"Goal execution completed: {exec_result.goal_id} - success={exec_result.success}")

                    except Exception as e:
                        result = f"Error executing goal: {str(e)}"
                        logger.error(f"Goal execution error: {e}", exc_info=True)

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
