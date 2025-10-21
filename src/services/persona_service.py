"""
Persona Service - Orchestrates the self-modifying persona system.

This service coordinates:
1. Persona prompt building (with self-concept)
2. LLM generation (with emotional co-analysis)
3. Emotional reconciliation (internal vs external)
4. File system access for persona autonomy
"""

from typing import Dict, Optional, Tuple
from src.services.persona_prompt import PersonaPromptBuilder, extract_emotional_assessment, remove_emotional_assessment
from src.services.emotional_reconciler import EmotionalReconciler
from src.services.persona_file_manager import PersonaFileManager
from src.services.llm import LLMService


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

    def generate_response(self, user_message: str) -> Tuple[str, Dict]:
        """
        Generate a persona response with emotional co-analysis.

        This is the main method that:
        1. Builds persona-aware prompt
        2. Generates response with internal emotional assessment
        3. Extracts and reconciles emotional perspectives
        4. Returns cleaned response and reconciliation data

        Args:
            user_message: The user's message

        Returns:
            Tuple of (response_text, reconciliation_data)
        """
        # Build the persona prompt with current context
        full_prompt = self.prompt_builder.build_prompt(user_message)

        # Generate response (includes internal emotional assessment)
        raw_response = self.llm.generate(
            prompt=full_prompt,
            temperature=0.9,  # Higher for natural personality expression
            max_tokens=1000
        )

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
