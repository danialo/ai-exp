"""Manual test for execute_goal tool in persona service.

Tests the integration of GoalExecutionService as a callable tool.
"""
import asyncio
from pathlib import Path
from src.services.persona_service import PersonaService
from src.services.llm import create_llm_service
from src.services.code_access import create_code_access_service

def test_execute_goal_tool():
    """Test execute_goal tool via persona_service._execute_tool."""

    # Initialize services
    project_root = Path("/home/d/git/ai-exp")
    code_access = create_code_access_service(
        project_root=project_root,
        max_file_size_kb=100,
        auto_branch=True
    )

    # Initialize persona service (minimal setup)
    llm_service = create_llm_service()
    persona_service = PersonaService(
        llm_service=llm_service,
        persona_space_path="personal_space",
        code_access_service=code_access
    )

    # Call execute_goal tool directly
    print("\n🧪 Testing execute_goal tool...")
    print("=" * 60)

    arguments = {
        "goal_text": "implement_feature",
        "context": {},
        "timeout_ms": 60000
    }

    result = persona_service._execute_tool("execute_goal", arguments)

    print("\n📋 RESULT:")
    print(result)
    print("=" * 60)

    # Verify files were created
    feature_files = list(Path("tests/generated").glob("feature_*.py"))
    test_files = list(Path("tests/generated").glob("test_*.py"))

    print(f"\n✅ Files created:")
    print(f"  Feature files: {len(feature_files)}")
    print(f"  Test files: {len(test_files)}")

    if feature_files:
        print(f"\n  Latest feature file: {feature_files[-1].name}")
        print(f"  Content preview:")
        with open(feature_files[-1]) as f:
            print(f"    {f.read()[:100]}...")

if __name__ == "__main__":
    test_execute_goal_tool()
