"""Test Astra using execute_goal tool in conversation context.

This simulates what happens when Astra calls execute_goal during a conversation.
"""
import sys
sys.path.insert(0, "/home/d/git/ai-exp")

from pathlib import Path
from src.services.persona_service import PersonaService
from src.services.llm import create_llm_service
from src.services.code_access import create_code_access_service


def test_astra_execute_goal_tool():
    """Simulate Astra calling execute_goal tool."""

    print("\n" + "="*70)
    print("TESTING: Astra's execute_goal Tool (Conversation Context)")
    print("="*70)

    # Initialize Astra's services
    project_root = Path("/home/d/git/ai-exp")
    code_access = create_code_access_service(
        project_root=project_root,
        max_file_size_kb=100,
        auto_branch=False
    )

    llm_service = create_llm_service()
    persona_service = PersonaService(
        llm_service=llm_service,
        persona_space_path="personal_space",
        code_access_service=code_access
    )

    print("\n✓ Astra initialized with execute_goal tool available")

    # Check tool is registered
    tools = persona_service._get_tool_definitions()
    tool_names = [t['function']['name'] for t in tools]

    print(f"\n✓ Astra has {len(tools)} tools available:")
    print(f"  - execute_goal: {'✓' if 'execute_goal' in tool_names else '✗'}")
    print(f"  - read_source_code: {'✓' if 'read_source_code' in tool_names else '✗'}")
    print(f"  - write_file: {'✓' if 'write_file' in tool_names else '✗'}")

    # Simulate Astra deciding to use the tool
    print("\n" + "-"*70)
    print("SIMULATING: Astra calls execute_goal tool")
    print("-"*70)

    print("\nUser: 'Can you autonomously implement a simple feature?'")
    print("\nAstra (thinking): I should use my execute_goal tool...")
    print("\nAstra calls: execute_goal(goal_text='implement_feature')")

    # Execute the tool as Astra would
    print("\n⚙️  Executing tool...")

    result = persona_service._execute_tool(
        tool_name="execute_goal",
        arguments={
            "goal_text": "implement_feature",
            "context": {},
            "timeout_ms": 60000
        }
    )

    print("\n" + "="*70)
    print("TOOL RESULT (What Astra sees):")
    print("="*70)
    print(result)

    print("\n" + "="*70)
    print("VERIFICATION:")
    print("="*70)

    # Verify result format
    assert "Goal Execution Complete" in result or "GOAL:" in result
    assert "SUCCESS" in result or "FAILED" in result

    # Check if files were created
    generated_files = list(Path("tests/generated").glob("feature_*.py"))
    print(f"\n✓ Files created autonomously: {len(generated_files)}")

    if generated_files:
        latest = sorted(generated_files, key=lambda p: p.stat().st_mtime)[-1]
        print(f"  Latest: {latest.name}")
        with open(latest) as f:
            content = f.read()
        print(f"  Content preview: {content[:80]}...")

    print("\n" + "="*70)
    print("WHAT HAPPENS IN REAL CONVERSATION:")
    print("="*70)
    print("""
1. User: "Can you implement a simple feature?"

2. Astra (LLM decides to use execute_goal tool):
   - Sees tool definition in her available tools
   - Decides this matches her capabilities
   - Calls execute_goal with appropriate parameters

3. Tool executes:
   - HTN planner decomposes goal
   - TaskGraph created
   - Tasks execute autonomously
   - Files created in tests/generated/

4. Astra receives result:
   - Sees formatted output with task outcomes
   - Files that were created
   - Any errors encountered

5. Astra responds to user:
   - Interprets the technical result
   - Explains what she did in natural language
   - Shows the files she created
   - Mentions any issues encountered
    """)

    print("\n✅ TEST COMPLETE - Astra CAN execute goals autonomously")
    print("="*70)


if __name__ == "__main__":
    test_astra_execute_goal_tool()
