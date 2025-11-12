"""Direct test of execute_goal functionality."""
import asyncio
import sys
sys.path.insert(0, '/home/d/git/ai-exp')

from src.services.goal_execution_service import GoalExecutionService
from src.services.code_access import CodeAccessService
from src.services.llm import LLMService
from src.services.code_generator import CodeGenerator, LLMServiceWrapper
from config.settings import settings

async def test_execute_goal():
    """Test execute_goal directly."""
    print("=" * 80)
    print("DIRECT TEST: execute_goal functionality")
    print("=" * 80)

    # Initialize services
    print("\n1. Initializing services...")
    code_access = CodeAccessService(project_root="/home/d/git/ai-exp")

    # Use the existing LLM service (Venice AI / OpenAI-compatible)
    llm_service = LLMService(
        api_key=settings.VENICEAI_API_KEY or settings.OPENAI_API_KEY,
        model=settings.LLM_MODEL,
        base_url=settings.LLM_BASE_URL,
        temperature=0.2  # Lower temp for code generation
    )
    llm_wrapper = LLMServiceWrapper(llm_service)
    code_generator = CodeGenerator(llm=llm_wrapper)

    exec_service = GoalExecutionService(
        code_access=code_access,
        code_generator=code_generator,
        workdir="/home/d/git/ai-exp"
    )

    print("✓ Services initialized")

    # Test execute_goal
    print("\n2. Calling execute_goal...")
    print("   Goal: implement_feature")
    print("   Feature: Simple calculator with add/subtract/multiply/divide")

    result = await exec_service.execute_goal(
        goal_text="implement_feature",
        context={
            "feature_name": "simple_calculator",
            "description": "A calculator function with add, subtract, multiply, divide operations"
        },
        timeout_ms=120000
    )

    print("\n3. Results:")
    print(f"   Success: {result.success}")
    print(f"   Total tasks: {result.total_tasks}")
    print(f"   Completed: {result.completed_tasks}")
    print(f"   Failed: {result.failed_tasks}")

    if result.files_created:
        print(f"\n4. Files created:")
        for f in result.files_created:
            print(f"   - {f}")

    if result.errors:
        print(f"\n❌ Errors:")
        for e in result.errors:
            print(f"   - {e}")

    print("\n" + "=" * 80)
    return result

if __name__ == "__main__":
    result = asyncio.run(test_execute_goal())
    print(f"\nTest {'PASSED ✓' if result.success else 'FAILED ✗'}")
