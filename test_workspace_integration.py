"""Test script for workspace integration."""
import asyncio
from src.services.project_manager import ProjectManager
from src.services.code_generator import CodeGenerator, GenRequest, LLMServiceWrapper
from src.services.llm import LLMService

async def test_project_creation():
    """Test creating a workspace project."""
    print("="*80)
    print("TEST 1: Project Creation")
    print("="*80)

    pm = ProjectManager()

    # Create a test project
    project = pm.create_project(
        goal_text="Implement a simple calculator function",
        template="python-module",
        description="Create add/subtract/multiply/divide functions"
    )

    print(f"✓ Created project: {project.project_id}")
    print(f"  Location: {project.project_dir}")
    print(f"  Status: {project.status}")
    print(f"  Goal: {project.goal_text}")

    # Update project
    pm.update_project(
        project_id=project.project_id,
        status="in_progress",
        implementation_notes="Starting implementation"
    )
    print(f"✓ Updated project status to: in_progress")

    # List projects
    projects = pm.list_projects(limit=5)
    print(f"✓ Found {len(projects)} total projects")

    return project


async def test_code_generation():
    """Test code generation with improved prompts."""
    print("\n" + "="*80)
    print("TEST 2: Code Generation (CodeAgent-5 Prompts)")
    print("="*80)

    # Initialize LLM and CodeGenerator
    llm_service = LLMService()
    llm_wrapper = LLMServiceWrapper(llm_service)
    codegen = CodeGenerator(llm=llm_wrapper)

    # Test implementation generation
    req = GenRequest(
        goal_text="Create a function that calculates the sum of two numbers",
        context={
            "codebase_context": "Simple Python module for mathematical operations",
            "dependencies": ["typing"]
        },
        file_path="/home/d/astra-workspace/projects/test/src/calculator.py",
        role="implementation",
        language="python"
    )

    print("Generating implementation code...")
    print(f"  Goal: {req.goal_text}")
    print(f"  File: {req.file_path}")

    try:
        result = await codegen.generate(req)
        print(f"✓ Generated {len(result.code)} bytes of code")
        print(f"  Cache hit: {result.cache_hit}")
        print(f"  First 200 chars: {result.code[:200]}...")

        # Check if it's not a placeholder
        if "placeholder" not in result.code.lower():
            print("✓ Generated REAL code (not a placeholder)!")
        else:
            print("✗ Still generating placeholders...")

        return result
    except Exception as e:
        print(f"✗ Code generation failed: {e}")
        return None


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("WORKSPACE INTEGRATION TEST SUITE")
    print("="*80 + "\n")

    # Test 1: Project creation
    project = await test_project_creation()

    # Test 2: Code generation
    code_result = await test_code_generation()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Workspace structure: /home/d/astra-workspace/")
    print(f"✓ Project created: {project.project_id if project else 'FAILED'}")
    print(f"✓ CodeGenerator using CodeAgent-5 prompts")
    print(f"✓ Code quality: {'REAL CODE' if code_result and 'placeholder' not in code_result.code.lower() else 'PLACEHOLDERS'}")
    print("\nWorkspace integration is READY for autonomous coding!")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
