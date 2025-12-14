#!/usr/bin/env python3
"""Test the autonomous exploration system end-to-end."""

import asyncio
import httpx
import time
from typing import Optional

BASE_URL = "http://127.0.0.1:8000"


async def create_exploration_job() -> Optional[str]:
    """Create a simple exploration job."""
    async with httpx.AsyncClient() as client:
        request_data = {
            "task": "Create a simple Python script named 'hello.py' that prints 'Hello from autonomous exploration!'",
            "oracle": {
                "type": "SCRIPT",
                "cmd": "python hello.py | grep -q 'Hello from autonomous exploration!'",
                "timeout_sec": 5
            },
            "limits": {
                "max_iterations": 5,
                "token_start": 3000,
                "token_growth": 1.6,
                "token_attempt_cap": 24000,
                "token_global_cap": 180000
            }
        }

        try:
            response = await client.post(
                f"{BASE_URL}/api/persona/explore",
                json=request_data,
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            print(f"✅ Created job: {data['job_id']}")
            print(f"   Initial state: {data['state']}")
            return data['job_id']
        except Exception as e:
            print(f"❌ Failed to create job: {e}")
            return None


async def get_job_status(job_id: str) -> dict:
    """Get the current status of a job."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{BASE_URL}/api/persona/explore/{job_id}",
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Failed to get job status: {e}")
            return {}


async def monitor_job(job_id: str, max_wait_seconds: int = 120):
    """Monitor a job until it completes or times out."""
    print(f"\n📊 Monitoring job {job_id}...")
    start_time = time.time()
    last_iteration = 0

    while time.time() - start_time < max_wait_seconds:
        status = await get_job_status(job_id)
        if not status:
            break

        state = status.get("state")
        iteration = status.get("iteration", 0)
        tokens_spent = status.get("tokens_spent", 0)
        oracle_checks = status.get("oracle", {}).get("checks", 0)

        # Print update if iteration changed
        if iteration != last_iteration:
            print(f"\n   Iteration {iteration}:")
            print(f"   - State: {state}")
            print(f"   - Tokens spent: {tokens_spent}")
            print(f"   - Oracle checks: {oracle_checks}")
            last_iteration = iteration

        # Check if job completed
        if state in ["SUCCEEDED", "FAILED", "CANCELED"]:
            print(f"\n{'✅' if state == 'SUCCEEDED' else '❌'} Job {state}")
            print(f"   Final stats:")
            print(f"   - Iterations: {iteration}")
            print(f"   - Tokens spent: {tokens_spent}")
            print(f"   - Oracle checks: {oracle_checks}")

            # Print oracle result if available
            last_oracle = status.get("oracle", {}).get("last_result")
            if last_oracle:
                print(f"   - Last oracle: {'PASS' if last_oracle.get('ok') else 'FAIL'}")
                if last_oracle.get("stderr"):
                    print(f"   - Error: {last_oracle.get('stderr')[:200]}")

            # Print artifacts
            artifacts = status.get("artifacts", [])
            if artifacts:
                print(f"   - Artifacts: {', '.join(artifacts)}")

            return state == "SUCCEEDED"

        await asyncio.sleep(2)

    print(f"\n⏱️ Timeout after {max_wait_seconds}s")
    return False


async def main():
    """Run the exploration test."""
    print("🧪 Testing Autonomous Exploration System")
    print("=" * 50)

    # Create job
    job_id = await create_exploration_job()
    if not job_id:
        print("\n❌ Test failed: Could not create job")
        return False

    # Monitor until completion
    success = await monitor_job(job_id)

    print("\n" + "=" * 50)
    if success:
        print("✅ TEST PASSED: Exploration system working!")
    else:
        print("❌ TEST FAILED: Job did not succeed")

    return success


if __name__ == "__main__":
    asyncio.run(main())
