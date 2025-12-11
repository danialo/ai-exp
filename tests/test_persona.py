#!/usr/bin/env python3
"""Test script for the persona system."""

import requests
import json
import sys


def test_persona(base_url="http://localhost:8000"):
    """Test the persona chat endpoint."""

    print("🧪 Testing Persona System\n")
    print("=" * 60)

    # Check persona info
    print("\n1. Checking persona info...")
    try:
        response = requests.get(f"{base_url}/api/persona/info")
        if response.status_code == 200:
            info = response.json()
            print(f"✓ Persona space: {info.get('persona_space')}")
            print(f"✓ File tree: {json.dumps(info.get('file_tree'), indent=2)}")
        elif response.status_code == 503:
            print("✗ Persona mode not enabled. Set PERSONA_MODE_ENABLED=true in .env")
            sys.exit(1)
        else:
            print(f"✗ Error: {response.status_code} - {response.text}")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        print(f"   Make sure the server is running at {base_url}")
        sys.exit(1)

    # Test conversations
    print("\n2. Testing persona conversations...")
    print("-" * 60)

    test_messages = [
        "Hello! I'm curious about who you might become.",
        "What interests you most right now?",
        "How do you feel about learning and growth?",
        "Tell me about yourself.",
    ]

    for i, message in enumerate(test_messages, 1):
        print(f"\n[{i}] User: {message}")

        try:
            response = requests.post(
                f"{base_url}/api/persona/chat",
                json={"message": message}
            )

            if response.status_code == 200:
                data = response.json()

                # Show response
                print(f"\nPersona: {data['response']}\n")

                # Show emotional analysis
                if data.get('reconciliation'):
                    print("─── Emotional Analysis ───")
                    recon = data['reconciliation']

                    if recon.get('internal_assessment'):
                        print(f"Internal: {recon['internal_assessment'][:100]}...")

                    if recon.get('external_assessment'):
                        print(f"External: {recon['external_assessment'][:100]}...")

                    if recon.get('reconciled_state'):
                        print(f"Reconciled: {recon['reconciled_state'][:100]}...")

                    if recon.get('learning_signal'):
                        print(f"Learning: {recon['learning_signal'][:100]}...")

            else:
                print(f"✗ Error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"✗ Request failed: {e}")

        print("-" * 60)

    # Check persona files
    print("\n3. Checking persona's files...")
    try:
        response = requests.get(f"{base_url}/api/persona/files")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Files in root: {data.get('files')}")
        else:
            print(f"✗ Error: {response.status_code}")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Check emotional state
    print("\n4. Checking current emotional state...")
    try:
        response = requests.get(f"{base_url}/api/persona/file/emotional_state/current.json")
        if response.status_code == 200:
            data = response.json()
            content = json.loads(data['content'])
            print(f"✓ Current state: {json.dumps(content.get('current_state'), indent=2)}")
        else:
            print(f"✗ Error: {response.status_code}")
    except Exception as e:
        print(f"✗ Failed: {e}")

    print("\n" + "=" * 60)
    print("✓ Persona testing complete!")
    print("\nNext steps:")
    print("  - Check persona_space/ directory to see self-organized files")
    print("  - Try more conversations to see emotional reconciliation")
    print("  - Persona can modify its own base_prompt.md and file structure")


if __name__ == "__main__":
    test_persona()
