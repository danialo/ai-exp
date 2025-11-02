#!/usr/bin/env python3
"""
Interactive API Explorer for Astra
Menu-driven interface to query all GET endpoints
"""

import requests
import json
import sys
from typing import Optional, Dict, Any

# Base URL for the API
BASE_URL = "http://172.239.66.45:8000"

# All GET endpoints organized by category
ENDPOINTS = {
    "Persona (Astra)": [
        ("Persona Info", "/api/persona/info"),
        ("List Persona Files", "/api/persona/files"),
        ("Browse Persona Filesystem", "/api/persona/browse"),
    ],
    "Memory & Experiences": [
        ("List All Memories", "/api/memories"),
        ("Get Narratives", "/api/narratives"),
    ],
    "Beliefs": [
        ("All Beliefs", "/api/beliefs"),
        ("Core Beliefs", "/api/beliefs/core"),
        ("Peripheral Beliefs", "/api/beliefs/peripheral"),
    ],
    "Self-Knowledge": [
        ("Self-Knowledge Index", "/api/self"),
        ("Self Traits", "/api/self/traits"),
        ("Self History", "/api/self/history"),
    ],
    "Emotions & Mood": [
        ("Current Mood", "/api/mood"),
        ("Current Emotions", "/api/emotions/current"),
        ("Emotion History", "/api/emotions/history"),
        ("Emotion Patterns", "/api/emotions/patterns"),
    ],
    "Tasks": [
        ("All Tasks", "/api/tasks"),
        ("Due Tasks", "/api/tasks/due"),
        ("Recent Task Results", "/api/tasks/results/recent"),
    ],
    "Session": [
        ("Current Session", "/api/session/current"),
    ],
    "System": [
        ("System Stats", "/api/stats"),
        ("Available Models", "/api/models"),
        ("Health Check", "/health"),
        ("Debug Prompt Info", "/api/debug/prompt"),
    ],
}


def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end="")


def print_header():
    """Print the application header."""
    print("=" * 70)
    print(" " * 20 + "ASTRA API EXPLORER")
    print("=" * 70)
    print()


def print_menu():
    """Print the main menu."""
    print_header()

    endpoint_list = []
    category_idx = 1

    for category, endpoints in ENDPOINTS.items():
        print(f"\n{category}:")
        print("-" * len(category))

        for name, path in endpoints:
            endpoint_list.append((name, path))
            print(f"  {len(endpoint_list)}. {name}")

    print(f"\n  0. Exit")
    print("\n" + "=" * 70)

    return endpoint_list


def call_endpoint(path: str) -> Optional[Dict[str, Any]]:
    """Call a GET endpoint and return the response."""
    try:
        url = f"{BASE_URL}{path}"
        print(f"\nCalling: {url}")
        print("Please wait...\n")

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        return response.json()
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Cannot connect to {BASE_URL}")
        print("   Make sure the server is running!")
        return None
    except requests.exceptions.Timeout:
        print("❌ Error: Request timed out")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   Response: {response.text}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def display_response(data: Any):
    """Display the API response in a formatted way."""
    print("\n" + "=" * 70)
    print("RESPONSE:")
    print("=" * 70)

    if isinstance(data, (dict, list)):
        # Pretty print JSON
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(data)

    print("=" * 70)


def get_specific_memory_id() -> Optional[str]:
    """Prompt for a specific memory ID."""
    print("\nEnter memory/experience ID (or press Enter to cancel): ", end="")
    memory_id = input().strip()
    return memory_id if memory_id else None


def get_specific_task_id() -> Optional[str]:
    """Prompt for a specific task ID."""
    print("\nEnter task ID (or press Enter to cancel): ", end="")
    task_id = input().strip()
    return task_id if task_id else None


def get_file_path() -> Optional[str]:
    """Prompt for a file path."""
    print("\nEnter file path (or press Enter to cancel): ", end="")
    file_path = input().strip()
    return file_path if file_path else None


def handle_parameterized_endpoints():
    """Handle endpoints that require parameters."""
    print("\n" + "=" * 70)
    print("PARAMETERIZED ENDPOINTS")
    print("=" * 70)
    print("\n1. Get Specific Memory (/api/memory/{id})")
    print("2. Get Specific Task (/api/tasks/{id})")
    print("3. Get Task Results (/api/tasks/{id}/results)")
    print("4. Get Persona File (/api/persona/file/{path})")
    print("0. Back to main menu")
    print()

    choice = input("Enter choice: ").strip()

    if choice == "1":
        memory_id = get_specific_memory_id()
        if memory_id:
            return f"/api/memory/{memory_id}"
    elif choice == "2":
        task_id = get_specific_task_id()
        if task_id:
            return f"/api/tasks/{task_id}"
    elif choice == "3":
        task_id = get_specific_task_id()
        if task_id:
            return f"/api/tasks/{task_id}/results"
    elif choice == "4":
        file_path = get_file_path()
        if file_path:
            return f"/api/persona/file/{file_path}"

    return None


def main():
    """Main application loop."""
    while True:
        clear_screen()
        endpoint_list = print_menu()

        print("\nOr enter 'p' for parameterized endpoints")
        choice = input("\nEnter your choice: ").strip()

        if choice == "0":
            print("\nGoodbye! 👋")
            sys.exit(0)

        if choice.lower() == 'p':
            clear_screen()
            path = handle_parameterized_endpoints()
            if path:
                result = call_endpoint(path)
                if result is not None:
                    display_response(result)
        else:
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(endpoint_list):
                    name, path = endpoint_list[choice_num - 1]
                    clear_screen()
                    print(f"\n🔍 {name}")
                    result = call_endpoint(path)
                    if result is not None:
                        display_response(result)
                else:
                    print("\n❌ Invalid choice!")
            except ValueError:
                print("\n❌ Please enter a number!")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
        sys.exit(0)
