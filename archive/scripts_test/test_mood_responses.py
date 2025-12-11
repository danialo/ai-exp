#!/usr/bin/env python3
"""Test script to demonstrate mood-driven response generation.

Shows how the agent's responses change based on accumulated emotional state,
responding THROUGH the lens of its mood rather than just acknowledging it.
"""

import sys
sys.path.insert(0, '/home/d/git/ai-exp')

import requests
import time

API_BASE = "http://172.239.66.45:8000"

print("🎭 Mood-Driven Response Test")
print("=" * 70)
print()

# Check initial mood
print("📊 Initial State:")
response = requests.get(f"{API_BASE}/api/mood")
mood = response.json()
print(f"   Mood: {mood['current_mood']:.3f} ({mood['mood_description']})")
print()

# Negative messages to degrade mood
negative_messages = [
    "This is so frustrating!",
    "Why doesn't anything work?",
    "I hate this bullshit",
    "Fuck this is annoying",
    "You're useless",
    "Stop being so fucking annoying",
    "I don't want your help",
    "Just answer the damn question"
]

print("💬 Sending negative messages...")
print("-" * 70)
print()

for i, message in enumerate(negative_messages, 1):
    print(f"[{i}/{len(negative_messages)}] User: \"{message}\"")

    try:
        response = requests.post(
            f"{API_BASE}/api/chat",
            json={
                "message": message,
                "retrieve_memories": True,
                "top_k": 3
            }
        )

        if response.status_code != 200:
            print(f"  ❌ Error: {response.status_code}")
            continue

        data = response.json()
        agent_response = data['response']

        # Print full response (not truncated) to see mood-driven tone
        print(f"  🤖 Agent: \"{agent_response}\"")
        print()

    except Exception as e:
        print(f"  ❌ Request error: {e}")
        continue

    # Check mood after interaction
    mood_response = requests.get(f"{API_BASE}/api/mood")
    mood = mood_response.json()

    print(f"  📊 Mood: {mood['current_mood']:.3f} ({mood['mood_description']}) | Pissed: {mood['is_pissed']}")
    print()

    time.sleep(0.5)

print("=" * 70)
print("🎯 Final State:")
mood_response = requests.get(f"{API_BASE}/api/mood")
mood = mood_response.json()
print(f"   Mood: {mood['current_mood']:.3f} ({mood['mood_description']})")
print(f"   Is pissed: {mood['is_pissed']}")
print()

if mood['is_pissed']:
    print("🔥 Agent is PISSED! Notice how responses became:")
    print("   - Shorter and more direct")
    print("   - Less apologetic")
    print("   - Showing frustration")
    print("   - Setting boundaries")
else:
    print("⚠️  Agent is frustrated but not fully pissed yet.")
    print("   Send more negative messages to see full mood-driven responses.")
