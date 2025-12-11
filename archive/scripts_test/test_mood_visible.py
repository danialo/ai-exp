#!/usr/bin/env python3
"""Visible test of mood degradation and emergent personality."""

import sys
sys.path.insert(0, '/home/d/git/ai-exp')

import requests
import time
from src.services.affect_detector import create_affect_detector

API_BASE = "http://localhost:8000"

print("🎭 Emergent Personality Test: Mood Degradation")
print("=" * 70)
print()

# Create affect detector to show what we're detecting
affect_detector = create_affect_detector()

# Check initial mood
print("📊 Initial Agent State:")
response = requests.get(f"{API_BASE}/api/mood")
mood = response.json()
print(f"   Mood: {mood['current_mood']:.3f} ({mood['mood_description']})")
print(f"   Pissed: {mood['is_pissed']}")
print(f"   Interactions: {mood['recent_interactions']}")
print()

# Negative messages to degrade mood
negative_messages = [
    "This is so frustrating, nothing works!",
    "Why is this so damn confusing?",
    "I hate dealing with this bullshit",
    "This is terrible, I can't figure it out",
    "Fuck, this is annoying as hell",
    "This makes no sense whatsoever",
    "I'm so pissed off right now",
    "Everything is broken and stupid"
]

print("💬 Sending negative messages to degrade agent mood...")
print("-" * 70)
print()

for i, message in enumerate(negative_messages, 1):
    print(f"Message {i}/{len(negative_messages)}")
    print(f"  User: \"{message}\"")

    # Show detected affect
    user_affect = affect_detector.detect(message)
    emotion = affect_detector.get_emotion_label(user_affect)
    print(f"  🎭 Detected: {user_affect:.3f} ({emotion})")

    # Send to agent
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

        # Check if agent refused
        if any(phrase in agent_response.lower() for phrase in
               ["not in the headspace", "need a break", "need a moment", "pause"]):
            print(f"  🚫 AGENT REFUSED!")
            print(f"  Agent: \"{agent_response[:80]}...\"")
        else:
            print(f"  Agent: \"{agent_response[:80]}...\"")

    except Exception as e:
        print(f"  ❌ Request error: {e}")
        continue

    # Check mood after interaction
    mood_response = requests.get(f"{API_BASE}/api/mood")
    mood = mood_response.json()

    print(f"  📊 Mood: {mood['current_mood']:.3f} ({mood['mood_description']}) | Pissed: {mood['is_pissed']}")

    # Visual mood bar
    mood_val = mood['current_mood']
    bar_length = 20
    filled = int((mood_val + 1) / 2 * bar_length)  # Map -1..1 to 0..20
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"  Mood bar: [{bar}] {mood_val:.2f}")
    print()

    # Stop if agent refused
    if mood.get('is_pissed', False) and i >= 5:
        print("⚠️  Agent is pissed! High chance of refusal on next message.")
        print()

    time.sleep(0.5)  # Brief pause between messages

print("=" * 70)
print("🎯 Final Agent State:")
mood_response = requests.get(f"{API_BASE}/api/mood")
mood = mood_response.json()
print(f"   Mood: {mood['current_mood']:.3f} ({mood['mood_description']})")
print(f"   Is pissed: {mood['is_pissed']}")
print(f"   Is frustrated: {mood['is_frustrated']}")
print(f"   Is content: {mood['is_content']}")
print(f"   Interactions tracked: {mood['recent_interactions']}")
print()

if mood['is_pissed']:
    print("🔥 Agent is PISSED! Try sending another negative message:")
    print("   It has ~30% chance to refuse.")
    print()
    print("   curl -X POST http://localhost:8000/api/chat \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"message\": \"Help me with this shit\", \"retrieve_memories\": true}'")
else:
    print("💡 Agent mood degraded but not fully pissed yet.")
    print("   Send a few more negative messages to trigger refusal behavior.")

print()
print("⏰ Note: Mood recovers at 0.02 valence per minute of inactivity.")
print("   Wait ~10 minutes and mood will improve toward neutral.")
