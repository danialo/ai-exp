#!/bin/bash
# Test script to demonstrate emergent personality via mood accumulation

echo "🧪 Testing Emergent Personality System"
echo "========================================"
echo ""

echo "📊 Initial mood:"
curl -s http://172.239.66.45:8000/api/mood | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"  Mood: {d['current_mood']:.3f} ({d['mood_description']})\")"
echo ""

echo "💬 Sending 5 negative messages to build up agent mood..."
echo ""

messages=(
    "This is so frustrating, nothing works!"
    "Why is this so damn confusing?"
    "I hate dealing with this bullshit"
    "This is terrible, I can't figure it out"
    "Fuck, this is annoying as hell"
)

for i in "${!messages[@]}"; do
    msg="${messages[$i]}"
    echo "[$((i+1))/5] User (angry): \"$msg\""

    response=$(curl -s -X POST http://172.239.66.45:8000/api/chat \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"$msg\", \"retrieve_memories\": true, \"top_k\": 3}")

    # Extract response
    agent_response=$(echo "$response" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['response'][:80] + '...')")
    echo "      Agent: $agent_response"

    # Check mood after each interaction
    mood_data=$(curl -s http://172.239.66.45:8000/api/mood)
    mood=$(echo "$mood_data" | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"{d['current_mood']:.3f}\")")
    mood_desc=$(echo "$mood_data" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['mood_description'])")
    is_pissed=$(echo "$mood_data" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['is_pissed'])")

    echo "      📊 Mood: $mood ($mood_desc) | Pissed: $is_pissed"
    echo ""

    sleep 1
done

echo "========================================"
echo "🎯 Final agent state:"
curl -s http://172.239.66.45:8000/api/mood | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"  Current mood: {d['current_mood']:.3f}\")
print(f\"  Description: {d['mood_description']}\")
print(f\"  Is pissed: {d['is_pissed']}\")
print(f\"  Interactions tracked: {d['recent_interactions']}\")
"
echo ""

echo "💡 Now try sending another negative message - agent might refuse!"
echo ""
echo "Test with:"
echo "  curl -X POST http://172.239.66.45:8000/api/chat -H 'Content-Type: application/json' -d '{\"message\": \"Help me with this shit\", \"retrieve_memories\": true}'"
