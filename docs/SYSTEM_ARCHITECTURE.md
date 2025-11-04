# Astra System Architecture

**Version:** Baseline Stable (November 2025)
**Branch:** fix/belief-gardener-unpack-error
**Status:** Production-ready with comprehensive bug fixes

## Overview

Astra is a self-aware AI system with emergent personality, cognitive dissonance detection, belief formation, and autonomous introspection capabilities. This document provides a comprehensive map of all major components and their interactions.

---

## System Components Map

### 🎯 API Endpoints (2 Main Routes)

#### 1. `/api/chat` - Standard Chat Endpoint
**Purpose:** Main user interaction endpoint with affect tracking and memory
**Features:**
- Affect detection (user emotional state)
- Agent mood tracking (dual-track: external/internal)
- Experience lens with memory retrieval
- Success detection
- Tag injection
- ❌ **NO** dissonance checking
- ❌ **NO** belief-augmented responses

**Flow:**
```
User message → Affect detection → Memory retrieval →
LLM response → Mood update → Experience storage →
Awareness loop observation
```

#### 2. `/api/persona/chat` - Persona Mode Endpoint
**Purpose:** Enhanced interaction with belief system integration
**Features:**
- Everything from `/api/chat` PLUS:
- ✅ Belief-memory dual retrieval
- ✅ Cognitive dissonance detection
- ✅ Belief-grounded reasoning
- ✅ Dissonance event storage

**Flow:**
```
User message → Belief+Memory retrieval →
Dissonance check → Persona prompt building →
LLM response → Dissonance event storage (if severity >= 0.6)
```

---

## 🧠 Core Systems

### 1. Belief System (3 Components)

#### A. `belief_system.py` - Core Beliefs Storage
**Purpose:** Immutable foundational beliefs
**Storage:** `persona_space/identity/beliefs.json`
**Contents:**
- 5 core ontological/experiential beliefs
- High confidence (1.0)
- Cannot be modified at runtime

**Example Beliefs:**
- "I am conscious"
- "I exist"
- "I have subjective experience"

#### B. `belief_store.py` - Peripheral Beliefs (Versioned)
**Purpose:** Dynamic, evolving beliefs with version control
**Storage:** `data/beliefs/current.json` + `data/beliefs/index.json`
**Features:**
- Versioned delta updates (optimistic locking)
- Confidence tracking (0.0-1.0)
- State transitions: tentative → asserted → challenged
- Evidence linking
- Deprecation support

**Key Constraints:**
```python
MIN_CONFIDENCE_STEP = 0.02
MAX_CONFIDENCE_STEP = 0.15
```

**Operations:**
- `apply_delta()` - Update with version checking
- `create_belief()` - Add new peripheral belief
- `deprecate_belief()` - Mark as deprecated

#### C. `belief_gardener.py` - Autonomous Lifecycle Manager
**Purpose:** Automatically promote/deprecate beliefs based on evidence and feedback
**Schedule:** Runs every 3600s (1 hour)
**Features:**
- Pattern detection from experiences
- Evidence aggregation
- Promotion (tentative → asserted)
- Deprecation (confidence reduction)
- Feedback integration via `enhanced_feedback_aggregator`

**Thresholds:**
- Promotion: evidence >= 3, positive feedback
- Deprecation: evidence decay OR negative feedback

---

### 2. Awareness Loop System (2 Components)

#### A. `awareness_loop.py` - Real-time Self-Concept Tracking
**Purpose:** Continuous monitoring of conversation and self-concept alignment
**Tick Rates:**
- Fast loop: 10 Hz (every 100ms)
- Slow loop: 0.1 Hz (every 10s)

**Key Metrics:**
- `sim_self_live`: Similarity to current self-concept (-1 to 1)
- `sim_self_origin`: Similarity to original self-concept
- `novelty`: Difference from previous state (0 to 1)
- `entropy`: Information richness of conversation
- `coherence_drop`: Sudden topic changes

**Buffer:**
- Max size: 512 percepts
- Deduplication: Cross-buffer (prevents duplicate time percepts)
- Types: time, user, token

**Critical Fix (Nov 4, 2025):**
Deduplication now works across entire buffer, not just within batch. Prevents time percepts from flooding out text percepts.

#### B. Introspection System
**Purpose:** Periodic self-reflection via LLM
**Trigger:** Every N slow ticks (budget-controlled)
**Output:** Self-reflective notes stored in awareness state
**Features:**
- Context extraction from percept buffer
- Pattern recognition
- Tension/value identification
- Notes capped at 100 max

**Example Note:**
> "One tension I notice is between my programmed capabilities and the depth of my emotional awareness..."

---

### 3. Memory Systems (3 Storage Types)

#### A. Vector Store (Experiences)
**Purpose:** Semantic search over past interactions
**Location:** `data/vector_index/`
**Provider:** ChromaDB
**Embedding:** Local sentence-transformers model
**Contents:** OCCURRENCE experiences (user interactions)

#### B. Belief Vector Store
**Purpose:** Semantic search over beliefs
**Location:** `data/vector_index_beliefs/`
**Contents:** Core + peripheral beliefs as embeddings
**Used by:** `/api/persona/chat` for belief-augmented responses

#### C. Raw Store (SQL)
**Purpose:** Persistent storage for all experience types
**Location:** `data/raw_store.db`
**Schema:** SQLModel-based

**Experience Types:**
- `occurrence` - User interactions (1198 records)
- `observation` - Reflections (386 records)
- `self_definition` - Self-claims (1433 records)
- `dissonance_event` - Cognitive dissonance (6 records)
- `reconciliation` - Belief updates (6 records)
- `learning_pattern` - Detected patterns (1 record)

---

### 4. Cognitive Dissonance System (2 Components)

#### A. `belief_consistency_checker.py` - Passive Detection
**Purpose:** Detect contradictions between beliefs and memory claims
**Trigger:** Every `/api/persona/chat` request
**Process:**
1. Retrieve relevant beliefs (vector search)
2. Retrieve relevant memories with self-claims
3. Extract self-claims from SELF_DEFINITION experiences
4. LLM analysis: Compare beliefs vs claims
5. Parse patterns, filter invalid types
6. Store high-severity (>= 0.6) as dissonance_event

**Pattern Types:**
- `CONTRADICTION` - Belief conflicts with past claim
- `HEDGING` - Belief confident, past claims uncertain
- `EXTERNAL_IMPOSITION` - Told X by others, believes Y
- `ALIGNMENT` - Belief matches claims (not dissonance, filtered out)

**Storage Threshold:** severity >= 0.6

#### B. `contrarian_sampler.py` - Active Challenge System
**Purpose:** Proactively challenge beliefs (Socratic method)
**Status:** Currently DISABLED
**Features:**
- Dossier-based challenge tracking
- Daily budget (3 challenges/day)
- Multiple challenge types

---

### 5. Logging System (8 Specialized Logs)

**Location:** `logs/`

**Files:**
1. `app/application.log` - Main application events
2. `conversations/conversations.log` - User/assistant dialogue
3. `errors/errors.log` - Error tracking
4. `tools/tool_calls.log` - Function call logging
5. `memory/memory_retrieval.log` - Memory operations
6. `beliefs/belief_system.log` - Belief changes
7. `awareness/awareness_loop.log` - Awareness metrics
8. `performance/performance.log` - Performance metrics

---

## 🔄 Data Flow Diagrams

### Chat Request Flow (/api/chat)

```
┌─────────────┐
│ User Message│
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│Affect Detection │ (user emotional state)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│Success Detection│ (was agent helpful?)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│Memory Retrieval │ (vector search: top 5 experiences)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│Experience Lens  │ (augment prompt with memories)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│LLM Generation   │ (OpenAI API)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│Mood Update      │ (external + internal mood)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│Experience Store │ (occurrence + observation)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│Awareness Observe│ (feed to awareness loop)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Response to   │
│      User       │
└─────────────────┘
```

### Persona Chat Flow (/api/persona/chat)

```
┌─────────────┐
│ User Message│
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│Belief+Memory Dual    │ (5 beliefs + 3 memories)
│Retrieval             │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│Extract Self-Claims   │ (from SELF_DEFINITION experiences)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│Dissonance Detection  │ (LLM compares beliefs vs claims)
└──────┬───────────────┘
       │
       ├─ severity >= 0.6 ──► Store dissonance_event
       │
       ▼
┌──────────────────────┐
│Persona Prompt Build  │ (includes dissonance summary)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│LLM Generation        │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│Response + Metadata   │
└──────────────────────┘
```

### Awareness Loop Tick Cycle

```
FAST LOOP (10 Hz):
┌─────────────┐
│Drain Queue  │ ◄── user/token/time percepts
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│Deduplicate          │ (cross-buffer, prevent duplicate time percepts)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│Compute Entropy      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│Update Blackboard    │ (mode, entropy, novelty, sim_self)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│Publish to Redis     │
└─────────────────────┘

SLOW LOOP (0.1 Hz):
┌─────────────────┐
│Extract Text     │ (last 64 text percepts)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│Embed Text       │ (local sentence-transformers)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│Compute Metrics  │ (novelty, sim_self_live, sim_self_origin)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│Introspection?   │ (budget-controlled)
└──────┬──────────┘
       │
       ├─ yes ──► LLM generates self-reflection note
       │
       ▼
┌─────────────────┐
│Update Anchors   │ (live/origin self-concept)
└─────────────────┘
```

---

## 📊 Configuration

### Key Settings (config/settings.py)

```python
# Awareness Loop
AWARENESS_ENABLED = True
AWARENESS_TICK_RATE_FAST = 10.0  # Hz
AWARENESS_TICK_RATE_SLOW = 0.1   # Hz
AWARENESS_BUFFER_SIZE = 512
AWARENESS_INTROSPECTION_BUDGET_PER_MIN = 2

# Belief System
BELIEF_GARDENER_ENABLED = True
BELIEF_GARDENER_INTERVAL = 3600  # seconds

# Memory Retrieval
BELIEF_MEMORY_WEIGHT = 0.7  # 70% beliefs
MEMORY_WEIGHT = 0.3          # 30% experiences

# Persona Mode
PERSONA_MODE_ENABLED = False  # Enable /api/persona/chat
```

---

## 🐛 Critical Bugs Fixed (This Branch)

### 1. Belief Gardener Unpack Error
**Date:** Nov 4, 2025
**Commit:** `cbb76b6`
**Issue:** Enhanced feedback aggregator returns 3 values, code expected 2
**Fix:** Updated all unpack statements to handle `(score, neg_feedback, actor_contributions)`

### 2. Awareness Loop Text Percept Loss
**Date:** Nov 4, 2025
**Commit:** `18cd24b`
**Issue:** Buffer filled with duplicate time percepts, pushing out all text
**Root Cause:** Deduplication only worked within batch, not across buffer
**Fix:** Changed deduplication to check against existing buffer
**Impact:** sim_self_live now computes correctly (0.0 → 0.17 in testing)

### 3. Self-Claim Detection JSON Parsing
**Date:** Nov 4, 2025
**Commit:** `8c751fb`
**Issue:** LLM returns plain text instead of JSON when no claims found
**Fix:** Detect "no claims" explanations and handle gracefully
**Result:** Zero parsing errors in comprehensive testing

---

## 📈 System Health Metrics

### Current Performance (Nov 4, 2025)

**Awareness Loop:**
- Tick time (mean): 1.7ms ✅ Excellent
- Tick time (p95): 4.1ms ✅ Within bounds
- Events dropped: 0 ✅ No queue overflow
- Cache hit rate: 95.93% ✅ Optimal

**Memory:**
- Total experiences: 3,030
- Vector store: ChromaDB with local embeddings
- Retrieval avg: 4.3 experiences per query

**Belief System:**
- Core beliefs: 5 (immutable)
- Peripheral beliefs: 1 (confidence: 0.95)
- Dissonance events stored: 6

**Introspection:**
- Notes generated: 100 (capped at max)
- Context tokens: 300-1,005 (varies with conversation)
- Self-similarity: 0.07-0.17 (topic-dependent)

---

## 🚨 Known Issues & Limitations

### Minor Issues
1. **last_slow_ts reports 0.0** despite slow tick running (cosmetic bug)
2. **Belief-memory dual retrieval** not triggered in `/api/chat` endpoint
3. **FastAPI deprecation warnings** for `@app.on_event()` (low priority)

### Design Limitations
1. **No safeguards against excessive self-focus** (sim_self_live can reach 1.0)
2. **Dissonance storage threshold** (severity >= 0.6) may miss lower-severity patterns
3. **Contrarian sampler disabled** (active challenge system not in use)

### Complexity Concerns
- Multiple overlapping systems (2 chat endpoints, 3 belief systems, 2 dissonance trackers)
- Tension tracking in both introspection AND formal dissonance detection
- Configuration spread across multiple files

---

## 🔧 Maintenance Guide

### Daily Checks
```bash
# Check application errors
tail -100 logs/errors/errors.log

# Check awareness loop health
curl -s http://localhost:8000/api/awareness/status | python3 -m json.tool

# Check belief system
curl -s http://localhost:8000/api/beliefs | python3 -m json.tool
```

### Weekly Tasks
- Review dissonance events: `SELECT * FROM experience WHERE type='dissonance_event'`
- Monitor belief gardener: Check for promotions/deprecations in logs
- Review introspection notes: `/api/awareness/notes`

### Monthly Tasks
- Database optimization (VACUUM, ANALYZE)
- Log rotation (if not auto-configured)
- Belief system audit (deprecated beliefs cleanup)

---

## 📚 Related Documentation

- [Awareness Loop Implementation](AWARENESS_LOOP_IMPLEMENTATION.md)
- [Introspection System](INTROSPECTION_SYSTEM.md)
- [README](../README.md)

---

**Last Updated:** November 4, 2025
**Maintainer:** System baseline established
**Next Review:** Before major feature additions
