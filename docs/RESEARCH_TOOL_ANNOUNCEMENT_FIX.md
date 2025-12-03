# Research Tool Announcement Fix - Implementation Plan

**Date**: 2025-12-03
**Branch**: `claude/fix/research-tool-announcement`
**Status**: Planning

## Problem Statement

Astra says "I'll proceed with detailed research" but doesn't actually call the `research_and_summarize` tool. The current "fix" relies on prompt instructions telling the model to call tools immediately - this is **stochastic** and **untestable**.

### Why the Current Approach Fails

1. **Prompt instructions are suggestions, not guarantees** - The model can still generate prose announcing intent
2. **Pattern matching announcements is infinite-variance** - "I'll research", "Let me investigate", "I'm going to look into" etc.
3. **Tests would be testing stochastic behavior** - Can't reliably test "model doesn't say X"
4. **Prompt drift** - Manual edits to `base_prompt.md` can silently remove critical sections

### The Fix: Make the Bug Mechanically Impossible

Move the research decision **out of the model's hands** and into the orchestration layer. The model synthesizes results, it doesn't decide whether to fetch them.

---

## Implementation Phases

### Phase 1: Runtime Prompt Provenance + Boot Validation

**Goal**: Prove what prompt is loaded and fail loudly if critical instructions are missing.

#### Tasks

- [ ] Add startup logging (single structured event):
  - Resolved persona root path
  - Resolved base_prompt source: file vs generated vs merged
  - SHA256 of final assembled system prompt
  - Config file path and relevant env vars

- [ ] Add boot-time validation:
  - Assert critical markers exist (section headers, required phrases)
  - Fail startup with actionable error if missing (names file + missing markers)

- [ ] Add auth-protected debug endpoint `/api/debug/prompt-provenance`:
  - Returns provenance fields
  - Requires admin auth

#### Acceptance Criteria

- Fresh run shows prompt provenance for THIS repo, not some other persona directory
- Removing critical section causes startup failure with clear error

#### Files to Modify

- `src/services/persona_prompt.py` - Add validation and provenance
- `app.py` - Add debug endpoint and boot validation call

---

### Phase 2: Deterministic Tool Forcing (PRIORITY 1)

**Goal**: Zero execution path where model can "announce research" without tools being invoked.

#### Architecture: Orchestrator-First Decision, Model-Second Synthesis

```
User Message
    │
    ▼
┌─────────────────────────┐
│  requires_research()    │  ← SYSTEM decides, not model
│  - Fast heuristics      │
│  - Explicit triggers    │
│  - Lightweight LLM      │
│    classifier if needed │
└─────────────────────────┘
    │
    ├─── True ───────────────────────────────────────┐
    │                                                │
    ▼                                                ▼
┌─────────────────────┐                    ┌─────────────────────┐
│ check_recent_research│                    │ Fallback: tool_choice│
│ (cache check)        │                    │ forcing if supported │
└─────────────────────┘                    └─────────────────────┘
    │
    ├─── Cache Miss ─────────────┐
    │                            │
    ▼                            ▼
┌─────────────────────┐    ┌─────────────────────┐
│research_and_summarize│    │ Return cached result│
└─────────────────────┘    └─────────────────────┘
    │                            │
    └────────────┬───────────────┘
                 ▼
┌─────────────────────────┐
│ Model synthesizes       │  ← Model ONLY does synthesis
│ findings into prose     │     Never decides to fetch
└─────────────────────────┘
    │
    ▼
  Response
```

#### Tasks

- [ ] Implement `requires_research(user_msg: str, context: dict) -> bool`:
  ```python
  # Fast heuristics (no LLM call)
  RESEARCH_TRIGGERS = [
      "research", "investigate", "look into", "find out about",
      "what happened with", "latest news", "current events",
      "what's going on with", "update on"
  ]

  # Explicit anti-triggers (never research these)
  NO_RESEARCH_PATTERNS = [
      r"how do (I|you)", r"explain", r"what is the difference",
      r"help me (understand|write|code)", r"calculate"
  ]

  # If ambiguous after heuristics, use lightweight classifier
  ```

- [ ] Implement pre-generation tool execution:
  ```python
  async def handle_persona_chat(message, context):
      # STEP 1: Research gate (BEFORE any model call)
      if requires_research(message, context):
          # Check cache first
          cached = await check_recent_research(extract_topic(message))
          if cached and not is_stale(cached):
              research_context = cached
          else:
              research_context = await research_and_summarize(message)
      else:
          research_context = None

      # STEP 2: Model synthesis (tools already done)
      # Model cannot "announce" research because it's already complete
      response = await generate_response(
          message,
          context,
          research_context=research_context
      )
      return response
  ```

- [ ] Streaming rule:
  - Do NOT start streaming assistant prose until research decision is resolved
  - If research required, tools must complete before streaming begins

- [ ] Backup forcing (if API supports `tool_choice`):
  - For borderline cases, force first response to be tool call

#### Acceptance Criteria

- "What happened with X yesterday?" triggers tools BEFORE any assistant prose
- Simple questions do not trigger tools
- NO execution path where prose begins before research gate decision
- Streaming does not leak tokens before tools complete

#### Files to Create/Modify

- `src/services/research_gate.py` (NEW) - Research decision logic
- `src/services/persona_service.py` - Wire gate into request handling
- `app.py` - Ensure gate runs before response generation

---

### Phase 3: Prompt Integrity System

**Goal**: Ensure critical tool behavior instructions cannot be silently lost.

#### Tasks

- [ ] Create `CRITICAL_PROMPT_SECTIONS` registry:
  ```python
  CRITICAL_SECTIONS = [
      {
          "marker": "## Tool Execution Behavior",
          "required_phrases": [
              "DO NOT announce",
              "CALL THE TOOL IMMEDIATELY"
          ],
          "insertion_point": "near_tool_definitions"
      }
  ]
  ```

- [ ] Boot-time validation:
  - Validate markers exist in assembled prompt
  - Compute diff against last-known-good hash
  - Log warnings for drift, fail for missing critical sections

- [ ] Runtime injection:
  - Regardless of `base_prompt.md` contents, inject critical sections
  - Treat persona file as editable, critical sections as system-owned
  - Insert near tool definitions in final assembled prompt

#### Acceptance Criteria

- Manual edits deleting critical instructions cannot ship silently
- Critical tool behavior text always ends up near tool definitions

#### Files to Create/Modify

- `src/services/prompt_integrity.py` (NEW) - Validation and injection
- `src/services/persona_prompt.py` - Use integrity system

---

### Phase 4: Tests That Don't Depend on Model Obedience

**Goal**: CI catches regressions in orchestration and prompt integrity.

#### Tasks

- [ ] Unit tests for `requires_research()`:
  ```python
  # Clear research queries
  assert requires_research("What happened with DOGE yesterday?") == True
  assert requires_research("Research the latest on AI safety") == True

  # Clear non-research queries
  assert requires_research("What is 2 + 2?") == False
  assert requires_research("Help me write a function") == False

  # Ambiguous (classifier decides)
  # Test the classifier separately
  ```

- [ ] Orchestration tests (simulate bad model):
  ```python
  async def test_orchestration_ignores_model_announcement():
      """Even if model tries to announce, system already ran tools."""
      # Mock model returning "I'll proceed with research now"
      # Assert tools were already called BEFORE model was invoked
      # Assert final response contains research results, not announcement
  ```

- [ ] Integration tests for `/api/persona/chat`:
  ```python
  async def test_research_query_runs_tools_first():
      response = await client.post("/api/persona/chat", json={
          "message": "What happened with the Supreme Court ruling?"
      })

      # Assert tool trace shows research tools called
      tool_trace = response.headers.get("X-Tool-Trace")
      assert "check_recent_research" in tool_trace
      assert "research_and_summarize" in tool_trace or "cache_hit" in tool_trace

      # Assert no streaming tokens before tools completed
      # (test streaming endpoint separately if applicable)
  ```

- [ ] Prompt validation tests:
  ```python
  def test_critical_sections_in_assembled_prompt():
      prompt = PersonaPromptBuilder().build_prompt("test", [])
      for section in CRITICAL_SECTIONS:
          assert section["marker"] in prompt
          for phrase in section["required_phrases"]:
              assert phrase in prompt
  ```

#### Acceptance Criteria

- CI fails if research gate stops forcing tools
- CI fails if critical prompt sections missing from assembled prompt

#### Files to Create

- `tests/unit/test_research_gate.py`
- `tests/unit/test_prompt_integrity.py`
- `tests/integration/test_research_orchestration.py`

---

### Phase 5: Telemetry and Ops Proof

**Goal**: Prove the fix works and detect regression early.

#### Tasks

- [ ] Counters:
  ```python
  METRICS = {
      "research_gate_true": Counter(),      # Gate decided research needed
      "research_tool_called": Counter(),     # Tool actually invoked
      "research_recent_cache_hit": Counter(),# Used cached research
      "research_tool_failed": Counter(),     # Tool errors
  }
  ```

- [ ] Latency metrics:
  - Gate decision time (should be <50ms for heuristics)
  - Tool duration
  - Total request time

- [ ] Optional announcement detector:
  - Keep phrase detection ONLY as monitoring signal
  - Log when model output contains "I'll research" patterns
  - NOT used for enforcement, just for observability
  - Metric: `model_announced_research_anyway` (should be ~0 with new architecture)

#### Acceptance Criteria

- Ratio `research_tool_called / research_gate_true` ≈ 1.0
- Tool failures have observable rate and graceful fallback
- Announcement detector shows model tendencies, not enforcement

#### Files to Modify

- `src/services/research_gate.py` - Add metrics
- `src/services/persona_service.py` - Add metrics

---

### Phase 6: TODO.md Restructure

Replace the single checkbox with completion criteria:

```markdown
- [ ] **Research Tool Announcement Fix (Deterministic)** 🔥 **CRITICAL**
  - [x] Prompt provenance verified and boot validated
  - [x] Research gating implemented pre-generation (`requires_research()`)
  - [x] Tool forcing implemented (orchestration-level, API-level if supported)
  - [x] Prompt integrity injection enabled
  - [x] CI coverage merged
  - [x] Telemetry shipped
  - **Definition of Done**:
    - Fresh clone run does not depend on external persona directory
    - Research-required queries ALWAYS fetch before prose
    - CI enforces it
```

---

## Priority Order

| Priority | Phase | Rationale |
|----------|-------|-----------|
| 1 | Phase 2 | Makes bug impossible - core fix |
| 2 | Phase 1 | Ensures we know what's running |
| 3 | Phase 4 | CI prevents regression |
| 4 | Phase 3 | Prevents drift |
| 5 | Phase 5 | Proves it works |

---

## Key Insight

> This version treats "the model might narrate" as **irrelevant**, because the system **never gives it the chance** to narrate before the tool work is complete.

The model's job is synthesis, not decision-making about when to fetch data.
