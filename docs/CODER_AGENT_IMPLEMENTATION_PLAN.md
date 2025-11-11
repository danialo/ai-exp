# CoderAgent Implementation Plan

**Created**: 2025-11-11
**Status**: Planning Phase
**Goal**: Extract code generation into specialized agent with JSON-based interface

## Overview

Migrate from monolithic PersonaService to multi-agent architecture, starting with CoderAgent as the first specialist.

### Why CoderAgent First?
1. **Highest cost** - Code generation is most expensive operation (16k token responses)
2. **Clear boundaries** - Code generation is distinct from chat/personality
3. **Immediate value** - Can optimize prompts without belief system overhead
4. **Validates approach** - Proves multi-agent pattern before extracting more

## Architecture

### Current (Monolithic)
```
User Request
    ↓
PersonaService
    ├─ Loads beliefs (20k chars)
    ├─ Loads memories (5-10k chars)
    ├─ Builds persona prompt (30k chars)
    ├─ Includes 14 tools
    ├─ execute_goal tool → GoalExecutionService → CodeGenerator
    └─ Returns response

Cost: GPT-4o with 30k prompt + 16k response = ~46k tokens = $0.69/request
```

### Target (Multi-Agent)
```
User Request
    ↓
AgentRouter (lightweight)
    ├─ Chat request → AstraAgent (personality, beliefs, memories)
    │                 Cost: ~46k tokens = $0.69/request
    │
    └─ Code request → CoderAgent (no beliefs, no personality)
                      Cost: ~8k tokens = $0.12/request
                      Savings: 83% per code request
```

## CoderAgent Specification

### Input Schema
```json
{
  "goal_text": "implement_feature | fix_bug | refactor_code | add_tests",
  "context": {
    "existing_files": ["path/to/relevant/file.py"],
    "requirements": "User's natural language description",
    "constraints": ["no network", "pure stdlib", "max 400 lines"]
  },
  "timeout_ms": 120000
}
```

### Output Schema (Per Your Prompt)
```json
{
  "plan": ["step 1", "step 2"],
  "artifacts": [
    {
      "filename": "path/to/file.py",
      "language": "python",
      "code": "<full file content>"
    }
  ],
  "checks": {
    "ruff_black_clean": true,
    "mypy_clean": true,
    "forbidden_apis_used": [],
    "size_ok": true,
    "idempotent_key": "<sha256>"
  },
  "assumptions": ["assumption A", "assumption B"]
}
```

### Safety Guarantees
- ✅ Forbidden APIs blocked (eval, exec, subprocess, etc.)
- ✅ Size limits enforced (400 lines impl, 250 lines tests)
- ✅ Deterministic output (idempotent_key for caching)
- ✅ JSON-only output (machine-consumable, no prose)
- ✅ Static checks validated (Ruff, Black, mypy)

## File Structure

```
src/agents/
├── __init__.py
├── base.py                    # BaseAgent interface
├── coder_agent.py             # CoderAgent implementation
├── router.py                  # AgentRouter (dispatches requests)
└── prompts/
    └── coder_v1_0.txt         # Your system prompt

src/services/
├── persona_service.py         # Keep for AstraAgent (chat)
└── goal_execution_service.py  # Modified to use CoderAgent

tests/agents/
├── test_coder_agent.py
├── test_router.py
└── fixtures/
    └── sample_requests.json
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1 - Days 1-2)

**Create base agent interface:**
```python
# src/agents/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """Base interface for all specialized agents."""

    @abstractmethod
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process request and return structured output."""
        pass

    @abstractmethod
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate output matches expected schema."""
        pass
```

**Create CoderAgent:**
```python
# src/agents/coder_agent.py
import json
import hashlib
from typing import Dict, Any, List
from pathlib import Path

class CoderAgent(BaseAgent):
    """Specialized agent for code generation with strict safety."""

    FORBIDDEN_APIS = [
        "eval", "exec", "compile", "__import__",
        "subprocess", "os.system", "os.popen",
        "socket", "requests", "httpx", "urllib",
        # ... full list from prompt
    ]

    MAX_IMPL_SIZE = 400  # lines
    MAX_TEST_SIZE = 250  # lines
    MAX_IMPL_BYTES = 40 * 1024  # 40 KB
    MAX_TEST_BYTES = 24 * 1024  # 24 KB

    def __init__(self, llm_service):
        """Initialize with LLM service (no beliefs, no memories)."""
        self.llm = llm_service
        self.system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        """Load coder system prompt from file."""
        prompt_path = Path(__file__).parent / "prompts" / "coder_v1_0.txt"
        return prompt_path.read_text()

    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate code artifacts from request.

        Args:
            request: {
                "goal_text": str,
                "context": dict,
                "timeout_ms": int
            }

        Returns:
            {
                "plan": list[str],
                "artifacts": list[dict],
                "checks": dict,
                "assumptions": list[str]
            }
        """
        # Build minimal prompt (NO beliefs, NO personality)
        user_prompt = self._build_user_prompt(request)

        # Call LLM with coder system prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = await self.llm.generate_with_tools(
            messages=messages,
            tools=None,  # No tools for coder
            temperature=0.2,  # Low temp for deterministic code
            max_tokens=8000,  # Smaller than Astra's 16k
        )

        # Parse JSON output
        output = self._parse_json_output(response["message"].content)

        # Validate safety
        self._validate_safety(output)

        # Validate schema
        if not self.validate_output(output):
            raise ValueError("CoderAgent output validation failed")

        return output

    def _build_user_prompt(self, request: Dict[str, Any]) -> str:
        """Build minimal user prompt from request."""
        lines = []
        lines.append(f"GOAL: {request['goal_text']}")

        if "context" in request:
            ctx = request["context"]
            if "requirements" in ctx:
                lines.append(f"\nREQUIREMENTS:\n{ctx['requirements']}")
            if "existing_files" in ctx:
                lines.append(f"\nEXISTING FILES TO CONSIDER:")
                for f in ctx["existing_files"]:
                    lines.append(f"  - {f}")
            if "constraints" in ctx:
                lines.append(f"\nCONSTRAINTS:")
                for c in ctx["constraints"]:
                    lines.append(f"  - {c}")

        return "\n".join(lines)

    def _parse_json_output(self, raw: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response."""
        # Remove markdown fences if present
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]

        return json.loads(raw.strip())

    def _validate_safety(self, output: Dict[str, Any]):
        """Validate no forbidden APIs used."""
        for artifact in output.get("artifacts", []):
            code = artifact.get("code", "")

            # Check forbidden APIs
            for forbidden in self.FORBIDDEN_APIS:
                if forbidden in code:
                    raise ValueError(f"Forbidden API '{forbidden}' found in artifact {artifact['filename']}")

            # Check size limits
            lines = len(code.splitlines())
            bytes_size = len(code.encode('utf-8'))

            is_test = "test" in artifact["filename"].lower()
            max_lines = self.MAX_TEST_SIZE if is_test else self.MAX_IMPL_SIZE
            max_bytes = self.MAX_TEST_BYTES if is_test else self.MAX_IMPL_BYTES

            if lines > max_lines:
                raise ValueError(f"Artifact {artifact['filename']} exceeds line limit: {lines} > {max_lines}")
            if bytes_size > max_bytes:
                raise ValueError(f"Artifact {artifact['filename']} exceeds byte limit: {bytes_size} > {max_bytes}")

    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate output schema."""
        required_keys = ["plan", "artifacts", "checks", "assumptions"]
        if not all(k in output for k in required_keys):
            return False

        # Validate artifacts structure
        for artifact in output.get("artifacts", []):
            if not all(k in artifact for k in ["filename", "language", "code"]):
                return False

        # Validate checks structure
        checks = output.get("checks", {})
        required_checks = ["ruff_black_clean", "mypy_clean", "forbidden_apis_used", "size_ok", "idempotent_key"]
        if not all(k in checks for k in required_checks):
            return False

        return True
```

**Files to create:**
- [ ] `src/agents/__init__.py`
- [ ] `src/agents/base.py`
- [ ] `src/agents/coder_agent.py`
- [ ] `src/agents/prompts/coder_v1_0.txt` (your provided prompt)
- [ ] `tests/agents/test_coder_agent.py`

### Phase 2: Router Implementation (Week 1 - Day 3)

**Create AgentRouter:**
```python
# src/agents/router.py
from enum import Enum
from typing import Dict, Any, Optional

class AgentType(Enum):
    ASTRA_CHAT = "astra_chat"     # Personality, beliefs, general chat
    CODER = "coder"                # Code generation
    RESEARCHER = "researcher"      # Future: web search
    PLANNER = "planner"           # Future: HTN planning

class AgentRouter:
    """Routes requests to appropriate specialized agent."""

    # Keywords that indicate code request
    CODE_KEYWORDS = [
        "implement", "code", "function", "class", "refactor",
        "execute_goal", "write code", "create file", "add tests",
        "fix bug", "debug", "optimize code"
    ]

    def __init__(self, astra_agent, coder_agent):
        self.astra = astra_agent
        self.coder = coder_agent

    def route(self, user_message: str, tools_requested: Optional[List[str]] = None) -> AgentType:
        """
        Decide which agent should handle this request.

        Args:
            user_message: User's message
            tools_requested: Optional explicit tool list from API

        Returns:
            AgentType enum indicating which agent to use
        """
        # Explicit tool check
        if tools_requested and "execute_goal" in tools_requested:
            return AgentType.CODER

        # Keyword detection
        message_lower = user_message.lower()
        if any(keyword in message_lower for keyword in self.CODE_KEYWORDS):
            return AgentType.CODER

        # Default to Astra for personality/chat
        return AgentType.ASTRA_CHAT

    async def process(self, user_message: str, **kwargs) -> Dict[str, Any]:
        """Route and process request through appropriate agent."""
        agent_type = self.route(user_message, kwargs.get("tools_requested"))

        if agent_type == AgentType.CODER:
            # Extract code request params
            request = self._build_coder_request(user_message, kwargs)
            return await self.coder.process(request)

        elif agent_type == AgentType.ASTRA_CHAT:
            # Use existing PersonaService
            return self.astra.generate_response(user_message, **kwargs)

        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def _build_coder_request(self, user_message: str, kwargs: Dict) -> Dict[str, Any]:
        """Convert user message to CoderAgent request format."""
        return {
            "goal_text": "implement_feature",  # Could parse this from message
            "context": {
                "requirements": user_message,
                "constraints": ["no network", "pure stdlib"]
            },
            "timeout_ms": kwargs.get("timeout_ms", 120000)
        }
```

**Files to create:**
- [ ] `src/agents/router.py`
- [ ] `tests/agents/test_router.py`

### Phase 3: Integration (Week 1 - Days 4-5)

**Modify app.py to use router:**
```python
# app.py (modifications)

# Initialize agents
from src.agents.coder_agent import CoderAgent
from src.agents.router import AgentRouter

# Create separate LLM for CoderAgent (no beliefs, smaller context)
coder_llm = create_llm_service(
    api_key=api_key,
    model=settings.LLM_MODEL,
    temperature=0.2,  # Lower temp for code
    max_tokens=8000,  # Smaller than Astra's 16k
    base_url=base_url,
    self_aware_prompt_builder=None,  # No self-awareness for coder
)

coder_agent = CoderAgent(llm_service=coder_llm)

# Create router
agent_router = AgentRouter(
    astra_agent=persona_service,  # Existing
    coder_agent=coder_agent        # New
)

@app.post("/api/persona/chat")
async def persona_chat(request: ChatRequest):
    """Chat endpoint now routes to appropriate agent."""
    try:
        # Router decides: Astra or Coder
        result = await agent_router.process(
            user_message=request.message,
            retrieve_memories=request.retrieve_memories,
            top_k=request.top_k,
            conversation_history=request.conversation_history
        )

        # Handle different response types
        if isinstance(result, dict) and "artifacts" in result:
            # CoderAgent response - format for user
            response_text = format_coder_response(result)
        else:
            # Astra response - existing format
            response_text, reconciliation = result

        # ... rest of endpoint
```

**Modify GoalExecutionService to use CoderAgent:**
```python
# src/services/goal_execution_service.py

class GoalExecutionService:
    def __init__(self, planner, task_graph, executor, code_generator, coder_agent=None):
        # ... existing init ...
        self.coder_agent = coder_agent  # New

    async def _materialize_task_content(self, action_name, params, action_num, goal_id):
        """Generate code - now uses CoderAgent if available."""
        if self.coder_agent:
            # Use new CoderAgent
            request = {
                "goal_text": "implement_feature",
                "context": {
                    "requirements": self.current_goal_text or "implement feature",
                    "existing_files": params.get("file_path", []),
                    "constraints": ["no network", "pure stdlib", "max 400 lines"]
                },
                "timeout_ms": 120000
            }

            result = await self.coder_agent.process(request)

            # Extract code from first artifact
            if result.get("artifacts"):
                return result["artifacts"][0]["code"]

            raise ValueError("CoderAgent returned no artifacts")

        else:
            # Fallback to old CodeGenerator
            return await self._old_materialize_task_content(action_name, params, action_num, goal_id)
```

**Files to modify:**
- [ ] `app.py` - Add router initialization
- [ ] `src/services/goal_execution_service.py` - Use CoderAgent
- [ ] `src/services/persona_service.py` - Add marker comments for future extraction

### Phase 4: Testing & Validation (Week 1 - Day 5, Week 2 - Day 1)

**Test Cases:**
```python
# tests/agents/test_coder_agent.py

import pytest
from src.agents.coder_agent import CoderAgent

@pytest.fixture
def coder_agent(llm_service_mock):
    return CoderAgent(llm_service=llm_service_mock)

def test_simple_function_generation(coder_agent):
    """Test generating a simple function."""
    request = {
        "goal_text": "implement_feature",
        "context": {
            "requirements": "Create parse_kv(s: str) that parses 'k=v' lines",
            "constraints": ["pure stdlib", "add tests"]
        }
    }

    result = await coder_agent.process(request)

    # Validate schema
    assert "plan" in result
    assert "artifacts" in result
    assert "checks" in result
    assert "assumptions" in result

    # Validate artifacts
    assert len(result["artifacts"]) >= 2  # impl + tests
    assert any("test_" in a["filename"] for a in result["artifacts"])

    # Validate checks
    assert result["checks"]["ruff_black_clean"] == True
    assert result["checks"]["forbidden_apis_used"] == []
    assert result["checks"]["size_ok"] == True

def test_forbidden_api_detection(coder_agent):
    """Test that forbidden APIs are caught."""
    # Mock LLM to return code with eval()
    coder_agent.llm.generate_with_tools = mock_response_with_eval()

    request = {"goal_text": "implement_feature", "context": {}}

    with pytest.raises(ValueError, match="Forbidden API 'eval'"):
        await coder_agent.process(request)

def test_size_limit_enforcement(coder_agent):
    """Test that oversized code is rejected."""
    # Mock LLM to return 500-line file
    coder_agent.llm.generate_with_tools = mock_response_with_large_file()

    request = {"goal_text": "implement_feature", "context": {}}

    with pytest.raises(ValueError, match="exceeds line limit"):
        await coder_agent.process(request)

def test_json_output_validation(coder_agent):
    """Test that invalid JSON schema is rejected."""
    # Mock LLM to return invalid schema
    coder_agent.llm.generate_with_tools = mock_response_missing_checks()

    request = {"goal_text": "implement_feature", "context": {}}

    with pytest.raises(ValueError, match="validation failed"):
        await coder_agent.process(request)
```

**Integration Test:**
```python
# tests/integration/test_coder_agent_e2e.py

def test_full_code_generation_pipeline():
    """Test complete flow: request → CoderAgent → files created → tests pass."""
    # Real LLM call
    coder = CoderAgent(llm_service=real_llm)

    request = {
        "goal_text": "implement_feature",
        "context": {
            "requirements": "Create a simple calculator with add/subtract/multiply/divide",
            "constraints": ["pure stdlib", "type hints", "docstrings", "tests"]
        }
    }

    result = await coder.process(request)

    # Verify artifacts
    assert len(result["artifacts"]) >= 2

    # Write artifacts to disk
    for artifact in result["artifacts"]:
        Path(artifact["filename"]).write_text(artifact["code"])

    # Run tests
    import subprocess
    test_result = subprocess.run(["pytest", "-q", "tests/generated"], capture_output=True)
    assert test_result.returncode == 0, f"Tests failed: {test_result.stderr}"

    # Run static checks
    ruff_result = subprocess.run(["ruff", "check", "src/generated"], capture_output=True)
    assert ruff_result.returncode == 0

    mypy_result = subprocess.run(["mypy", "src/generated"], capture_output=True)
    assert mypy_result.returncode == 0
```

**Files to create:**
- [ ] `tests/agents/test_coder_agent.py`
- [ ] `tests/agents/test_router.py`
- [ ] `tests/integration/test_coder_agent_e2e.py`
- [ ] `tests/agents/fixtures/sample_requests.json`

### Phase 5: Documentation (Week 2 - Day 2)

**Documentation to create:**
- [ ] `docs/CODER_AGENT.md` - Usage guide
- [ ] `docs/MULTI_AGENT_ARCHITECTURE.md` - System overview
- [ ] `docs/MIGRATION_GUIDE.md` - Monolithic → Multi-agent
- [ ] Update `docs/EXECUTE_GOAL_QUICK_REF.md` with CoderAgent info

## Success Metrics

### Performance
- [ ] **Cost reduction**: 83% cheaper for code requests ($0.69 → $0.12)
- [ ] **Speed improvement**: 2x faster (smaller prompts, lower temp)
- [ ] **Context efficiency**: 8k tokens vs 30k tokens per request

### Quality
- [ ] **Code quality**: 100% of outputs pass Ruff/Black/mypy
- [ ] **Safety**: 0 forbidden APIs in generated code
- [ ] **Tests**: 100% of generated code includes passing tests
- [ ] **Determinism**: Idempotent outputs (same request → same SHA256)

### Architecture
- [ ] **Separation**: CoderAgent has zero dependencies on beliefs/personality
- [ ] **Reusability**: CoderAgent can be called from any service
- [ ] **Extensibility**: Easy to add ResearchAgent, PlannerAgent next

## Migration Strategy

### Week 1: Parallel Operation
- [ ] CoderAgent implemented but optional
- [ ] Router defaults to Astra unless `use_coder_agent=true` flag
- [ ] Both paths work, gather metrics

### Week 2: Gradual Rollout
- [ ] Router uses keyword detection (50% of code requests → CoderAgent)
- [ ] Monitor error rates, code quality
- [ ] Adjust routing logic based on results

### Week 3: Full Migration
- [ ] Router default for all code requests
- [ ] Astra only for chat/personality
- [ ] Old CodeGenerator kept as fallback

### Week 4: Cleanup
- [ ] Remove old CodeGenerator if CoderAgent stable
- [ ] Extract ResearchAgent (next specialist)
- [ ] Document multi-agent patterns

## Risk Mitigation

### Risk: CoderAgent produces lower quality code
**Mitigation:** Keep old CodeGenerator as fallback; A/B test quality

### Risk: JSON parsing fails
**Mitigation:** Retry with "fix your JSON" prompt; fallback to old generator

### Risk: Routing logic incorrect
**Mitigation:** Explicit `agent_type` parameter in API; manual override available

### Risk: Cost doesn't actually decrease
**Mitigation:** Add token tracking to measure before/after; log all requests

## Open Questions

1. **Caching:** Should CoderAgent cache outputs by idempotent_key?
2. **Streaming:** Can we stream CoderAgent JSON output for real-time updates?
3. **Validation:** Run Ruff/mypy inside CoderAgent or in executor?
4. **Context:** Should CoderAgent read existing files or just receive snippets?
5. **Feedback:** How does CoderAgent learn from test failures?

## Next Steps After CoderAgent

1. **Extract ResearchAgent** - Web search specialist (GPT-4o-mini)
2. **Extract PlannerAgent** - HTN decomposition specialist
3. **Add MemoryAgent** - Retrieval specialist (cheap embeddings)
4. **Build Orchestrator** - Multi-agent coordination for complex tasks

---

Ready to start Phase 1?
