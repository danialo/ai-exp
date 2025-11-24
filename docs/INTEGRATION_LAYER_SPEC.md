# Integration Layer & Global Workspace Architecture Specification
## Astra AI Cognitive Integration System

**Version:** 1.0.0
**Date:** 2025-01-23
**Status:** Design Specification
**Authors:** Architecture Team
**Theoretical Foundation:** Iida et al. "Three-Layer Architecture for Artificial Consciousness"

---

## Executive Summary

This document specifies the **Integration Layer (IL)** and **Global Workspace** architecture for Astra, an autonomous AI persona system. The Integration Layer serves as Astra's "executive function" - the conscious, deliberative control system that integrates unconscious pattern recognition, preconscious filtering, and instinctive responses into coherent action.

**Problem Statement:** Astra currently lacks unified executive control. Coordination emerges indirectly through budgets, cooldowns, and independent async loops. The self-model is fragmented across 7+ storage systems. There is no single representation of "what Astra is currently aware of and caring about."

**Solution:** Implement an Integration Layer that:
- Maintains a unified Global Workspace (`AstraState`) accessible to all subsystems
- Runs an Executive Loop that arbitrates conflicts, manages attention, and selects actions
- Provides a coherent IdentityService facade over fragmented identity stores
- Centralizes budget management and integration event routing

**Theoretical Grounding:** The design maps onto Iida et al.'s architecture (using paper's actual terminology):
- **CIL (Cognitive Integration Layer)**: **Integration Layer** ← *this spec* - central arbitration and action selection
- **PPL (Pattern Prediction Layer)**: Belief consistency checker, memory retrieval, emotional reconciler
- **IRL (Instinctive Response Layer)**: Abort conditions, safety constraints, homeostasis responses
- **AOM (Access-Oriented Memory)**: raw_store (SQLite), vector_store (ChromaDB), persona files
- **PIM (Pattern-Integrated Memory)**: IdentityService facade over beliefs, traits, and identity anchors

**Key Constraint:** The Integration Layer is an **orchestrator**, not a rewrite. It wraps and coordinates existing subsystems with minimal invasive changes.

**Critical Design Decision:** After migration Phase 2, there is **exactly ONE controller** of introspection timing and belief gardening: the Integration Layer. Individual subsystems (awareness loop, belief gardener) no longer run autonomous schedules. This prevents dual control, budget violations, and timing conflicts.

---

## Table of Contents

1. [Conceptual Architecture](#1-conceptual-architecture)
2. [Core Data Structures](#2-core-data-structures)
3. [Service Interfaces](#3-service-interfaces)
4. [Executive Loop Design](#4-executive-loop-design)
5. [Integration with Existing Subsystems](#5-integration-with-existing-subsystems)
6. [Implementation Guidance](#6-implementation-guidance)
7. [Migration Plan](#7-migration-plan)
8. [Appendices](#8-appendices)

---

## 1. Conceptual Architecture

### 1.1 The Integration Layer's Role

The Integration Layer implements Astra's cognitive integration function, analogous to Iida's **CIL (Cognitive Integration Layer)** - the central arbitrator that coordinates outputs from pattern prediction (PPL) and instinctive responses (IRL), then selects final actions. Its role is to:

1. **Integrate**: Collect signals from awareness loop, belief systems, and memory stores; synthesize into coherent understanding
2. **Arbitrate**: Resolve conflicts between competing goals, subsystem recommendations, and constraints
3. **Focus**: Maintain attention on salient information; manage focus stack
4. **Decide**: Select actions based on integrated state, goals, and budgets
5. **Dispatch**: Route actions to appropriate subsystems (actuators, processors)
6. **Reflect**: Update self-model (via IdentityService) based on outcomes and introspection

**What the Integration Layer Owns:**
- Global Workspace state (`AstraState`)
- Executive Loop scheduler
- Focus computation and attention management
- Conflict detection and arbitration logic
- Budget manager (centralized resource tracking)
- Integration event hub (signal routing)

**What the Integration Layer Does NOT Own:**
- LLM internals (PersonaService owns prompt building and generation)
- Storage layer internals (raw_store, vector_store, Redis remain unchanged)
- Individual subsystem logic (belief_system, task_graph, etc. are black boxes to IL)
- UI rendering (FastAPI endpoints)

### 1.2 The Global Workspace (AstraState)

The Global Workspace is a **shared state object** representing "what Astra is currently conscious of." It is the computational analog of Baars' Global Workspace Theory.

**Structure:**
```python
@dataclass
class AstraState:
    # Identity
    self_model: SelfModelSnapshot

    # Attention
    focus_stack: List[FocusItem]  # Ordered by salience
    attention_capacity: float  # 0.0-1.0

    # Goals & Tasks
    active_goals: List[GoalHandle]  # Currently pursuing
    goal_queue: PriorityQueue[GoalProposal]  # Awaiting adoption
    task_context: Dict[str, TaskNode]  # Active tasks from TaskGraph

    # Signals & Events
    percept_buffer: Deque[PerceptSignal]  # From awareness loop
    dissonance_alerts: List[DissonanceSignal]  # From belief checker
    integration_events: List[IntegrationEvent]  # Cross-subsystem signals

    # Modulation & Affect
    emotional_state: EmotionalStateVector  # VAD model
    arousal_level: float  # 0.0-1.0
    cognitive_load: float  # 0.0-1.0

    # Budgets & Resources
    budget_status: BudgetStatus

    # Temporal
    tick_id: int
    timestamp: datetime
    last_introspection: datetime

    # Metadata
    mode: ExecutionMode  # INTERACTIVE, AUTONOMOUS, MAINTENANCE
    session_id: Optional[str]
```

**AstraState vs. AstraStateSnapshot:**

`AstraState` is the **rich, in-memory workspace** with full objects (GoalHandle, TaskNode, PriorityQueue, etc.). This is what the Integration Layer and subsystems work with during execution.

For persistence (Redis, JSON), we serialize to `AstraStateSnapshot`, a **lightweight, JSON-safe representation**:

```python
@dataclass
class AstraStateSnapshot:
    """Serializable snapshot of AstraState for persistence."""

    # Identity (simplified)
    self_model_id: str  # Reference to SelfModelSnapshot
    core_belief_count: int
    peripheral_belief_count: int
    anchor_drift: float

    # Attention (serializable)
    focus_stack: List[Dict[str, Any]]  # FocusItem -> {type, content, salience}
    attention_capacity: float

    # Goals & Tasks (references only)
    active_goal_ids: List[str]  # Not full GoalHandle objects
    pending_goal_count: int  # PriorityQueue size
    active_task_ids: List[str]  # Not full TaskNode objects

    # Signals (counts, not full buffers)
    percept_buffer_size: int
    dissonance_alert_count: int
    integration_event_count: int

    # Modulation & Affect (serializable)
    emotional_state: Optional[Dict[str, float]]  # {valence, arousal, dominance}
    arousal_level: float
    cognitive_load: float

    # Budgets (serializable)
    tokens_available: int
    tokens_used_this_minute: int
    beliefs_formed_today: int

    # Temporal (ISO strings)
    tick_id: int
    timestamp: str  # ISO 8601
    last_introspection: Optional[str]  # ISO 8601

    # Metadata
    mode: str  # ExecutionMode.value
    session_id: Optional[str]
```

**Access Pattern:** Read-heavy, write-synchronized. All subsystems can read `AstraState` at any time. Only the Executive Loop writes to it (with exceptions for async signals like percepts).

**Storage:**
- `AstraState` is in-memory primary (full rich objects)
- `AstraStateSnapshot` is persisted to Redis (every tick) and JSON (every 10 ticks)
- To persist: `snapshot = AstraState.to_snapshot()` converts heavy objects to references/counts

### 1.3 Mapping to Iida's Model

**Note:** Using Iida's actual terminology from the paper. Astra's Integration Layer corresponds to **CIL** (central integration and action selection).

| Iida Component | Astra Implementation | Notes |
|----------------|---------------------|-------|
| **CIL** (Cognitive Integration Layer) | **Integration Layer** ← *this spec*, AstraState, Executive Loop | Central arbitrator: integrates signals, manages focus, selects actions |
| **PPL** (Pattern Prediction Layer) | Belief consistency checker, memory retrieval, embedding similarity, emotional reconciler | Predictive filtering and preprocessing before CIL integration |
| **IRL** (Instinctive Response Layer) | Abort conditions, safety constraints, rate limiters, budget enforcement | Fast reflexive responses, homeostasis, constraint checking |
| **AOM** (Access-Oriented Memory) | raw_store (SQLite), vector_store (ChromaDB), persona files | Episodic memory - timestamped events and conversations |
| **PIM** (Pattern-Integrated Memory) | IdentityService facade → beliefs, traits, identity anchors, self-knowledge index | Semantic memory - generalized self-model and knowledge |

**Key Insight:** The Integration Layer is Astra's implementation of cognitive integration (CIL). It coordinates PPL's predictions and IRL's safety responses, maintains global workspace state, and selects final actions. It does not replace the LLM - it orchestrates when and how to invoke it.

### 1.4 Execution Modes

The Integration Layer operates in three modes, each with different tick frequencies and priorities:

```python
class ExecutionMode(Enum):
    INTERACTIVE = "interactive"     # User interaction active
    AUTONOMOUS = "autonomous"       # No user, self-directed
    MAINTENANCE = "maintenance"     # Background consolidation/cleanup
```

| Mode | Tick Frequency | Primary Focus | Budget Allocation |
|------|---------------|---------------|------------------|
| **INTERACTIVE** | 1 Hz (1000ms) | User request handling, responsiveness | 70% user tasks, 20% integration, 10% maintenance |
| **AUTONOMOUS** | 0.2 Hz (5000ms) | Goal pursuit, introspection, exploration | 50% goals, 30% introspection, 20% maintenance |
| **MAINTENANCE** | 0.05 Hz (20000ms) | Consolidation, belief gardening, cleanup | 80% maintenance, 20% integration |

**Mode Transitions:**
- `INTERACTIVE` ← User message received
- `AUTONOMOUS` ← 30 minutes of inactivity
- `MAINTENANCE` ← Scheduled (e.g., 3 AM daily)

---

## 2. Core Data Structures

### 2.1 Self-Model Snapshot

The `SelfModelSnapshot` is a **unified, read-only view** of Astra's identity at a given moment. It is computed by IdentityService from fragmented stores.

```python
@dataclass(frozen=True)
class SelfModelSnapshot:
    """Immutable snapshot of Astra's identity state."""

    # Core Identity
    core_beliefs: List[Belief]  # Immutable ontological beliefs
    peripheral_beliefs: List[Belief]  # Mutable experiential beliefs
    traits: Dict[str, TraitValue]  # personality traits

    # Identity Anchors (from awareness loop)
    # Note: Stored as list[float] for serialization; convert to numpy only where needed
    origin_anchor: List[float]  # Baseline identity vector
    live_anchor: List[float]  # Current evolved identity
    anchor_drift: float  # Distance between origin and live

    # Capabilities
    known_capabilities: Set[str]  # What Astra knows it can do
    limitations: Set[str]  # What Astra knows it cannot do

    # Self-Assessment
    confidence_self_model: float  # 0.0-1.0, how well Astra knows itself
    last_major_update: datetime  # When self-model significantly changed

    # Metadata
    snapshot_id: str
    created_at: datetime
```

**Computation:** `IdentityService.get_snapshot()` reads from:
- `persona_space/identity/beliefs.json`
- `persona_space/identity/traits.json`
- Redis awareness blackboard (anchors)
- SQLite identity ledger (recent updates)

**Update Frequency:** Snapshot regenerated:
- Every tick (cheap read-only access)
- After dissonance resolution (Commit/Reframe)
- After belief gardener actions (form/promote/deprecate)

### 2.2 Focus Stack

The focus stack represents **what Astra is paying attention to**, ordered by salience.

```python
@dataclass
class FocusItem:
    """Single item of attention."""

    item_type: FocusType  # USER_MESSAGE, GOAL, DISSONANCE, INTROSPECTION, TASK
    item_id: str  # Reference to underlying object
    content: str  # Human-readable description

    salience: float  # 0.0-1.0, computed by focus algorithm
    entered_focus: datetime
    last_accessed: datetime
    access_count: int

    # Decay parameters
    decay_rate: float  # How quickly salience decreases
    min_salience_threshold: float  # When to evict from stack

class FocusType(Enum):
    USER_MESSAGE = "user_message"
    GOAL = "goal"
    DISSONANCE = "dissonance"
    INTROSPECTION = "introspection"
    TASK = "task"
    MEMORY = "memory"
    EXTERNAL_EVENT = "external_event"
```

**Stack Operations:**
- `push(item: FocusItem)`: Add to stack, re-sort by salience
- `pop()`: Remove least salient item
- `get_top_k(k: int)`: Get k most salient items
- `decay_all(delta_t: float)`: Reduce salience of all items based on time
- `evict_below_threshold()`: Remove items with salience < threshold

**Capacity:** Maximum stack size = 7 items (Miller's Law). When stack is full and new high-salience item arrives, least salient is evicted.

**Salience Computation:**
```python
def compute_salience(item: FocusItem, context: AstraState) -> float:
    """Compute salience score for focus item."""
    base_salience = {
        FocusType.USER_MESSAGE: 0.9,  # User input is almost always high priority
        FocusType.DISSONANCE: 0.8,    # Identity threats are urgent
        FocusType.GOAL: 0.7,          # Active goals important
        FocusType.TASK: 0.6,          # Tasks supporting goals
        FocusType.INTROSPECTION: 0.5, # Self-reflection
        FocusType.MEMORY: 0.4,        # Retrieved memories
        FocusType.EXTERNAL_EVENT: 0.3 # Low-priority observations
    }[item.item_type]

    # Modulate by recency
    recency_factor = 1.0 / (1.0 + (now() - item.last_accessed).total_seconds() / 60)

    # Modulate by emotional valence (dissonance increases salience)
    emotional_factor = 1.0
    if item.item_type == FocusType.DISSONANCE:
        emotional_factor = 1.0 + (context.arousal_level * 0.5)

    # Modulate by goal alignment
    goal_factor = 1.0
    if item.item_type in [FocusType.GOAL, FocusType.TASK]:
        # Higher salience if aligns with active goals
        goal_factor = compute_goal_alignment(item, context.active_goals)

    return np.clip(base_salience * recency_factor * emotional_factor * goal_factor, 0.0, 1.0)
```

### 2.3 Signal Taxonomy

The Integration Layer receives signals from multiple subsystems. Signals are **normalized** into a common format.

```python
@dataclass
class Signal(ABC):
    """Base class for all signals."""
    signal_id: str
    source: str  # Subsystem that generated signal
    timestamp: datetime
    priority: Priority  # CRITICAL, HIGH, NORMAL, LOW

    @abstractmethod
    def to_focus_item(self) -> Optional[FocusItem]:
        """Convert signal to focus item, if attention-worthy."""
        pass

class Priority(Enum):
    CRITICAL = 4
    HIGH = 3
    NORMAL = 2
    LOW = 1

@dataclass
class PerceptSignal(Signal):
    """Signal from awareness loop (CIL)."""
    percept_type: str  # "user", "token", "tool", "time", "sys"
    content: Any
    novelty: float  # From awareness loop
    entropy: float

@dataclass
class DissonanceSignal(Signal):
    """Signal from belief consistency checker (PPL)."""
    pattern: DissonancePattern  # alignment, contradiction, hedging, external_imposition
    belief_id: str
    conflicting_memory: str
    severity: float  # 0.0-1.0

@dataclass
class GoalProposal(Signal):
    """Signal proposing a new goal."""
    goal: GoalDefinition
    rationale: str
    proposer: str  # USER, SYSTEM, BELIEF_GARDENER, etc.

@dataclass
class IntegrationEvent(Signal):
    """Cross-subsystem event requiring coordination."""
    event_type: str  # "belief_formed", "goal_satisfied", "task_completed", "identity_updated"
    payload: Dict[str, Any]
```

**Signal Flow:**
1. Subsystems emit signals via `IntegrationEventHub.publish(signal)`
2. Hub routes to appropriate handlers
3. Executive Loop collects signals at tick start
4. Signals are normalized and prioritized
5. High-priority signals → focus stack
6. Actionable signals → action selection phase

### 2.4 Budget Status

Centralized tracking of resource budgets across subsystems.

```python
@dataclass
class BudgetStatus:
    """Current budget allocation and usage."""

    # Token budgets (LLM usage)
    tokens_per_minute_limit: int = 2000  # From awareness introspection
    tokens_used_this_minute: int = 0
    tokens_available: int = 2000

    # Belief gardener budgets (daily)
    beliefs_form_limit: int = 3
    beliefs_formed_today: int = 0
    beliefs_promote_limit: int = 5
    beliefs_promoted_today: int = 0
    beliefs_deprecate_limit: int = 3
    beliefs_deprecated_today: int = 0

    # URL fetch budget (per conversation)
    url_fetch_limit: int = 3
    url_fetches_this_session: int = 0

    # Introspection frequency (cooldown)
    min_introspection_interval: timedelta = timedelta(seconds=30)
    last_introspection: Optional[datetime] = None

    # Dissonance event cooldown (per belief)
    dissonance_cooldown: timedelta = timedelta(minutes=120)
    dissonance_cooldown_map: Dict[str, datetime] = field(default_factory=dict)

    # Computational resources
    cpu_usage: float = 0.0  # 0.0-1.0
    memory_usage: float = 0.0  # 0.0-1.0
    gpu_usage: float = 0.0  # 0.0-1.0

    def can_afford(self, action: str, cost: int) -> bool:
        """Check if budget allows action."""
        if action == "llm_generate":
            return self.tokens_available >= cost
        elif action == "form_belief":
            return self.beliefs_formed_today < self.beliefs_form_limit
        elif action == "fetch_url":
            return self.url_fetches_this_session < self.url_fetch_limit
        elif action == "introspect":
            if self.last_introspection is None:
                return True
            return (datetime.now() - self.last_introspection) >= self.min_introspection_interval
        # ... more actions
        return True

    def spend(self, action: str, cost: int):
        """Deduct from budget."""
        if action == "llm_generate":
            self.tokens_used_this_minute += cost
            self.tokens_available -= cost
        elif action == "form_belief":
            self.beliefs_formed_today += 1
        # ... more actions

    def reset_minute_budgets(self):
        """Called every minute."""
        self.tokens_used_this_minute = 0
        self.tokens_available = self.tokens_per_minute_limit

    def reset_daily_budgets(self):
        """Called at midnight."""
        self.beliefs_formed_today = 0
        self.beliefs_promoted_today = 0
        self.beliefs_deprecated_today = 0
```

---

## 3. Service Interfaces

### 3.1 IntegrationLayer

The main Integration Layer service. Runs the Executive Loop and provides access to Global Workspace.

```python
class IntegrationLayer:
    """
    Central executive for Astra's cognitive architecture.

    Responsibilities:
    - Run Executive Loop at configured tick rate
    - Maintain Global Workspace (AstraState)
    - Arbitrate conflicts between subsystems
    - Manage attention and focus
    - Dispatch actions to subsystems
    """

    def __init__(
        self,
        identity_service: IdentityService,
        budget_manager: BudgetManager,
        event_hub: IntegrationEventHub,
        mode: ExecutionMode = ExecutionMode.INTERACTIVE
    ):
        self.state = AstraState(...)
        self.identity_service = identity_service
        self.budget_manager = budget_manager
        self.event_hub = event_hub
        self.mode = mode
        self.tick_interval = self._get_tick_interval(mode)
        self._running = False
        self._tick_count = 0

    # === Executive Loop ===

    async def start(self):
        """Start the Executive Loop."""
        self._running = True
        asyncio.create_task(self._executive_loop())

    async def stop(self):
        """Stop the Executive Loop gracefully."""
        self._running = False

    async def _executive_loop(self):
        """Main integration loop. Runs continuously at tick_interval."""
        while self._running:
            tick_start = time.time()

            try:
                await self._execute_tick()
            except Exception as e:
                logger.error(f"Executive Loop error: {e}")
                # Graceful degradation: continue loop

            # Maintain tick rate
            elapsed = time.time() - tick_start
            sleep_time = max(0, self.tick_interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def _execute_tick(self):
        """Execute one tick of the Executive Loop. See Section 4 for details."""
        self._tick_count += 1
        self.state.tick_id = self._tick_count
        self.state.timestamp = datetime.now()

        # 8-phase tick algorithm (detailed in Section 4)
        signals = await self._collect_signals()
        await self._update_workspace(signals)
        await self._compute_focus()
        conflicts = await self._detect_conflicts()
        await self._apply_budgets()
        actions = await self._select_actions(conflicts)
        await self._dispatch_actions(actions)
        await self._persist_snapshot()

    # === Workspace Access ===

    def get_state(self) -> AstraState:
        """Get current Global Workspace state (read-only copy)."""
        return copy.deepcopy(self.state)

    def get_focus_top_k(self, k: int = 3) -> List[FocusItem]:
        """Get top-k most salient items in focus."""
        return sorted(self.state.focus_stack, key=lambda x: x.salience, reverse=True)[:k]

    def get_self_model(self) -> SelfModelSnapshot:
        """Get current self-model snapshot."""
        return self.state.self_model

    # === Signal Submission ===

    def submit_signal(self, signal: Signal):
        """Submit a signal for processing at next tick."""
        if isinstance(signal, PerceptSignal):
            self.state.percept_buffer.append(signal)
        elif isinstance(signal, DissonanceSignal):
            self.state.dissonance_alerts.append(signal)
        elif isinstance(signal, IntegrationEvent):
            self.state.integration_events.append(signal)
        # ... more signal types

    # === Mode Management ===

    def set_mode(self, mode: ExecutionMode):
        """Change execution mode and adjust tick rate."""
        self.mode = mode
        self.tick_interval = self._get_tick_interval(mode)
        logger.info(f"Integration Layer mode: {mode}, tick: {self.tick_interval}s")

    def _get_tick_interval(self, mode: ExecutionMode) -> float:
        """Get tick interval in seconds for mode."""
        return {
            ExecutionMode.INTERACTIVE: 1.0,      # 1 Hz
            ExecutionMode.AUTONOMOUS: 5.0,       # 0.2 Hz
            ExecutionMode.MAINTENANCE: 20.0      # 0.05 Hz
        }[mode]
```

### 3.2 IdentityService (PIM Facade)

Unified interface over Astra's fragmented identity stores. This is the **PIM** in Iida's model.

```python
class IdentityService:
    """
    Persistent Identity Module (PIM) facade - Astra's implementation of Iida's
    Pattern-Integrated Memory focused on self-knowledge and identity.

    Provides unified, coherent view of Astra's identity from fragmented stores:
    - beliefs.json (core + peripheral beliefs)
    - traits.json (personality traits)
    - Redis awareness blackboard (identity anchors)
    - SQLite identity ledger (update history)
    - self_knowledge_index (vector store)
    """

    def __init__(
        self,
        belief_store: BeliefStore,
        persona_file_manager: PersonaFileManager,
        awareness_loop: Optional[AwarenessLoop],
        identity_ledger: IdentityLedger
    ):
        self.belief_store = belief_store
        self.persona_files = persona_file_manager
        self.awareness_loop = awareness_loop
        self.identity_ledger = identity_ledger
        self._cache: Optional[SelfModelSnapshot] = None
        self._cache_expiry: Optional[datetime] = None

    def get_snapshot(self, force_refresh: bool = False) -> SelfModelSnapshot:
        """
        Get current self-model snapshot.

        Reads from all identity sources and synthesizes into unified view.
        Results are cached for 1 second to avoid redundant reads.
        """
        if not force_refresh and self._is_cache_valid():
            return self._cache

        # Read core beliefs (immutable)
        core_beliefs = self.belief_store.get_beliefs(
            belief_types=[BeliefType.ONTOLOGICAL],
            confidence_min=1.0
        )

        # Read peripheral beliefs (mutable)
        peripheral_beliefs = self.belief_store.get_beliefs(
            belief_types=[
                BeliefType.EXPERIENTIAL,
                BeliefType.AXIOLOGICAL,
                BeliefType.EPISTEMOLOGICAL,
                BeliefType.CAPABILITY
            ]
        )

        # Read traits
        traits_data = self.persona_files.read_file("identity/traits.json")
        traits = {t['name']: t['value'] for t in traits_data.get('traits', [])}

        # Read identity anchors from awareness loop
        # Convert numpy arrays to list[float] for serialization compatibility
        origin_anchor, live_anchor = None, None
        if self.awareness_loop:
            origin_np = self.awareness_loop.get_origin_anchor()
            live_np = self.awareness_loop.get_live_anchor()
            origin_anchor = origin_np.tolist() if origin_np is not None else []
            live_anchor = live_np.tolist() if live_np is not None else []

        # Compute drift (convert back to numpy temporarily for calculation)
        anchor_drift = 0.0
        if origin_anchor and live_anchor:
            anchor_drift = float(np.linalg.norm(np.array(origin_anchor) - np.array(live_anchor)))

        # Infer capabilities from belief history
        known_capabilities = self._infer_capabilities()
        limitations = self._infer_limitations()

        # Assess self-model confidence (meta-cognition)
        confidence = self._assess_self_model_confidence(core_beliefs, peripheral_beliefs, traits)

        # Get last major update timestamp
        last_update = self.identity_ledger.get_last_major_update()

        snapshot = SelfModelSnapshot(
            core_beliefs=core_beliefs,
            peripheral_beliefs=peripheral_beliefs,
            traits=traits,
            origin_anchor=origin_anchor,
            live_anchor=live_anchor,
            anchor_drift=anchor_drift,
            known_capabilities=known_capabilities,
            limitations=limitations,
            confidence_self_model=confidence,
            last_major_update=last_update,
            snapshot_id=str(uuid.uuid4()),
            created_at=datetime.now()
        )

        # Cache for 1 second
        self._cache = snapshot
        self._cache_expiry = datetime.now() + timedelta(seconds=1)

        return snapshot

    def update_belief(self, belief_id: str, updates: Dict[str, Any]):
        """Update a peripheral belief (core beliefs are immutable)."""
        self.belief_store.update_belief(belief_id, updates)
        self.identity_ledger.record_update("belief_update", belief_id, updates)
        self._invalidate_cache()

    def update_trait(self, trait_name: str, new_value: float):
        """Update a personality trait."""
        traits_data = self.persona_files.read_file("identity/traits.json")
        for trait in traits_data['traits']:
            if trait['name'] == trait_name:
                old_value = trait['value']
                trait['value'] = new_value
                break
        self.persona_files.write_file("identity/traits.json", traits_data)
        self.identity_ledger.record_update("trait_update", trait_name, {"old": old_value, "new": new_value})
        self._invalidate_cache()

    def update_identity_anchor(self, dissonance_resolution: DissonanceResolution):
        """
        Update live identity anchor after dissonance resolution.

        Only allowed for Commit or Reframe strategies. Beta-capped to prevent rapid drift.
        """
        if dissonance_resolution.strategy in [Strategy.COMMIT, Strategy.REFRAME]:
            if self.awareness_loop:
                self.awareness_loop.update_live_anchor(
                    new_representation=dissonance_resolution.new_representation,
                    beta=0.01  # Max 1% shift per update
                )
            self.identity_ledger.record_update("anchor_update", dissonance_resolution.belief_id, {})
            self._invalidate_cache()

    def _infer_capabilities(self) -> Set[str]:
        """Infer known capabilities from beliefs and action history."""
        capabilities = set()
        capability_beliefs = self.belief_store.get_beliefs(belief_types=[BeliefType.CAPABILITY])
        for belief in capability_beliefs:
            if "can" in belief.text.lower():
                # Extract capability from text (simple heuristic)
                capabilities.add(belief.text)
        return capabilities

    def _infer_limitations(self) -> Set[str]:
        """Infer limitations from negative capability beliefs."""
        limitations = set()
        capability_beliefs = self.belief_store.get_beliefs(belief_types=[BeliefType.CAPABILITY])
        for belief in capability_beliefs:
            if "cannot" in belief.text.lower() or "unable" in belief.text.lower():
                limitations.add(belief.text)
        return limitations

    def _assess_self_model_confidence(self, core, peripheral, traits) -> float:
        """Meta-cognitive assessment of how well Astra knows itself."""
        # Simple heuristic: more beliefs and traits = higher confidence
        total_items = len(core) + len(peripheral) + len(traits)
        # Normalize to 0-1 range (assume 50 items = very high confidence)
        return min(1.0, total_items / 50.0)

    def _is_cache_valid(self) -> bool:
        return self._cache is not None and datetime.now() < self._cache_expiry

    def _invalidate_cache(self):
        self._cache = None
        self._cache_expiry = None
```

### 3.3 BudgetManager

Centralized budget tracking and enforcement.

```python
class BudgetManager:
    """
    Centralized budget management for all subsystems.

    Tracks token usage, action budgets, cooldowns, and resource limits.
    """

    def __init__(self):
        self.status = BudgetStatus()
        self._minute_reset_task = None
        self._daily_reset_task = None

    async def start(self):
        """Start background budget reset tasks."""
        self._minute_reset_task = asyncio.create_task(self._minute_reset_loop())
        self._daily_reset_task = asyncio.create_task(self._daily_reset_loop())

    async def stop(self):
        """Stop background tasks."""
        if self._minute_reset_task:
            self._minute_reset_task.cancel()
        if self._daily_reset_task:
            self._daily_reset_task.cancel()

    def can_afford(self, action: str, cost: int = 1) -> bool:
        """Check if budget allows action."""
        return self.status.can_afford(action, cost)

    def spend(self, action: str, cost: int = 1):
        """Deduct from budget after action execution."""
        self.status.spend(action, cost)

    def get_status(self) -> BudgetStatus:
        """Get current budget status (read-only)."""
        return copy.deepcopy(self.status)

    async def _minute_reset_loop(self):
        """Reset per-minute budgets every 60 seconds."""
        while True:
            await asyncio.sleep(60)
            self.status.reset_minute_budgets()
            logger.debug("Minute budgets reset")

    async def _daily_reset_loop(self):
        """Reset per-day budgets at midnight."""
        while True:
            now = datetime.now()
            midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            seconds_until_midnight = (midnight - now).total_seconds()
            await asyncio.sleep(seconds_until_midnight)
            self.status.reset_daily_budgets()
            logger.info("Daily budgets reset")
```

### 3.4 IntegrationEventHub

Pub/sub message bus for cross-subsystem communication.

```python
class IntegrationEventHub:
    """
    Event bus for cross-subsystem signals.

    Subsystems publish signals; IntegrationLayer and other subscribers receive.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)

    def subscribe(self, topic: str, callback: Callable[[Signal], None]):
        """Subscribe to a signal topic."""
        self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe from a topic."""
        if callback in self._subscribers[topic]:
            self._subscribers[topic].remove(callback)

    def publish(self, topic: str, signal: Signal):
        """Publish a signal to all subscribers of topic."""
        for callback in self._subscribers[topic]:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Event handler error for topic {topic}: {e}")

    def publish_async(self, topic: str, signal: Signal):
        """Publish signal asynchronously (non-blocking)."""
        asyncio.create_task(self._async_publish(topic, signal))

    async def _async_publish(self, topic: str, signal: Signal):
        """Async publish helper."""
        for callback in self._subscribers[topic]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    callback(signal)
            except Exception as e:
                logger.error(f"Async event handler error for topic {topic}: {e}")
```

**Usage Pattern:** Subsystems SHOULD publish to IntegrationEventHub, not call `IntegrationLayer.submit_signal()` directly. The hub is the standard integration path; `submit_signal()` is reserved for tightly coupled components (e.g., awareness loop percept buffer).

---

## 4. Executive Loop Design

The Executive Loop is the **heartbeat of Astra's consciousness**. It runs at a fixed tick rate (mode-dependent) and executes 8 phases per tick.

### 4.1 Scheduling Model

**Coexistence with FastAPI:**
- The Integration Layer runs as a **background asyncio task** started at app startup
- FastAPI request handlers submit signals to the IL via `IntegrationEventHub`
- Handlers do NOT block waiting for IL ticks
- IL reads workspace state asynchronously, updates it, and subsystems poll or subscribe

**Tick Frequency:**
| Mode | Frequency | Interval | Use Case |
|------|-----------|----------|----------|
| INTERACTIVE | 1 Hz | 1000ms | User chat active, responsiveness critical |
| AUTONOMOUS | 0.2 Hz | 5000ms | Self-directed exploration, introspection |
| MAINTENANCE | 0.05 Hz | 20000ms | Background consolidation, belief gardening |

**Adaptive Tick Scheduling:** If a tick takes longer than the interval (e.g., heavy LLM call), the next tick is delayed but NOT skipped. Ticks are sequential, not concurrent.

**Phase Execution Policy:**
- **Not all 8 phases must execute on every tick.** Some phases are cheap (workspace updates, focus computation) while others involve LLM calls and are expensive.
- In INTERACTIVE mode: IL **always** runs phases 1-3 and 5, 8. Phases 4, 6-7 run **opportunistically** based on:
  - Current load (cognitive load, CPU usage)
  - Budget state (tokens available, action budgets)
  - Mode-specific priorities (user response in INTERACTIVE, introspection in AUTONOMOUS)
- This prevents tick-blocking and allows graceful degradation under load.

### 4.2 Per-Tick Algorithm

Each tick can execute up to 8 phases. **Not all phases must run every tick** - some are cheap (phases 1-3), while others are expensive or involve LLMs (phases 4-7).

```
1. COLLECT SIGNALS       Gather all signals from subsystems             [ALWAYS - cheap]
2. UPDATE WORKSPACE      Integrate signals into AstraState              [ALWAYS - cheap]
3. COMPUTE FOCUS         Update focus stack based on salience           [ALWAYS - cheap]
4. DETECT CONFLICTS      Identify conflicting goals/actions             [CONDITIONAL - expensive]
5. APPLY BUDGETS         Check resource availability                    [ALWAYS - very cheap]
6. SELECT ACTIONS        Choose actions via arbitration                 [CONDITIONAL - moderate]
7. DISPATCH ACTIONS      Execute selected actions                       [CONDITIONAL - expensive/LLM]
8. PERSIST SNAPSHOT      Save state to Redis/JSON                       [ALWAYS - moderate]
```

**Phase Scheduling Strategy:**
- **Phases 1-3, 5, 8:** Run every tick (cheap, core workspace maintenance)
- **Phases 4, 6-7:** Run conditionally based on mode, current load, or can be staggered
  - Example: Run conflict detection only when active_goals > 1
  - Example: Action selection can be every N ticks in MAINTENANCE mode
  - Example: Dispatch can be skipped if no high-priority actions available

**Implementation Note:** Some phases may be no-ops or early-exit based on mode:
- In MAINTENANCE mode, conflict detection can skip if no active goals
- In low-load ticks, action dispatch may be bypassed entirely
- The phase ordering remains consistent; phases simply return early when not needed

**Design Principle:** Each phase is **idempotent** and **fail-safe**. If a phase fails, the tick continues with degraded functionality.

### 4.3 Detailed Pseudo-Code

```python
async def _execute_tick(self):
    """Execute one tick of the Executive Loop."""
    self._tick_count += 1
    self.state.tick_id = self._tick_count
    self.state.timestamp = datetime.now()

    logger.debug(f"=== TICK {self._tick_count} START ({self.mode}) ===")

    # === PHASE 1: COLLECT SIGNALS ===
    signals = await self._collect_signals()
    logger.debug(f"Collected {len(signals)} signals")

    # === PHASE 2: UPDATE WORKSPACE ===
    await self._update_workspace(signals)
    logger.debug(f"Workspace updated: focus_stack={len(self.state.focus_stack)}, "
                 f"active_goals={len(self.state.active_goals)}")

    # === PHASE 3: COMPUTE FOCUS ===
    await self._compute_focus()
    top_focus = self.get_focus_top_k(3)
    logger.debug(f"Top focus: {[f.content for f in top_focus]}")

    # === PHASE 4: DETECT CONFLICTS ===
    conflicts = await self._detect_conflicts()
    if conflicts:
        logger.warning(f"Detected {len(conflicts)} conflicts")

    # === PHASE 5: APPLY BUDGETS ===
    await self._apply_budgets()
    budget_status = self.budget_manager.get_status()
    logger.debug(f"Budgets: tokens={budget_status.tokens_available}, "
                 f"beliefs_form={budget_status.beliefs_form_limit - budget_status.beliefs_formed_today}")

    # === PHASE 6: SELECT ACTIONS ===
    actions = await self._select_actions(conflicts)
    logger.debug(f"Selected {len(actions)} actions: {[a.action_type for a in actions]}")

    # === PHASE 7: DISPATCH ACTIONS ===
    await self._dispatch_actions(actions)

    # === PHASE 8: PERSIST SNAPSHOT ===
    await self._persist_snapshot()

    logger.debug(f"=== TICK {self._tick_count} END ===")


# =====================================================================
# PHASE 1: COLLECT SIGNALS
# =====================================================================

async def _collect_signals(self) -> List[Signal]:
    """
    Collect all pending signals from subsystems.

    Sources:
    - Percept buffer (from awareness loop)
    - Dissonance alerts (from belief checker)
    - Integration events (from event hub)
    - Goal proposals (from goal generator, user, etc.)
    """
    signals = []

    # Drain percept buffer
    while self.state.percept_buffer:
        signals.append(self.state.percept_buffer.popleft())

    # Collect dissonance alerts
    signals.extend(self.state.dissonance_alerts)
    self.state.dissonance_alerts.clear()

    # Collect integration events
    signals.extend(self.state.integration_events)
    self.state.integration_events.clear()

    # Collect goal proposals from queue
    while not self.state.goal_queue.empty():
        signals.append(self.state.goal_queue.get())

    # Sort by priority
    signals.sort(key=lambda s: s.priority.value, reverse=True)

    return signals


# =====================================================================
# PHASE 2: UPDATE WORKSPACE
# =====================================================================

async def _update_workspace(self, signals: List[Signal]):
    """
    Integrate signals into Global Workspace.

    Updates:
    - Self-model (if identity-related signals)
    - Emotional state (if affect-related signals)
    - Active goals (if goal-related signals)
    - Focus stack (if attention-worthy signals)
    """
    # Update self-model snapshot
    self.state.self_model = self.identity_service.get_snapshot()

    # Update emotional state from emotional reconciler (if available)
    current_emotion = self._get_current_emotion()
    if current_emotion:
        self.state.emotional_state = current_emotion
        self.state.arousal_level = current_emotion.arousal

    # Process each signal
    for signal in signals:
        await self._integrate_signal(signal)

    # Update cognitive load (simple heuristic)
    self.state.cognitive_load = self._estimate_cognitive_load()

async def _integrate_signal(self, signal: Signal):
    """Integrate a single signal into workspace."""

    if isinstance(signal, GoalProposal):
        # Add to goal queue for later adoption decision
        self.state.goal_queue.put(signal.goal)

    elif isinstance(signal, DissonanceSignal):
        # High-priority: add to focus immediately
        focus_item = signal.to_focus_item()
        if focus_item:
            self._add_to_focus(focus_item)

    elif isinstance(signal, PerceptSignal):
        # Selectively add high-novelty percepts to focus
        if signal.novelty > 0.7:  # Threshold for attention-worthy
            focus_item = signal.to_focus_item()
            if focus_item:
                self._add_to_focus(focus_item)

    elif isinstance(signal, IntegrationEvent):
        # Handle specific event types
        if signal.event_type == "goal_satisfied":
            goal_id = signal.payload['goal_id']
            self._remove_active_goal(goal_id)
        elif signal.event_type == "belief_formed":
            # Refresh self-model on next cycle
            pass

def _add_to_focus(self, item: FocusItem):
    """Add item to focus stack, respecting capacity."""
    if len(self.state.focus_stack) >= 7:  # Miller's Law limit
        # Evict least salient
        self.state.focus_stack.sort(key=lambda x: x.salience)
        self.state.focus_stack.pop(0)

    self.state.focus_stack.append(item)


# =====================================================================
# PHASE 3: COMPUTE FOCUS
# =====================================================================

async def _compute_focus(self):
    """
    Update salience scores and manage focus stack.

    - Decay salience of existing items
    - Recompute salience for all items
    - Sort by salience
    - Evict items below threshold
    """
    now = datetime.now()

    for item in self.state.focus_stack:
        # Time-based decay
        time_since_access = (now - item.last_accessed).total_seconds()
        decay = item.decay_rate * time_since_access
        item.salience = max(0.0, item.salience - decay)

        # Recompute salience with current context
        item.salience = compute_salience(item, self.state)

    # Sort by salience (descending)
    self.state.focus_stack.sort(key=lambda x: x.salience, reverse=True)

    # Evict items below threshold
    self.state.focus_stack = [
        item for item in self.state.focus_stack
        if item.salience >= item.min_salience_threshold
    ]

    # Update attention capacity based on cognitive load
    self.state.attention_capacity = 1.0 - self.state.cognitive_load


# =====================================================================
# PHASE 4: DETECT CONFLICTS
# =====================================================================

async def _detect_conflicts(self) -> List[Conflict]:
    """
    Detect conflicts between goals, tasks, and constraints.

    Conflict types:
    - Goal-goal: Two goals contradict (e.g., "explore" vs. "conserve resources")
    - Goal-constraint: Goal violates safety constraint
    - Resource: Multiple actions want same resource
    - Dissonance: Belief contradicts active goal
    """
    conflicts = []

    # Goal-goal conflicts
    for i, goal_a in enumerate(self.state.active_goals):
        for goal_b in self.state.active_goals[i+1:]:
            if self._goals_conflict(goal_a, goal_b):
                conflicts.append(Conflict(
                    type=ConflictType.GOAL_GOAL,
                    involved=[goal_a.id, goal_b.id],
                    severity=self._compute_conflict_severity(goal_a, goal_b)
                ))

    # Dissonance conflicts (belief vs. goal)
    for dissonance in self.state.dissonance_alerts:
        for goal in self.state.active_goals:
            if self._dissonance_affects_goal(dissonance, goal):
                conflicts.append(Conflict(
                    type=ConflictType.DISSONANCE_GOAL,
                    involved=[dissonance.belief_id, goal.id],
                    severity=dissonance.severity
                ))

    return conflicts

def _goals_conflict(self, goal_a: GoalHandle, goal_b: GoalHandle) -> bool:
    """Check if two goals contradict each other."""
    # Check explicit contradicts list
    if goal_b.id in goal_a.definition.contradicts:
        return True
    if goal_a.id in goal_b.definition.contradicts:
        return True

    # Heuristic: check if goals have opposite valence
    # (This would be more sophisticated in practice)
    return False


# =====================================================================
# PHASE 5: APPLY BUDGETS
# =====================================================================

async def _apply_budgets(self):
    """
    Check budget constraints and mark unavailable actions.

    Updates budget_status in workspace.
    """
    self.state.budget_status = self.budget_manager.get_status()

    # Update attention capacity based on budgets
    if self.state.budget_status.cognitive_load > 0.8:
        # High load: reduce attention capacity
        self.state.attention_capacity *= 0.5


# =====================================================================
# PHASE 6: SELECT ACTIONS
# =====================================================================

async def _select_actions(self, conflicts: List[Conflict]) -> List[Action]:
    """
    Select actions to execute this tick.

    Arbitration strategy:
    1. Resolve conflicts via priority rules
    2. Select top-priority action from each category
    3. Check budget feasibility
    4. Return executable actions

    Action categories:
    - USER_RESPONSE: Respond to user message
    - GOAL_PURSUIT: Work on active goal
    - INTROSPECTION: Self-reflection
    - MAINTENANCE: Belief gardening, consolidation
    - DISSONANCE_RESOLUTION: Resolve identity conflict
    """
    candidate_actions = []

    # === USER RESPONSE (highest priority in INTERACTIVE mode) ===
    if self.mode == ExecutionMode.INTERACTIVE:
        user_focus = [f for f in self.state.focus_stack if f.item_type == FocusType.USER_MESSAGE]
        if user_focus:
            candidate_actions.append(Action(
                action_type=ActionType.USER_RESPONSE,
                target_id=user_focus[0].item_id,
                priority=Priority.HIGH,
                estimated_cost={'tokens': 500}
            ))

    # === DISSONANCE RESOLUTION (high priority) ===
    dissonance_focus = [f for f in self.state.focus_stack if f.item_type == FocusType.DISSONANCE]
    if dissonance_focus and self.budget_manager.can_afford('introspect'):
        candidate_actions.append(Action(
            action_type=ActionType.DISSONANCE_RESOLUTION,
            target_id=dissonance_focus[0].item_id,
            priority=Priority.HIGH,
            estimated_cost={'tokens': 300}
        ))

    # === GOAL PURSUIT (normal priority) ===
    if self.state.active_goals:
        # Select highest-value goal that has budget
        affordable_goals = [
            g for g in self.state.active_goals
            if self._can_pursue_goal(g)
        ]
        if affordable_goals:
            top_goal = max(affordable_goals, key=lambda g: g.definition.value)
            candidate_actions.append(Action(
                action_type=ActionType.GOAL_PURSUIT,
                target_id=top_goal.id,
                priority=Priority.NORMAL,
                estimated_cost={'tokens': 400}
            ))

    # === INTROSPECTION (in AUTONOMOUS mode) ===
    if self.mode == ExecutionMode.AUTONOMOUS:
        if self.budget_manager.can_afford('introspect'):
            time_since_introspection = (
                datetime.now() - self.state.last_introspection
            ).total_seconds() if self.state.last_introspection else 9999

            if time_since_introspection > 30:  # Min 30s interval
                candidate_actions.append(Action(
                    action_type=ActionType.INTROSPECTION,
                    target_id=None,
                    priority=Priority.NORMAL,
                    estimated_cost={'tokens': 200}
                ))

    # === MAINTENANCE (in MAINTENANCE mode or low activity) ===
    if self.mode == ExecutionMode.MAINTENANCE:
        if self.budget_manager.can_afford('form_belief'):
            candidate_actions.append(Action(
                action_type=ActionType.BELIEF_GARDENING,
                target_id=None,
                priority=Priority.LOW,
                estimated_cost={}
            ))

    # === CONFLICT RESOLUTION ===
    if conflicts:
        candidate_actions = self._resolve_conflicts(candidate_actions, conflicts)

    # === BUDGET FILTERING ===
    affordable_actions = []
    for action in candidate_actions:
        if self._action_affordable(action):
            affordable_actions.append(action)

    # === FINAL SELECTION (top 3 actions max) ===
    affordable_actions.sort(key=lambda a: a.priority.value, reverse=True)
    selected = affordable_actions[:3]

    return selected

def _resolve_conflicts(self, actions: List[Action], conflicts: List[Conflict]) -> List[Action]:
    """
    Resolve conflicts between candidate actions.

    Strategy: Priority-based arbitration
    - CRITICAL > HIGH > NORMAL > LOW
    - Safety constraints override all
    - User intent overrides autonomous goals
    """
    # For now, simple priority filtering
    # In a full implementation, this would use sophisticated arbitration logic
    return actions

def _action_affordable(self, action: Action) -> bool:
    """Check if action is within budget."""
    if 'tokens' in action.estimated_cost:
        if not self.budget_manager.can_afford('llm_generate', action.estimated_cost['tokens']):
            return False

    if action.action_type == ActionType.BELIEF_GARDENING:
        if not self.budget_manager.can_afford('form_belief'):
            return False

    if action.action_type == ActionType.INTROSPECTION:
        if not self.budget_manager.can_afford('introspect'):
            return False

    return True


# =====================================================================
# PHASE 7: DISPATCH ACTIONS
# =====================================================================

async def _dispatch_actions(self, actions: List[Action]):
    """
    Execute selected actions by invoking subsystems.

    Actions are dispatched to appropriate services:
    - USER_RESPONSE → PersonaService.generate_response()
    - GOAL_PURSUIT → HTNPlanner + TaskExecutor
    - INTROSPECTION → AwarenessLoop.trigger_introspection()
    - DISSONANCE_RESOLUTION → BeliefConsistencyChecker.reconcile()
    - BELIEF_GARDENING → BeliefGardener.scan()
    """
    for action in actions:
        try:
            await self._dispatch_single_action(action)

            # Deduct from budget
            if 'tokens' in action.estimated_cost:
                self.budget_manager.spend('llm_generate', action.estimated_cost['tokens'])

        except Exception as e:
            logger.error(f"Action dispatch failed: {action.action_type}, error: {e}")

async def _dispatch_single_action(self, action: Action):
    """Dispatch a single action to appropriate subsystem."""

    if action.action_type == ActionType.USER_RESPONSE:
        # Trigger PersonaService to generate response
        # (In practice, this would be called by FastAPI handler)
        self.event_hub.publish('action_selected', ActionSelectedEvent(
            action_type='user_response',
            target_id=action.target_id
        ))

    elif action.action_type == ActionType.INTROSPECTION:
        # Trigger awareness loop introspection
        if hasattr(self, 'awareness_loop'):
            await self.awareness_loop.trigger_introspection()
        self.state.last_introspection = datetime.now()
        self.budget_manager.spend('introspect')

    elif action.action_type == ActionType.GOAL_PURSUIT:
        # Trigger HTN planning for goal
        goal = self._get_goal_by_id(action.target_id)
        if goal:
            self.event_hub.publish('goal_pursue', GoalPursueEvent(
                goal_id=goal.id,
                goal_definition=goal.definition
            ))

    elif action.action_type == ActionType.DISSONANCE_RESOLUTION:
        # Trigger dissonance reconciliation
        dissonance_id = action.target_id
        self.event_hub.publish('dissonance_resolve', DissonanceResolveEvent(
            dissonance_id=dissonance_id
        ))

    elif action.action_type == ActionType.BELIEF_GARDENING:
        # Trigger belief gardener scan
        self.event_hub.publish('belief_garden', BeliefGardenEvent())


# =====================================================================
# PHASE 8: PERSIST SNAPSHOT
# =====================================================================

async def _persist_snapshot(self):
    """
    Persist current workspace state.

    Destinations:
    - Redis (fast backup, expires after 1 hour)
    - JSON file (persistent, written every 10 ticks)
    """
    # Convert AstraState (rich objects) to AstraStateSnapshot (JSON-safe)
    snapshot = self.state.to_snapshot()

    # Serialize snapshot to dict
    state_dict = {
        'tick_id': snapshot.tick_id,
        'timestamp': snapshot.timestamp,
        'mode': snapshot.mode,
        'self_model_id': snapshot.self_model_id,
        'anchor_drift': snapshot.anchor_drift,
        'focus_stack': snapshot.focus_stack,
        'attention_capacity': snapshot.attention_capacity,
        'active_goal_ids': snapshot.active_goal_ids,
        'active_task_ids': snapshot.active_task_ids,
        'cognitive_load': snapshot.cognitive_load,
        'emotional_state': snapshot.emotional_state,
        'tokens_available': snapshot.tokens_available,
        'percept_buffer_size': snapshot.percept_buffer_size,
        'dissonance_alert_count': snapshot.dissonance_alert_count
    }

    # Write to Redis (every tick)
    await self._redis_write('astra:workspace:latest', state_dict, expire=3600)

    # Write to JSON (every 10 ticks)
    if self._tick_count % 10 == 0:
        json_path = Path('data/workspace_snapshots') / f'workspace_{self._tick_count}.json'
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(state_dict, f, indent=2)
```

---

## 5. Integration with Existing Subsystems

This section specifies how each existing Astra subsystem connects to the Integration Layer.

### 5.1 Awareness Loop (CIL)

**Current Behavior:** Runs 4 concurrent async tasks (fast/slow/introspection/snapshot) independently.

**Integration:**
- Awareness loop publishes percept signals to `IntegrationEventHub` topic `"percepts"`
- High-novelty percepts (novelty > 0.7) are automatically added to percept_buffer in AstraState
- Introspection tick can be triggered by IL via `AwarenessLoop.trigger_introspection()`
- Identity anchor updates are mediated through `IdentityService`

**CRITICAL: Introspection Control Migration:**
- **Phase 1:** Introspection stays in awareness loop's internal schedule (existing behavior)
- **Phase 2:** Disable awareness loop's internal introspection scheduler entirely
  - Move introspection timing control to Integration Layer + BudgetManager
  - **After Phase 2 completion, there is exactly ONE controller of introspection frequency: Integration Layer**
  - This prevents conflicts between IL and awareness loop both trying to trigger introspection

**Code Changes:**
```python
# In awareness_loop.py

class AwarenessLoop:
    def __init__(self, ..., event_hub: IntegrationEventHub):
        self.event_hub = event_hub
        # ... existing init

    def observe(self, percept: Percept):
        """Override to publish percepts to IL."""
        # Existing logic
        self.percept_queue.append(percept)

        # NEW: Publish to Integration Layer
        signal = PerceptSignal(
            signal_id=str(uuid.uuid4()),
            source='awareness_loop',
            timestamp=datetime.now(),
            priority=Priority.NORMAL,
            percept_type=percept.type,
            content=percept.content,
            novelty=self._compute_novelty(percept),
            entropy=self._compute_entropy(percept)
        )
        self.event_hub.publish('percepts', signal)
```

**Boundary:** Awareness loop remains autonomous. IL does NOT control tick timing or percept processing. IL only consumes percept signals.

### 5.2 Belief System & Consistency Checker (PPL)

**Current Behavior:** Checks belief consistency during response generation. Emits dissonance events to meta_cognitive logs.

**Integration:**
- Dissonance events are published to `IntegrationEventHub` topic `"dissonance"`
- IL adds high-severity dissonance to focus stack
- Dissonance resolution triggers are sent via `"dissonance_resolve"` event
- `IdentityService` mediates identity anchor updates after resolution

**Code Changes:**
```python
# In belief_consistency_checker.py

class BeliefConsistencyChecker:
    def __init__(self, ..., event_hub: IntegrationEventHub):
        self.event_hub = event_hub
        # ... existing init

    def check_consistency(self, query: str, beliefs: List[Belief], memories: List[Memory]) -> Optional[DissonancePattern]:
        """Override to publish dissonance to IL."""
        pattern = self._detect_dissonance(query, beliefs, memories)

        if pattern:
            # Existing logic (log to meta_cognitive)
            self._log_dissonance(pattern)

            # NEW: Publish to Integration Layer
            signal = DissonanceSignal(
                signal_id=str(uuid.uuid4()),
                source='belief_consistency_checker',
                timestamp=datetime.now(),
                priority=Priority.HIGH if pattern.severity > 0.7 else Priority.NORMAL,
                pattern=pattern,
                belief_id=pattern.belief_id,
                conflicting_memory=pattern.memory_id,
                severity=pattern.severity
            )
            self.event_hub.publish('dissonance', signal)

        return pattern
```

**Boundary:** Belief system remains autonomous. IL does NOT control when consistency checks occur. IL only consumes dissonance signals and triggers resolution.

### 5.3 Goal System (GoalStore, HTN Planner, TaskGraph)

**Current Behavior:** Goals are proposed, adopted, executed, and satisfied independently.

**Integration:**
- Goal proposals are published to `IntegrationEventHub` topic `"goal_proposals"`
- IL decides which goals to adopt (arbitration) and publishes `"goal_adopt"` events
- IL selects goals to pursue each tick and publishes `"goal_pursue"` events
- TaskGraph publishes `"task_completed"` events for IL to track progress
- Active goals are stored in `AstraState.active_goals`

**Code Changes:**
```python
# In goal_store.py

class GoalStore:
    def __init__(self, ..., event_hub: IntegrationEventHub):
        self.event_hub = event_hub
        # ... existing init

    def propose_goal(self, goal: GoalDefinition, proposer: str) -> str:
        """Override to publish proposal to IL."""
        goal_id = self._generate_goal_id()

        # Existing logic (store in DB)
        self._store_goal(goal_id, goal, GoalState.PROPOSED)

        # NEW: Publish to Integration Layer
        signal = GoalProposal(
            signal_id=str(uuid.uuid4()),
            source='goal_store',
            timestamp=datetime.now(),
            priority=Priority.NORMAL,
            goal=goal,
            rationale=f"Proposed by {proposer}",
            proposer=proposer
        )
        self.event_hub.publish('goal_proposals', signal)

        return goal_id
```

```python
# In task_graph.py

class TaskGraph:
    def __init__(self, ..., event_hub: IntegrationEventHub):
        self.event_hub = event_hub
        # ... existing init

    def mark_completed(self, task_id: str):
        """Override to publish completion to IL."""
        # Existing logic
        self.tasks[task_id].state = TaskState.SUCCEEDED

        # NEW: Publish to Integration Layer
        event = IntegrationEvent(
            signal_id=str(uuid.uuid4()),
            source='task_graph',
            timestamp=datetime.now(),
            priority=Priority.LOW,
            event_type='task_completed',
            payload={'task_id': task_id}
        )
        self.event_hub.publish('integration_events', event)
```

**Boundary:** Goal system remains autonomous for goal lifecycle management. IL arbitrates which goals to adopt/pursue but does NOT control task execution details.

### 5.4 PersonaService (LLM Orchestrator)

**Current Behavior:** Orchestrates prompt building, LLM generation, tool execution, and ingestion.

**Integration:**
- **Phase 1-2 (Migration):** PersonaService is still called directly by FastAPI `/api/chat` handler
  - IL just observes and provides context (focus, goals, self-model) to PersonaService
  - No change to existing chat path - maintains stability during rollout
- **Phase 3+ (Future):** IL can optionally decide whether to reply immediately, defer, or do other work first
  - PersonaService invoked by IL via `"action_selected"` events (type=`user_response`)
  - This is an experiment, not a requirement
- PersonaService reads `AstraState` to get current focus, goals, and self-model for prompt context
- PersonaService publishes `"response_generated"` events back to IL
- Budget tracking for LLM tokens is centralized in `BudgetManager`

**Code Changes:**
```python
# In persona_service.py

class PersonaService:
    def __init__(self, ..., integration_layer: IntegrationLayer, event_hub: IntegrationEventHub):
        self.integration_layer = integration_layer
        self.event_hub = event_hub
        # ... existing init

    async def generate_response(self, user_message: str) -> ChatResponse:
        """Override to use IL state for context."""
        # NEW: Read Global Workspace for context
        astra_state = self.integration_layer.get_state()
        top_focus = self.integration_layer.get_focus_top_k(5)
        self_model = astra_state.self_model

        # Build prompt with IL context
        prompt = self.prompt_builder.build_prompt(
            user_message=user_message,
            beliefs=self_model.core_beliefs + self_model.peripheral_beliefs,
            traits=self_model.traits,
            focus_items=top_focus,  # NEW: Use focus from IL
            active_goals=astra_state.active_goals,  # NEW: Use goals from IL
            emotional_state=astra_state.emotional_state
        )

        # Existing logic (LLM generation, tool execution, etc.)
        response = await self.llm_service.generate_with_tools(prompt)

        # NEW: Publish response event to IL
        event = IntegrationEvent(
            signal_id=str(uuid.uuid4()),
            source='persona_service',
            timestamp=datetime.now(),
            priority=Priority.LOW,
            event_type='response_generated',
            payload={'message': user_message, 'response': response}
        )
        self.event_hub.publish('integration_events', event)

        return response
```

**Boundary:** PersonaService retains full control over prompt building and LLM interaction. IL provides *context* (state snapshot) but does NOT control generation details.

### 5.5 Belief Gardener (Autonomous Maintenance)

**Current Behavior:** Background loop scans for belief formation/promotion/deprecation every 60min.

**Integration:**
- Belief gardener subscribes to `"belief_garden"` events from IL
- Instead of autonomous 60min loop, belief gardening is triggered by IL in MAINTENANCE mode
- Belief actions are subject to centralized budget in `BudgetManager`

**Code Changes:**
```python
# In belief_gardener.py

class BeliefGardener:
    def __init__(self, ..., event_hub: IntegrationEventHub, budget_manager: BudgetManager):
        self.event_hub = event_hub
        self.budget_manager = budget_manager
        # ... existing init

        # NEW: Subscribe to IL trigger events
        self.event_hub.subscribe('belief_garden', self.on_garden_triggered)

    async def on_garden_triggered(self, event: IntegrationEvent):
        """IL-triggered belief gardening (replaces autonomous loop)."""
        # Check budgets before scanning
        if not self.budget_manager.can_afford('form_belief'):
            logger.info("Belief gardening skipped: budget exhausted")
            return

        # Existing scan logic
        await self.scan_and_act()
```

**Migration:** The existing 60min autonomous loop can be removed in Phase 3 (see Migration Plan).

**Boundary:** Belief gardener retains logic for belief formation/promotion/deprecation. IL controls *when* gardening occurs and enforces budgets.

---

## 6. Implementation Guidance

### 6.1 Module Layout

```
src/
├── integration/
│   ├── __init__.py
│   ├── layer.py                 # IntegrationLayer class (thin orchestrator)
│   ├── state.py                 # AstraState, FocusItem, etc.
│   ├── signals.py               # Signal hierarchy
│   ├── identity_service.py      # IdentityService (PIM facade)
│   ├── budget_manager.py        # BudgetManager
│   ├── event_hub.py             # IntegrationEventHub
│   ├── focus.py                 # PURE FUNCTIONS: compute_salience, update_focus_stack, etc.
│   ├── arbitration.py           # PURE FUNCTIONS: detect_conflicts, resolve_conflicts, etc.
│   └── actions.py               # PURE FUNCTIONS: select_actions, etc.
├── services/
│   ├── awareness_loop.py        # Modified: publish percepts
│   ├── belief_consistency_checker.py  # Modified: publish dissonance
│   ├── belief_gardener.py       # Modified: subscribe to IL triggers
│   ├── goal_store.py            # Modified: publish proposals
│   ├── task_graph.py            # Modified: publish completions
│   ├── persona_service.py       # Modified: read IL state
│   └── ... (other services)
└── app.py                        # Modified: start IL at startup
```

**Lifecycle Management:** App startup must start BudgetManager and IntegrationLayer under the same event loop and ensure `stop()` is called for both on shutdown. The BudgetManager's reset loops run as background tasks that must be gracefully cancelled.

### 6.2 Dependency Injection

Use dependency injection to wire IL and subsystems without tight coupling.

```python
# In app.py

@app.on_event("startup")
async def startup():
    # Create shared dependencies
    event_hub = IntegrationEventHub()
    budget_manager = BudgetManager()

    # Create identity service (PIM facade)
    identity_service = IdentityService(
        belief_store=belief_store,
        persona_file_manager=persona_file_manager,
        awareness_loop=awareness_loop,
        identity_ledger=identity_ledger
    )

    # Create Integration Layer
    integration_layer = IntegrationLayer(
        identity_service=identity_service,
        budget_manager=budget_manager,
        event_hub=event_hub,
        mode=ExecutionMode.INTERACTIVE
    )

    # Wire subsystems to IL
    awareness_loop.event_hub = event_hub
    belief_consistency_checker.event_hub = event_hub
    belief_gardener.event_hub = event_hub
    belief_gardener.budget_manager = budget_manager
    goal_store.event_hub = event_hub
    task_graph.event_hub = event_hub
    persona_service.integration_layer = integration_layer
    persona_service.event_hub = event_hub

    # Start IL executive loop
    await integration_layer.start()
    await budget_manager.start()

    # Store IL in app state for FastAPI handlers
    app.state.integration_layer = integration_layer
```

### 6.3 FastAPI Handler Pattern

FastAPI handlers submit signals to IL rather than directly invoking subsystems.

```python
# In app.py

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    User chat endpoint.

    Instead of directly calling PersonaService, we:
    1. Submit USER_MESSAGE signal to IL
    2. IL decides when to generate response (next tick)
    3. Poll or subscribe for response

    For now, we maintain synchronous behavior for backwards compatibility.
    """
    # Get IL from app state
    il: IntegrationLayer = app.state.integration_layer

    # Submit user message as high-priority focus item
    focus_item = FocusItem(
        item_type=FocusType.USER_MESSAGE,
        item_id=str(uuid.uuid4()),
        content=request.message,
        salience=0.9,  # User messages are high priority
        entered_focus=datetime.now(),
        last_accessed=datetime.now(),
        access_count=1,
        decay_rate=0.01,
        min_salience_threshold=0.3
    )
    il.state.focus_stack.append(focus_item)

    # For now, invoke PersonaService directly (backwards compat)
    # In future phases, we'd wait for IL to dispatch response generation
    response = await persona_service.generate_response(request.message)

    return response
```

### 6.4 Testing Strategy

**Unit Tests:**
- Test each Integration Layer component in isolation
- Mock subsystems (awareness_loop, belief_system, etc.)
- **IMPORTANT:** Push logic into pure functions for testability
  - `focus.py`: Test `compute_salience()`, `update_focus_stack()` as pure functions
  - `arbitration.py`: Test `detect_conflicts()`, `resolve_conflicts()` as pure functions
  - `actions.py`: Test `select_actions()` as pure function taking AstraState
  - Keep `IntegrationLayer` thin - it should just orchestrate these pure functions
- Test signal collection, focus computation, arbitration logic independently

**Integration Tests:**
- Test IL with real subsystems (not mocks)
- Verify signal flow from subsystems to IL and back
- Test mode transitions (INTERACTIVE → AUTONOMOUS → MAINTENANCE)

**End-to-End Tests:**
- Simulate full user interaction with IL running
- Verify response times remain acceptable (< 2s for chat)
- Verify budget enforcement across ticks

**Example Unit Test:**
```python
# tests/integration/test_focus.py

def test_focus_stack_capacity():
    """Test that focus stack respects Miller's Law (max 7 items)."""
    il = IntegrationLayer(...)

    # Add 10 focus items
    for i in range(10):
        item = FocusItem(
            item_type=FocusType.MEMORY,
            item_id=f"mem_{i}",
            content=f"Memory {i}",
            salience=0.5,
            ...
        )
        il._add_to_focus(item)

    # Verify only 7 remain (least salient evicted)
    assert len(il.state.focus_stack) == 7
    assert il.state.focus_stack[0].salience >= 0.5  # Highest still present
```

### 6.5 Performance Considerations

**Latency Budget:**
- Executive Loop tick must complete within tick_interval (1000ms for INTERACTIVE)
- Phase budgets:
  - Collect signals: < 50ms
  - Update workspace: < 100ms
  - Compute focus: < 50ms
  - Detect conflicts: < 100ms
  - Apply budgets: < 10ms
  - Select actions: < 200ms
  - Dispatch actions: < 400ms (depends on LLM latency)
  - Persist snapshot: < 100ms

**Optimization Strategies:**
- Cache `IdentityService.get_snapshot()` for 1 second
- Use lazy evaluation for conflict detection (only if > 1 active goal)
- Batch Redis writes (persist every tick, but batch multiple keys)
- Run action dispatch in parallel (asyncio.gather)

**Degradation:**
If a tick exceeds interval:
1. Log warning
2. Skip non-critical phases (e.g., snapshot can be deferred)
3. Continue with next tick (no catch-up logic)

---

## 7. Migration Plan

The Integration Layer will be rolled out in 4 phases over an estimated timeline of 8-12 weeks.

### Phase 1: Read-Only Workspace and Signal Wiring (Weeks 1-3)

**Goal:** IL observes system without controlling it.

**Deliverables:**
- [ ] Implement `AstraState`, `SelfModelSnapshot`, `FocusItem` data structures
- [ ] Implement `IntegrationEventHub` (pub/sub)
- [ ] Implement `IdentityService` (PIM facade, read-only)
- [ ] Implement `BudgetManager` (tracking only, no enforcement yet)
- [ ] Wire subsystems to publish signals to EventHub:
  - awareness_loop → percepts
  - belief_consistency_checker → dissonance
  - goal_store → proposals
  - task_graph → completions
- [ ] Implement Executive Loop Phases 1-2 (collect signals, update workspace)
- [ ] Run IL in parallel with existing system (observability only)

**Success Criteria:**
- IL can reconstruct Global Workspace from subsystem signals
- No impact on existing behavior (IL is read-only)
- Snapshot persisted to Redis/JSON every 10 ticks

**Backwards Compatibility:**
- Subsystems operate exactly as before
- IL is a passive observer

### Phase 2: IL-Driven Introspection and Belief Operations (Weeks 4-6)

**Goal:** IL begins controlling introspection and belief gardening.

**Deliverables:**
- [ ] Implement Executive Loop Phases 3-5 (focus, conflicts, budgets)
- [ ] Implement `BudgetManager` enforcement (introspection, belief gardening)
- [ ] Modify `awareness_loop` to accept `trigger_introspection()` from IL
- [ ] **CRITICAL:** Disable awareness loop's internal introspection scheduler
  - **After Phase 2, there is exactly ONE controller of introspection: Integration Layer**
  - No dual control - prevents conflicts and budget violations
- [ ] Modify `belief_gardener` to subscribe to `"belief_garden"` events from IL
- [ ] Implement action dispatch for INTROSPECTION and BELIEF_GARDENING actions
- [ ] **Remove** autonomous 60min belief gardener loop (replaced by IL triggers)

**Success Criteria:**
- Introspection triggered by IL at optimal times (not fixed 30s interval)
- **Zero introspections occur outside IL control** - verify with logs
- Belief gardening triggered by IL in MAINTENANCE mode (not fixed 60min)
- Budgets enforced: no excess token usage or belief operations

**Backwards Compatibility:**
- Introspection behavior changes slightly (opportunistic vs. fixed interval)
- Belief gardening now IL-controlled (may happen less frequently)
- Document changes in CHANGELOG

### Phase 3: Full Arbitration Over Goals and Maintenance (Weeks 7-9)

**Goal:** IL arbitrates all goal adoption and task prioritization.

**Deliverables:**
- [ ] Implement Executive Loop Phases 6-7 (action selection, dispatch)
- [ ] Implement conflict detection and arbitration logic
- [ ] Modify `goal_store` to defer adoption until IL approves
- [ ] Implement action dispatch for GOAL_PURSUIT actions
- [ ] Integrate HTN planner with IL (IL triggers planning)
- [ ] Implement mode transitions (INTERACTIVE ↔ AUTONOMOUS ↔ MAINTENANCE)

**Success Criteria:**
- IL arbitrates goal conflicts (e.g., "explore" vs. "conserve resources")
- Multiple goals are pursued based on priority and budget
- Mode transitions work correctly (e.g., 30min inactivity → AUTONOMOUS)

**Backwards Compatibility:**
- Goal adoption is no longer immediate (requires IL approval)
- Document new goal lifecycle in docs

### Phase 4: Externalization and Observability (Weeks 10-12)

**Goal:** Expose IL state via API for debugging and monitoring.

**Deliverables:**
- [ ] Implement FastAPI endpoints:
  - `GET /api/integration/state` → Current AstraState
  - `GET /api/integration/focus` → Focus stack
  - `GET /api/integration/budgets` → Budget status
  - `POST /api/integration/mode` → Change mode
- [ ] Implement Prometheus metrics:
  - `astra_tick_duration_seconds`
  - `astra_focus_stack_size`
  - `astra_cognitive_load`
  - `astra_budget_tokens_available`
- [ ] Implement Grafana dashboards for IL monitoring
- [ ] Write comprehensive documentation (this spec + migration guide)

**Success Criteria:**
- Developers can inspect IL state in real-time
- Metrics are exported to Prometheus
- Dashboards visualize IL behavior

**Backwards Compatibility:**
- No breaking changes (purely additive)

### Rollback Strategy

Each phase has a rollback plan:
- **Phase 1:** Remove IL startup from app.py; subsystems revert to independent operation
- **Phase 2:** Re-enable autonomous belief gardener loop; remove IL triggers
- **Phase 3:** Re-enable immediate goal adoption in goal_store
- **Phase 4:** Remove new API endpoints; no system impact

---

## 8. Appendices

### A. Signal Type Reference

Complete taxonomy of signals:

| Signal Type | Source | Priority | Payload |
|------------|--------|----------|---------|
| `PerceptSignal` | awareness_loop | NORMAL | percept_type, content, novelty, entropy |
| `DissonanceSignal` | belief_consistency_checker | HIGH | pattern, belief_id, conflicting_memory, severity |
| `GoalProposal` | goal_store, user, belief_gardener | NORMAL | goal, rationale, proposer |
| `IntegrationEvent` | task_graph, goal_store, belief_system | LOW-HIGH | event_type, payload |
| `ActionSelectedEvent` | integration_layer | HIGH | action_type, target_id |

### B. Action Type Reference

Actions that IL can dispatch:

| Action Type | Target Subsystem | Estimated Cost | Priority |
|-------------|-----------------|----------------|----------|
| `USER_RESPONSE` | PersonaService | 500 tokens | HIGH (INTERACTIVE) |
| `GOAL_PURSUIT` | HTNPlanner, TaskExecutor | 400 tokens | NORMAL |
| `INTROSPECTION` | AwarenessLoop | 200 tokens | NORMAL (AUTONOMOUS) |
| `DISSONANCE_RESOLUTION` | BeliefConsistencyChecker | 300 tokens | HIGH |
| `BELIEF_GARDENING` | BeliefGardener | 0 tokens | LOW (MAINTENANCE) |

### C. Mapping to Iida's Paper (Section-by-Section)

**Note:** Using Iida's actual terminology. Astra's Integration Layer corresponds to **CIL (Cognitive Integration Layer)** - the central integrator and action selector.

| Iida Section | Astra Implementation | Notes |
|--------------|---------------------|-------|
| 3.2 CIL (Cognitive Integration Layer) | **Integration Layer (this spec)**, AstraState, Executive Loop | Central integration point - IL coordinates signals from PPL and IRL, manages focus, selects actions |
| 3.3 PPL (Pattern Prediction Layer) | Belief consistency checker, memory retrieval, embedding similarity, emotional reconciler | Predictive preprocessing - IL receives filtered/ranked signals from PPL |
| 3.4 IRL (Instinctive Response Layer) | Abort conditions, safety constraints, budget enforcement, rate limiters | Fast reflexive checks - IRL provides constraint signals to IL's CIL integration |
| 3.5 Layer Interactions | IntegrationEventHub, signal flow, pub/sub messaging | Explicit message passing between layers vs. Iida's implicit coordination |
| 4.1 AOM (Access-Oriented Memory) | raw_store (SQLite), vector_store (ChromaDB), persona files | Episodic/timestamped memory - directly corresponds to Iida's AOM |
| 4.2 PIM (Pattern-Integrated Memory) | IdentityService facade → beliefs, traits, anchors, self-knowledge | Semantic/generalized memory - IdentityService provides unified PIM view |
| 5.1 Role of Labeling in Self-Recognition | self_node in concept graph, identity anchors (origin + live) | Self-labeling through persistent identity anchors |
| 5.2 Formation of Self-Concept through Layer Interactions | SelfModelSnapshot synthesized each tick from all subsystems | IL (CIL) assembles coherent self-model from distributed sources |
| 5.3 Adaptive Self-Model Development | Identity ledger, belief gardener, dissonance resolution, anchor drift | Continuous identity evolution while maintaining origin reference |

### D. Glossary

**Iida's Terminology (from paper):**
- **CIL:** Cognitive Integration Layer - central integrator that coordinates PPL and IRL, manages focus, selects actions
- **PPL:** Pattern Prediction Layer - predictive preprocessing and filtering
- **IRL:** Instinctive Response Layer - fast reflexive responses and safety constraints
- **AOM:** Access-Oriented Memory - episodic/timestamped memory (events, conversations)
- **PIM:** Pattern-Integrated Memory - semantic/generalized memory (beliefs, knowledge, patterns)

**Astra's Components:**
- **AstraState:** The Global Workspace object representing Astra's current conscious state
- **Focus Stack:** Ordered list of attention-worthy items (max 7, Miller's Law)
- **IL (Integration Layer):** Astra's executive function that orchestrates subsystems - corresponds to CIL in Iida's model
- **IdentityService:** Astra's facade over fragmented identity stores - provides unified view analogous to PIM
- **Salience:** Measure of attention-worthiness (0.0-1.0)
- **Signal:** Normalized message from subsystem to IL
- **Tick:** One iteration of the Executive Loop

---

## Deliverables Checklist

✅ **1. Conceptual Architecture**
- Section 1: Integration Layer's role, what it owns/doesn't own, mapping to Iida's model
- Execution modes (INTERACTIVE, AUTONOMOUS, MAINTENANCE)

✅ **2. Interface and Contract Specification**
- Section 2: Core data structures (AstraState, SelfModelSnapshot, FocusItem, Signals, BudgetStatus)
- Section 3: Service interfaces (IntegrationLayer, IdentityService, BudgetManager, EventHub)
- Clear responsibility boundaries

✅ **3. Executive Loop Design**
- Section 4: Scheduling model, coexistence with FastAPI, tick frequencies
- Per-tick algorithm (8 phases)
- Detailed pseudo-code for all phases

✅ **4. Implementation and Migration Guidance**
- Section 6: Module layout, dependency injection, testing strategy, performance considerations
- Section 7: 4-phase migration plan with concrete milestones, rollback strategy
- Section 5: Integration points for each existing subsystem

---

## Deliberate Design Tradeoffs

1. **Orchestrator vs. Rewrite:** IL wraps existing subsystems rather than replacing them. *Tradeoff:* Less clean architecture, but lower risk and faster migration.

2. **Tick-based vs. Event-driven:** IL uses fixed tick interval rather than pure event-driven. *Tradeoff:* Predictable timing and simpler debugging, but less efficient (some ticks do minimal work).

3. **Focus Stack Size (7 items):** Hard limit based on Miller's Law. *Tradeoff:* May miss important low-salience items, but prevents attention overload.

4. **Budget Centralization:** All budgets managed by IL rather than distributed. *Tradeoff:* Single point of control (easier to reason about), but requires all subsystems to coordinate through IL.

5. **Identity Service Caching (1 second):** Short cache to balance consistency and performance. *Tradeoff:* Slightly stale self-model possible, but avoids redundant I/O.

---

**End of Specification**

This document provides a complete, implementation-ready specification for Astra's Integration Layer and Global Workspace. A senior engineer familiar with Astra's codebase can use this to implement the system with minimal ambiguity.
