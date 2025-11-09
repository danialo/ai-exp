# Autonomous Goal Generation - Design Specification

**Status**: Design Phase
**Date**: 2025-11-09
**Branch**: TBD (new feature branch)

## Overview

Enable Astra to autonomously generate goals based on observed patterns, creating a collaborative goal-driven system where both user and system contribute goals to a unified prioritization pipeline.

## Goals

1. **Complement** user-created goals (not replace)
2. **Proactive** maintenance and improvement
3. **Safe** goal proposals (align with beliefs, safety checks)
4. **Transparent** (user can see what system wants to do)
5. **Adaptive** (learn what types of goals are valuable)

## Architecture

### High-Level Flow

```
Pattern Detection (observations)
    ↓
Impact Assessment (is this worth a goal?)
    ↓
Goal Proposal Generation
    ↓
Safety & Alignment Check
    ↓
Create Goal (state=PROPOSED, source=SYSTEM)
    ↓
User Review (optional)
    ↓
Adoption & Execution (same pipeline as user goals)
```

### Components

#### 1. GoalGenerator Service

**File**: `src/services/goal_generator.py`

```python
class GoalGenerator:
    """
    Autonomous goal generation from observed patterns.

    Detects opportunities for improvement and generates goal proposals.
    Goals start in PROPOSED state and compete with user goals in
    unified prioritization.
    """

    def __init__(
        self,
        goal_store: GoalStore,
        belief_store: BeliefStore,
        pattern_detectors: List[PatternDetector],
        min_confidence: float = 0.7
    ):
        pass

    async def scan_for_opportunities(self) -> List[GoalProposal]:
        """Scan all pattern detectors for goal opportunities."""
        pass

    def evaluate_proposal(self, proposal: GoalProposal) -> Optional[GoalDefinition]:
        """Assess if proposal should become a goal."""
        pass

    def create_system_goal(self, proposal: GoalProposal) -> GoalDefinition:
        """Create goal with source=SYSTEM."""
        pass
```

#### 2. Pattern Detectors (Pluggable)

Each detector observes a specific aspect and proposes goals:

**a) TestCoverageDetector**
- Monitors test coverage metrics
- Triggers when coverage drops below threshold
- Proposes: "Improve test coverage for module X"

**b) DocumentationStalenessDetector**
- Checks for outdated documentation
- Compares code changes to doc updates
- Proposes: "Update documentation for service Y"

**c) ComplexityDetector**
- Monitors code complexity metrics
- Detects complexity growth
- Proposes: "Refactor high-complexity module Z"

**d) BeliefCoherenceDetector**
- Watches coherence drops from awareness loop
- Detects belief contradictions
- Proposes: "Resolve belief contradiction in category X"

**e) FeedbackPatternDetector**
- Analyzes user feedback trends
- Detects repeated issues
- Proposes: "Address user concern about feature Y"

**f) TaskFailureDetector**
- Monitors task execution failures
- Detects recurring failures
- Proposes: "Fix failing task: task_name"

**g) DependencyUpdateDetector**
- Checks for outdated dependencies
- Security vulnerabilities
- Proposes: "Update dependency X (security)"

#### 3. Goal Proposal Data Structure

```python
@dataclass
class GoalProposal:
    """A proposed goal from pattern detection."""

    # What
    text: str  # "Improve test coverage for auth module"
    category: GoalCategory

    # Why
    pattern_detected: str  # "test_coverage_drop"
    evidence: Dict[str, Any]  # {"old": 0.85, "new": 0.62, "module": "auth"}
    confidence: float  # How confident detector is (0-1)

    # Priority estimation
    estimated_value: float  # How valuable is this?
    estimated_effort: float  # How much work?
    estimated_risk: float  # How risky?

    # Alignment
    aligns_with: List[str]  # Belief IDs
    contradicts: List[str]  # Belief IDs

    # Metadata
    detector_name: str
    detected_at: datetime
    expires_at: Optional[datetime]  # Proposal expiry
```

#### 4. Pattern Detector Interface

```python
class PatternDetector(ABC):
    """Base class for pattern detectors."""

    @abstractmethod
    async def detect(self) -> List[GoalProposal]:
        """Scan for patterns and return goal proposals."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Detector identifier."""
        pass

    @abstractmethod
    def scan_interval_minutes(self) -> int:
        """How often to run this detector."""
        pass
```

### Goal Source Tracking

**Enhancement to GoalDefinition**:

```python
class GoalSource(str, Enum):
    USER = "user"
    SYSTEM = "system"
    COLLABORATIVE = "collaborative"  # User + system refinement

@dataclass
class GoalDefinition:
    ...
    source: GoalSource = GoalSource.USER
    created_by: Optional[str] = None  # user_id or detector_name
    proposal_id: Optional[str] = None  # Link to original proposal
    auto_approved: bool = False  # True if created without review
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
- [ ] Create GoalGenerator service
- [ ] Add PatternDetector base class
- [ ] Enhance GoalDefinition with source tracking
- [ ] Add database migrations for new fields
- [ ] Wire into application lifecycle

### Phase 2: Initial Detectors (Week 2)
- [ ] TestCoverageDetector
- [ ] DocumentationStalenessDetector
- [ ] TaskFailureDetector
- [ ] Unit tests for each detector

### Phase 3: Safety & Alignment (Week 3)
- [ ] Proposal evaluation logic
- [ ] Belief alignment checking
- [ ] Duplicate detection (don't create redundant goals)
- [ ] Rate limiting (max N system goals per day)
- [ ] Integration tests

### Phase 4: API & UI (Week 4)
- [ ] GET /v1/goals/proposals - View pending proposals
- [ ] POST /v1/goals/proposals/{id}/approve - Approve proposal
- [ ] POST /v1/goals/proposals/{id}/reject - Reject proposal
- [ ] GET /v1/goals/system - Filter system-generated goals
- [ ] Dashboard endpoint for goal sources

## Safety Mechanisms

### 1. Alignment Checks
- All system goals must align with at least one active belief
- No system goals that contradict core beliefs
- Belief alignment score must be > 0.5

### 2. Rate Limiting
- Max 10 system-generated goals per day
- Max 3 goals per detector per day
- Cooldown period between similar goals (24h)

### 3. Proposal Review
- System goals start as PROPOSED
- Require explicit adoption (or auto-approve if safe)
- User can reject with feedback → detector learns

### 4. Expiration
- Proposals expire after 7 days if not adopted
- Prevents stale goal accumulation
- Can be re-proposed if pattern still detected

### 5. Confidence Threshold
- Minimum confidence 0.7 to create proposal
- Low confidence → log but don't create goal
- Adaptive threshold based on acceptance rate

## Decision Framework Integration

**New Decision Point**: `goal_generated`

```python
register_decision(
    decision_id="goal_generated",
    subsystem="goal_generator",
    description="Autonomous goal generation from pattern detection",
    parameters={
        "min_confidence": Parameter(...),
        "value_weight": Parameter(...),
        "belief_alignment_threshold": Parameter(...)
    },
    success_metrics=["goal_adopted", "goal_completed", "outcome_positive"],
    context_features=["pattern_type", "evidence_strength", "belief_alignment"]
)
```

**Learning**:
- Which patterns lead to adopted goals?
- Which detectors produce valuable goals?
- Optimal confidence thresholds per detector

## Example Workflow

### Scenario: Test Coverage Drop

1. **Detection** (every 6 hours):
```python
# TestCoverageDetector.detect()
coverage_data = get_test_coverage()
if coverage_data["auth_module"] < 0.70:
    return GoalProposal(
        text="Improve test coverage for auth module",
        category=GoalCategory.MAINTENANCE,
        pattern_detected="test_coverage_drop",
        evidence={"module": "auth", "coverage": 0.62, "threshold": 0.70},
        confidence=0.85,
        estimated_value=0.8,
        estimated_effort=0.4,
        estimated_risk=0.2
    )
```

2. **Evaluation**:
```python
# GoalGenerator.evaluate_proposal()
- Check alignment: matches belief "code_quality_important" ✓
- Check duplicates: no existing goal for auth coverage ✓
- Check rate limit: only 2 system goals today ✓
- Confidence 0.85 > threshold 0.7 ✓
→ APPROVED
```

3. **Creation**:
```python
goal = GoalDefinition(
    text="Improve test coverage for auth module",
    category=GoalCategory.MAINTENANCE,
    value=0.8,
    effort=0.4,
    risk=0.2,
    source=GoalSource.SYSTEM,
    created_by="test_coverage_detector",
    state=GoalState.PROPOSED
)
goal_store.create_goal(goal)
```

4. **Prioritization**:
- System goal enters prioritization queue
- Competes with user goals using same weights
- May be selected if high priority

5. **Execution**:
- If selected: HTN planner decomposes → tasks execute
- Same execution pipeline as user goals

6. **Outcome Learning**:
- Goal completed → coverage improved → positive outcome
- ParameterAdapter learns: test_coverage_detector is valuable
- Future test coverage goals get higher confidence

## Metrics & Monitoring

### Goal Generation Metrics
- System goals created per day
- Proposals per detector
- Approval rate by detector
- Rejection rate by detector
- Average time to adoption

### Outcome Metrics
- System goal completion rate
- Success rate (outcome positive)
- User satisfaction with system goals
- Belief alignment correlation

### Learning Metrics
- Confidence threshold evolution
- Detector weights adaptation
- Pattern → outcome correlation

## API Examples

### View Proposals
```bash
GET /v1/goals/proposals
Response:
{
  "proposals": [
    {
      "id": "prop_123",
      "text": "Improve test coverage for auth module",
      "detector": "test_coverage_detector",
      "confidence": 0.85,
      "evidence": {"coverage": 0.62, "threshold": 0.70},
      "created_at": "2025-11-09T10:00:00Z",
      "expires_at": "2025-11-16T10:00:00Z"
    }
  ]
}
```

### Approve Proposal
```bash
POST /v1/goals/proposals/prop_123/approve
Response:
{
  "goal_id": "goal_auto_123",
  "state": "proposed",
  "source": "system",
  "created_by": "test_coverage_detector"
}
```

### Filter System Goals
```bash
GET /v1/goals?source=system&state=adopted
```

## Testing Strategy

### Unit Tests
- Each detector in isolation
- Proposal evaluation logic
- Safety checks
- Rate limiting

### Integration Tests
- End-to-end: detection → proposal → creation → adoption
- Mixed goal pools (user + system)
- Prioritization with both sources
- Duplicate detection

### E2E Tests
- Full workflow in production environment
- Monitor actual pattern detection
- Measure proposal quality
- User acceptance rates

## Future Enhancements

### Short-term
- Collaborative goals (user initiates, system refines)
- Goal templates (common patterns)
- Detector marketplace (add new detectors easily)

### Long-term
- Multi-step goal reasoning (goal → sub-goals)
- Goal negotiation (system proposes, user counters)
- Meta-learning (learn which detectors work best)
- External pattern sources (GitHub issues, Slack, logs)

## Success Criteria

**Phase 1 Success**:
- [ ] System generates at least 1 valid goal per week
- [ ] 70%+ of generated goals pass safety checks
- [ ] No goals that contradict core beliefs

**Phase 2 Success**:
- [ ] 50%+ approval rate for system proposals
- [ ] System goals contribute 20% of completed work
- [ ] Positive user feedback on system-generated goals

**Phase 3 Success**:
- [ ] System adapts detector thresholds based on outcomes
- [ ] 80%+ of system goals align with user priorities
- [ ] Measurable improvement in codebase health metrics

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Too many low-value goals | Rate limiting, confidence thresholds, user feedback |
| Goals conflict with user work | Prioritization ensures user goals can win, rejection feedback |
| Pattern detection too noisy | Adaptive confidence thresholds, detector weights |
| Goals misaligned with beliefs | Mandatory alignment check, belief scoring |
| System becomes too autonomous | Start conservative (PROPOSED state), user approval gates |

## Timeline

- **Week 1**: Core infrastructure + source tracking
- **Week 2**: Initial detectors (test coverage, docs, failures)
- **Week 3**: Safety, alignment, integration tests
- **Week 4**: API endpoints, monitoring, production deploy

**Total**: 4 weeks to production-ready autonomous goal generation

## Conclusion

This design creates a collaborative goal-driven system where:
- **User sets strategic direction** with explicit goals
- **System maintains tactical health** with autonomous goals
- **Both feed unified prioritization** using adaptive learning
- **Safety mechanisms** prevent runaway autonomy
- **Transparency** ensures user understands system intent

The result is a more resilient, self-maintaining AI system that complements human direction rather than replacing it.
