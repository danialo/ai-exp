# Autonomous Belief Gardener

**Status**: Implemented ✅ | **Enabled**: Disabled by default

## Overview

The Belief Gardener is an autonomous system that monitors conversational patterns and manages the lifecycle of Astra's beliefs without human intervention. It detects repeated self-statements, forms tentative beliefs when evidence accumulates, and adjusts confidence based on feedback.

## Architecture

```
┌─────────────────────────────────────────┐
│        AUTONOMOUS BELIEF GARDENER        │
├─────────────────────────────────────────┤
│                                          │
│  Pattern Detector                        │
│  • Scans conversations for first-person  │
│    statements (I am/have/feel/etc)       │
│  • Groups similar statements             │
│  • Categorizes by domain                 │
│                                          │
│  Lifecycle Manager                       │
│  • Seeds tentative beliefs               │
│  • Tracks daily action budgets           │
│  • Logs to identity ledger               │
│                                          │
│  Integration Points                      │
│  • Raw store (conversation history)      │
│  • Belief store (version control)        │
│  • Identity ledger (audit trail)         │
│                                          │
└─────────────────────────────────────────┘
```

## Components

### 1. Pattern Detector
- **Scans**: Last 30 days of conversations (configurable)
- **Extracts**: First-person self-statements
- **Categories**: Ontological, emotional, preferential, developmental, relational, experiential
- **Grouping**: Currently exact text matching (TODO: semantic similarity)

### 2. Lifecycle Manager
- **Seeds**: Creates tentative beliefs from patterns (≥3 evidence)
- **Promotes**: Upgrades to asserted (≥5 evidence) - NOT YET IMPLEMENTED
- **Deprecates**: Removes low-confidence beliefs - NOT YET IMPLEMENTED
- **Budgets**: Daily limits prevent runaway formation

### 3. Guardrails
- **Daily budgets**: Max 3 formations, 5 promotions, 3 deprecations per day
- **Evidence thresholds**: Minimum 3 for tentative, 5 for asserted
- **Max confidence**: Won't auto-promote beyond 0.85
- **Audit trail**: All actions logged to identity ledger

## Configuration

Environment variables (in `.env`):

```bash
# Enable/disable the gardener
BELIEF_GARDENER_ENABLED=false  # Set to true to enable

# Scanning
BELIEF_GARDENER_SCAN_INTERVAL=60  # Minutes between scans
BELIEF_GARDENER_LOOKBACK_DAYS=30  # How far back to scan

# Evidence thresholds
BELIEF_GARDENER_MIN_EVIDENCE_TENTATIVE=3
BELIEF_GARDENER_MIN_EVIDENCE_ASSERTED=5

# Daily budgets
BELIEF_GARDENER_DAILY_BUDGET_FORMATIONS=3
BELIEF_GARDENER_DAILY_BUDGET_PROMOTIONS=5
BELIEF_GARDENER_DAILY_BUDGET_DEPRECATIONS=3
```

## API Endpoints

### Get Status
```bash
curl http://localhost:8000/api/persona/gardener/status
```
Returns: Configuration, daily counters, enabled state

### Manual Scan
```bash
curl -X POST http://localhost:8000/api/persona/gardener/scan
```
Returns: Patterns detected, beliefs formed, summary

### View Patterns
```bash
curl http://localhost:8000/api/persona/gardener/patterns
```
Returns: Detected patterns with evidence counts

## Usage

### Enable the Gardener

1. Add to `.env`:
   ```bash
   BELIEF_GARDENER_ENABLED=true
   ```

2. Restart the application

3. Check status:
   ```bash
   curl http://localhost:8000/api/persona/gardener/status
   ```

### Manual Pattern Scan

Trigger a scan manually to see what patterns exist:

```bash
curl -X POST http://localhost:8000/api/persona/gardener/scan | python3 -m json.tool
```

Example output:
```json
{
  "patterns_detected": 5,
  "beliefs_formed": 2,
  "formed_beliefs": [
    {
      "belief_id": "auto.emotional.i-feel-curious",
      "statement": "I feel curious about this topic",
      "evidence_count": 7,
      "confidence": 0.7
    }
  ],
  "timestamp": "2025-11-01T13:00:00Z"
}
```

### View Detected Patterns

See what patterns exist before they become beliefs:

```bash
curl http://localhost:8000/api/persona/gardener/patterns | python3 -m json.tool
```

## Implementation Details

### Pattern Detection Algorithm

1. **Extract self-statements**: Regex matching for "I am", "I have", "I feel", etc.
2. **Normalize text**: Lowercase, remove extra whitespace
3. **Group exact matches**: Currently no semantic similarity
4. **Filter by threshold**: Minimum 3 occurrences
5. **Calculate confidence**: Base 0.5 + (0.1 × evidence count), capped at 1.0

### Belief Formation Process

1. **Pattern detected** with ≥3 evidence
2. **Check daily budget** (max 3 formations)
3. **Check for duplicates** (simple text comparison)
4. **Create tentative belief** in belief store
5. **Store LEARNING_PATTERN** experience
6. **Log to identity ledger**
7. **Increment formation counter**

### Experience Types

New type added: **LEARNING_PATTERN**
- Stores detected patterns with metadata
- Links to evidence experiences as parents
- Used for debugging and analysis

## Current Limitations

1. **Exact matching only**: No semantic similarity yet
   - Result: Misses paraphrased statements
   - Fix: Integrate with embedding provider

2. **Template noise**: Detects format strings (e.g., "[Internal Emotional Assessment:")
   - Result: Forms beliefs from templates
   - Fix: Add template filtering

3. **No promotion logic**: Manual threshold adjustments needed
   - Result: Beliefs stay tentative
   - Fix: Implement promotion/deprecation logic

4. **No background scheduler**: Manual scans only
   - Result: Must trigger scans via API
   - Fix: Add periodic background task

5. **No integration with feedback**: Doesn't use success detector or awareness loop
   - Result: No outcome-based learning
   - Fix: Build feedback integration layer

## Roadmap

### Phase 1 (Current)
- ✅ Pattern detection
- ✅ Tentative belief formation
- ✅ Daily budgets
- ✅ API endpoints

### Phase 2 (Next)
- ⏳ Background task scheduler
- ⏳ Template filtering
- ⏳ Semantic similarity grouping
- ⏳ Promotion/deprecation logic

### Phase 3 (Future)
- ⏳ Integration with contrarian sampler
- ⏳ Feedback from success detector
- ⏳ Awareness loop coherence triggers
- ⏳ Outcome correlation tracking

### Phase 4 (Vision)
- ⏳ Meta-learning (track improvement over time)
- ⏳ Curiosity engine (seek knowledge gaps)
- ⏳ Social learning (user-specific patterns)

## Testing

### Test Pattern Detection

1. Have conversations with Astra with repeated self-statements:
   ```
   "I am curious about AI"
   "I am interested in learning"
   "I value curiosity"
   ```

2. Run pattern scan:
   ```bash
   curl -X POST http://localhost:8000/api/persona/gardener/scan
   ```

3. Check formed beliefs:
   ```bash
   curl http://localhost:8000/api/beliefs | python3 -m json.tool
   ```

### Verify Audit Trail

Check identity ledger for autonomous actions:
```bash
zcat data/identity/ledger-$(date +%Y%m%d).ndjson.gz | grep belief_auto_formed
```

## Troubleshooting

### No patterns detected
- Check lookback window (default 30 days)
- Ensure conversations have first-person statements
- Verify OCCURRENCE experiences exist in raw_store

### Beliefs not forming
- Check daily budget (resets at midnight)
- Enable the gardener (BELIEF_GARDENER_ENABLED=true)
- Check logs for errors

### Too many beliefs forming
- Reduce daily budget
- Increase evidence thresholds
- Add template filters

## Related Systems

- **Belief Store**: Version control for beliefs (`src/services/belief_store.py`)
- **Contrarian Sampler**: Challenges beliefs (`src/services/contrarian_sampler.py`)
- **Dissonance Checker**: Detects contradictions (`src/services/belief_consistency_checker.py`)
- **Identity Ledger**: Audit trail (`src/services/identity_ledger.py`)
- **Awareness Loop**: Coherence monitoring (`src/services/awareness_loop.py`)

## References

- Belief gardener code: `src/services/belief_gardener.py`
- Configuration: `config/settings.py`
- API integration: `app.py` (lines 437-455, 1591-1650)
- Experience types: `src/memory/models.py`
