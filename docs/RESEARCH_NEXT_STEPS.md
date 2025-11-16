# Research System - Next Steps

**Date**: 2025-11-15
**Status**: Complete subsystem + QA lab ready, first run pending

## What's Built

### Complete Research Stack
1. **P0**: HTN task queue & execution with budget controls
2. **P1**: Session synthesis with automatic triggering
3. **Astra Integration**: `research_and_summarize` + `check_recent_research` tools
4. **Policy Layer**: When to call, how to speak from results, session reuse
5. **Presentation Layer**: Risk-calibrated formatting, provenance clustering
6. **Belief Updates**: Automatic kind classification (reinforce/contest_minor/informational)

### Complete QA Lab
1. **Tool Tracing**: PersonaService hooks capture tool usage, timing, metadata
2. **Benchmark Harness**: 4 diverse questions through Astra herself
3. **Automated Detector**: Flags policy violations (tool patterns, risk/hedging, current events)
4. **Manual Scoring**: 6-dimension framework (30 points per question)

---

## Next: Baseline Run (Your Plan)

### Step 1: Run & Freeze Artifact

```bash
# Fix missing config/__init__.py (already created)
source venv/bin/activate
cd /home/d/git/ai-exp
python3 src/test_research_benchmark_astra.py

# Then analyze
python -m src.analyze_benchmark_results data/research_benchmark_astra_results_*.json
```

**Freeze baseline**:
```bash
# Copy artifacts to dated directory
mkdir -p data/benchmarks/2025-11-15
cp data/research_benchmark_astra_results_*.json data/benchmarks/2025-11-15/
cp benchmark_analyzer_output.txt data/benchmarks/2025-11-15/

# Snapshot manual scores
# Fill in docs/RESEARCH_BENCHMARK_NOTES.md with scores for this run
```

---

### Step 2: Map Violations → P2 Tasks

For each violation type in analyzer output:

**MISSING_CHECK / WRONG_ORDER**:
- [ ] Tighten prompt: "Always call check_recent_research first for current events questions"
- Or: Hard-gate in code (PersonaService checks tool pattern before allowing research)

**HEDGING_MISMATCH**:
- [ ] Adjust risk classifier thresholds (research_formatter.py:_assess_risk_level)
- Or: Adjust formatter hedging templates per risk bucket (research_formatter.py:format_research_answer)

**No research on obvious current events**:
- [ ] Tweak persona-level rules in base_prompt.md (add more trigger phrases)
- Or: Add heuristic in PersonaService to detect current events patterns

**Example TODO entries**:
```markdown
- [ ] Fix MISSING_CHECK for Q2 by adding "Use check_recent_research BEFORE research_and_summarize" to tool description
- [ ] Fix HEDGING_MISMATCH for high risk by increasing contested_claims threshold from 3 to 2
```

---

### Step 3: Pick One Dimension, Lock It In

Choose exactly ONE must-pass dimension for next iteration:

**Option A**: "Tool policy compliance"
- For obviously current-events questions, Astra ALWAYS calls check_recent_research then research_and_summarize
- Fix: Update tool descriptions, add ordering hint to base_prompt.md

**Option B**: "Risk calibration"
- High risk → heavy hedging, medium risk → balanced, low risk → assertive
- Fix: Tune risk classifier thresholds, test hedging markers

**Implementation**:
1. Make ONLY the changes needed for that dimension
2. Re-run benchmark
3. Confirm that aspect is now clean
4. Move to next dimension

---

### Step 4: Regression Suite

Any time you change:
- Research prompts (research_htn_methods.py)
- Formatter (research_formatter.py)
- Policy text (base_prompt.md)

Run:
```bash
python3 src/test_research_benchmark_astra.py
python -m src.analyze_benchmark_results data/research_benchmark_astra_results_*.json
```

**Compare against baseline**:
- Tool policy compliance: same or better?
- Risk calibration: same or better?
- Presentation quality: same or better?

If regression detected → revert change or fix immediately.

---

## Files Ready to Run

**Harness** (executable):
- `src/test_research_benchmark_astra.py` - Main benchmark
- `src/analyze_benchmark_results.py` - Violation detector

**Scoring** (manual):
- `docs/RESEARCH_BENCHMARK_NOTES.md` - Scoring template

**Documentation**:
- `docs/RESEARCH_BENCHMARK_HARNESS.md` - Complete harness docs
- `docs/RESEARCH_POLICY_LAYER.md` - Policy layer docs
- `docs/RESEARCH_SYSTEM_COMPLETE.md` - Technical implementation docs

---

## Known Issues (First Run)

1. **Missing config/__init__.py**: Fixed - created file to expose get_settings()
2. **PYTHONPATH**: Use `cd /home/d/git/ai-exp && python3 src/...` instead of `PYTHONPATH=...`
3. **venv activation**: Need `source venv/bin/activate` before running

---

## What First Run Will Tell You

**If clean** (no violations):
- Policy layer works out of the box
- Astra follows guidance correctly
- Move to P2 (dedup, drift, source quality)

**If violations detected**:
- Map each to concrete patch
- Pick highest-impact dimension
- Fix, test, verify
- Repeat for next dimension

**Expected first-run patterns**:
- All 4 questions trigger research (they're all current/consensus)
- Tool ordering: check_recent_research → research_and_summarize
- Risk levels vary: Epstein (high), AI safety (medium), shutdown (medium-high), food (low-medium)
- Hedging should match risk

---

## Bottom Line

**Status**: Subsystem complete + QA lab ready

**Next**: Run baseline, analyze violations, map to P2 tasks

**Blocker**: None - harness is executable once venv activated and config fixed (both done)

**Time**: ~10-15 minutes for full benchmark run (4 questions × ~2-3 min each)

**Output**: JSON results + violation analysis → concrete P2 patch set

Ready to run when you are.
