#!/usr/bin/env python3
"""Quick test to verify outcome-driven trust system initializes correctly."""

import sys
from pathlib import Path

# Test imports
try:
    from src.services.provenance_trust import create_provenance_trust, TrustConfig
    from src.services.outcome_evaluator import create_outcome_evaluator, OutcomeConfig
    from src.services.feedback_aggregator_enhanced import EnhancedFeedbackAggregator, FeedbackConfig
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test initialization
try:
    data_dir = Path("./data")

    # 1. ProvenanceTrust
    trust_config = TrustConfig(enabled=True, alpha_0=0.3, k_samples=50)
    provenance_trust = create_provenance_trust(data_dir, trust_config)
    print(f"✅ ProvenanceTrust initialized: T_user={provenance_trust.get_trust('user'):.3f}")

    # 2. OutcomeEvaluator (without awareness_loop for now)
    outcome_config = OutcomeConfig(enabled=True)
    outcome_evaluator = create_outcome_evaluator(
        provenance_trust=provenance_trust,
        awareness_loop=None,
        belief_store=None,
        raw_store=None,
        config=outcome_config
    )
    print(f"✅ OutcomeEvaluator initialized: pending={len(outcome_evaluator.pending_evals)}")

    # 3. Test trust update
    provenance_trust.update_trust("user", reward=0.5)
    new_trust = provenance_trust.get_trust("user")
    print(f"✅ Trust update works: T_user={new_trust:.3f} (after r=0.5)")

    # 4. Test telemetry
    telemetry = provenance_trust.get_telemetry()
    print(f"✅ Telemetry: {len(telemetry['actors'])} actors tracked")

    print("\n🎉 All components initialized successfully!")
    print("\nNext steps:")
    print("1. Start the application: uvicorn app:app --reload")
    print("2. Check trust status: curl http://localhost:8000/api/persona/trust/status")
    print("3. Check gardener status: curl http://localhost:8000/api/persona/gardener/status")

except Exception as e:
    print(f"❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
