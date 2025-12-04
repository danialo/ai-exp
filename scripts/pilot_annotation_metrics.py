#!/usr/bin/env python3
"""
Compute annotation agreement metrics from pilot annotation files.

Reads annotator JSON files and computes:
- Pairwise F1 on atom text (fuzzy match)
- Agreement on belief_type per atom
- Agreement on temporal_scope per atom
- Krippendorff's alpha

Usage:
    python scripts/pilot_annotation_metrics.py data/annotation_pilot/annotator_*.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from itertools import combinations

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class Atom:
    """Extracted belief atom from annotation."""
    atom_text: str
    belief_type: str
    polarity: str
    temporal_scope: str
    confidence: float


@dataclass
class AnnotatedExperience:
    """Single annotated experience."""
    experience_id: str
    text: str
    atoms: List[Atom]
    notes: str


@dataclass
class AnnotatorData:
    """All annotations from one annotator."""
    annotator_id: str
    experiences: Dict[str, AnnotatedExperience]


def load_annotator_file(file_path: Path) -> AnnotatorData:
    """Load annotations from a single annotator file."""
    with open(file_path) as f:
        data = json.load(f)

    # Derive annotator ID from filename
    annotator_id = file_path.stem.replace('annotator_', '').replace('annotation_pilot_', '')

    experiences = {}
    for entry in data:
        exp_id = entry['experience_id']
        atoms = []

        for atom_data in entry.get('expected_atoms', []):
            atoms.append(Atom(
                atom_text=atom_data.get('atom_text', ''),
                belief_type=atom_data.get('belief_type', 'UNKNOWN'),
                polarity=atom_data.get('polarity', 'affirm'),
                temporal_scope=atom_data.get('temporal_scope', 'ongoing'),
                confidence=atom_data.get('confidence', 1.0),
            ))

        experiences[exp_id] = AnnotatedExperience(
            experience_id=exp_id,
            text=entry.get('text', ''),
            atoms=atoms,
            notes=entry.get('notes', ''),
        )

    return AnnotatorData(annotator_id=annotator_id, experiences=experiences)


def levenshtein_ratio(s1: str, s2: str) -> float:
    """
    Compute Levenshtein similarity ratio between two strings.

    Returns value in [0, 1] where 1 = identical.
    """
    # Normalize strings
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()

    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Dynamic programming for edit distance
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )

    distance = dp[len1][len2]
    max_len = max(len1, len2)

    return 1.0 - (distance / max_len)


def find_best_match(atom: Atom, candidates: List[Atom], threshold: float = 0.7) -> Optional[Tuple[Atom, float]]:
    """Find best matching atom from candidates based on text similarity."""
    best_match = None
    best_score = 0.0

    for candidate in candidates:
        score = levenshtein_ratio(atom.atom_text, candidate.atom_text)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = candidate

    if best_match:
        return (best_match, best_score)
    return None


def compute_pairwise_f1(
    annotator1: AnnotatorData,
    annotator2: AnnotatorData,
    threshold: float = 0.7
) -> Dict[str, float]:
    """
    Compute pairwise F1 score on atom text between two annotators.

    Uses fuzzy matching with configurable threshold.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Find common experiences
    common_ids = set(annotator1.experiences.keys()) & set(annotator2.experiences.keys())

    for exp_id in common_ids:
        exp1 = annotator1.experiences[exp_id]
        exp2 = annotator2.experiences[exp_id]

        # Match atoms from annotator1 to annotator2
        matched_in_2 = set()
        for atom1 in exp1.atoms:
            match = find_best_match(atom1, exp2.atoms, threshold)
            if match:
                matched_atom, score = match
                total_tp += 1
                matched_in_2.add(id(matched_atom))
            else:
                total_fn += 1  # atom1 not found in annotator2

        # Count atoms in annotator2 not matched
        for atom2 in exp2.atoms:
            if id(atom2) not in matched_in_2:
                total_fp += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
    }


def compute_attribute_agreement(
    annotators: List[AnnotatorData],
    attribute: str,
    threshold: float = 0.7
) -> Dict[str, float]:
    """
    Compute agreement on a specific attribute (belief_type, temporal_scope, polarity).

    Only considers matched atoms (where text similarity >= threshold).
    """
    agreements = 0
    total_matches = 0

    for ann1, ann2 in combinations(annotators, 2):
        common_ids = set(ann1.experiences.keys()) & set(ann2.experiences.keys())

        for exp_id in common_ids:
            exp1 = ann1.experiences[exp_id]
            exp2 = ann2.experiences[exp_id]

            for atom1 in exp1.atoms:
                match = find_best_match(atom1, exp2.atoms, threshold)
                if match:
                    matched_atom, _ = match
                    total_matches += 1

                    val1 = getattr(atom1, attribute)
                    val2 = getattr(matched_atom, attribute)
                    if val1 == val2:
                        agreements += 1

    agreement_rate = agreements / total_matches if total_matches > 0 else 0.0

    return {
        'agreement_rate': agreement_rate,
        'agreements': agreements,
        'total_matches': total_matches,
    }


def compute_krippendorff_alpha(
    annotators: List[AnnotatorData],
    attribute: str,
    threshold: float = 0.7
) -> float:
    """
    Compute Krippendorff's alpha for a categorical attribute.

    This is a simplified implementation - for production use,
    consider using the `krippendorff` package.

    Returns alpha in [-1, 1] where:
    - 1 = perfect agreement
    - 0 = agreement expected by chance
    - negative = worse than chance
    """
    # Build reliability data matrix
    # Rows = items (matched atom pairs), Columns = annotators
    # Values = attribute values (encoded as integers)

    # First, collect all unique items (experience_id + normalized atom text)
    items = defaultdict(dict)  # item_key -> {annotator_id: value}

    value_to_int = {}
    int_counter = 0

    for ann in annotators:
        for exp_id, exp in ann.experiences.items():
            for atom in exp.atoms:
                # Create item key from experience and normalized atom text
                item_key = f"{exp_id}::{atom.atom_text.lower().strip()}"

                val = getattr(atom, attribute)
                if val not in value_to_int:
                    value_to_int[val] = int_counter
                    int_counter += 1

                items[item_key][ann.annotator_id] = value_to_int[val]

    # Filter to items with at least 2 annotators
    valid_items = {k: v for k, v in items.items() if len(v) >= 2}

    if len(valid_items) < 2:
        return 0.0  # Not enough data

    # Compute observed disagreement
    n_annotators = len(annotators)
    n_values = len(value_to_int)

    # Count value frequencies
    value_counts = defaultdict(int)
    total_codes = 0

    for item_key, annotator_vals in valid_items.items():
        for ann_id, val in annotator_vals.items():
            value_counts[val] += 1
            total_codes += 1

    # Observed disagreement
    observed_disagreement = 0.0
    n_items = len(valid_items)

    for item_key, annotator_vals in valid_items.items():
        vals = list(annotator_vals.values())
        n = len(vals)
        if n < 2:
            continue

        # Count disagreements within this item
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if vals[i] != vals[j]:
                    observed_disagreement += 1

        # Normalize by number of pairs
        n_pairs = n * (n - 1) / 2

    if n_items > 0:
        observed_disagreement /= n_items

    # Expected disagreement (assuming random assignment based on marginal distribution)
    expected_disagreement = 0.0
    for v1 in range(n_values):
        for v2 in range(n_values):
            if v1 != v2:
                p1 = value_counts[v1] / total_codes if total_codes > 0 else 0
                p2 = value_counts[v2] / total_codes if total_codes > 0 else 0
                expected_disagreement += p1 * p2

    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0

    alpha = 1.0 - (observed_disagreement / expected_disagreement)
    return alpha


def print_summary_table(metrics: Dict) -> None:
    """Print a formatted summary table of metrics."""
    print("\n" + "=" * 70)
    print("ANNOTATION AGREEMENT METRICS SUMMARY")
    print("=" * 70)

    # Pairwise F1 scores
    print("\n1. PAIRWISE F1 SCORES (Atom Text Matching)")
    print("-" * 50)
    print(f"{'Pair':<20} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 50)

    for pair, scores in metrics.get('pairwise_f1', {}).items():
        print(f"{pair:<20} {scores['precision']:.3f}{'':<8} {scores['recall']:.3f}{'':<8} {scores['f1']:.3f}")

    avg_f1 = sum(s['f1'] for s in metrics.get('pairwise_f1', {}).values()) / len(metrics.get('pairwise_f1', {})) if metrics.get('pairwise_f1') else 0
    print(f"{'Average':<20} {'':<12} {'':<12} {avg_f1:.3f}")

    # Attribute agreement
    print("\n2. ATTRIBUTE AGREEMENT (on matched atoms)")
    print("-" * 50)
    print(f"{'Attribute':<20} {'Agreement Rate':<15} {'Krippendorff α':<15}")
    print("-" * 50)

    for attr in ['belief_type', 'temporal_scope', 'polarity']:
        attr_metrics = metrics.get('attribute_agreement', {}).get(attr, {})
        agreement = attr_metrics.get('agreement_rate', 0)
        alpha = metrics.get('krippendorff_alpha', {}).get(attr, 0)
        print(f"{attr:<20} {agreement:.3f}{'':<11} {alpha:.3f}")

    # Overall assessment
    print("\n3. OVERALL ASSESSMENT")
    print("-" * 50)

    avg_f1_val = avg_f1
    avg_agreement = sum(
        metrics.get('attribute_agreement', {}).get(attr, {}).get('agreement_rate', 0)
        for attr in ['belief_type', 'temporal_scope', 'polarity']
    ) / 3

    if avg_f1_val >= 0.7 and avg_agreement >= 0.7:
        assessment = "GOOD - Ready for production annotation"
    elif avg_f1_val >= 0.5 and avg_agreement >= 0.5:
        assessment = "MODERATE - Consider refining guidelines"
    else:
        assessment = "LOW - Significant guideline revision needed"

    print(f"Average F1: {avg_f1_val:.3f}")
    print(f"Average Attribute Agreement: {avg_agreement:.3f}")
    print(f"Assessment: {assessment}")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/pilot_annotation_metrics.py <annotator_files...>")
        print("\nExample:")
        print("  python scripts/pilot_annotation_metrics.py data/annotation_pilot/annotator_*.json")
        sys.exit(1)

    # Load annotator files
    annotator_files = [Path(f) for f in sys.argv[1:]]

    print(f"Loading {len(annotator_files)} annotator files...")
    annotators = []
    for f in annotator_files:
        if not f.exists():
            print(f"  Warning: File not found: {f}")
            continue
        ann = load_annotator_file(f)
        print(f"  Loaded {ann.annotator_id}: {len(ann.experiences)} experiences")
        annotators.append(ann)

    if len(annotators) < 2:
        print("\nError: Need at least 2 annotators for agreement metrics")
        print("Creating mock annotator files for demonstration...")

        # Create mock data for testing
        mock_dir = Path("data/annotation_pilot")
        mock_dir.mkdir(parents=True, exist_ok=True)

        mock_data = [
            {
                "experience_id": "exp_001",
                "text": "I am a curious person who loves learning",
                "expected_atoms": [
                    {"atom_text": "I am curious", "belief_type": "TRAIT", "polarity": "affirm", "temporal_scope": "ongoing", "confidence": 0.9},
                    {"atom_text": "I love learning", "belief_type": "PREFERENCE", "polarity": "affirm", "temporal_scope": "ongoing", "confidence": 0.95},
                ],
                "notes": "",
            }
        ]

        mock_data_2 = [
            {
                "experience_id": "exp_001",
                "text": "I am a curious person who loves learning",
                "expected_atoms": [
                    {"atom_text": "I am a curious person", "belief_type": "TRAIT", "polarity": "affirm", "temporal_scope": "habitual", "confidence": 0.85},
                    {"atom_text": "I love learning", "belief_type": "VALUE", "polarity": "affirm", "temporal_scope": "ongoing", "confidence": 0.9},
                ],
                "notes": "",
            }
        ]

        mock_file_1 = mock_dir / "annotator_mock_1.json"
        mock_file_2 = mock_dir / "annotator_mock_2.json"

        with open(mock_file_1, 'w') as f:
            json.dump(mock_data, f, indent=2)
        with open(mock_file_2, 'w') as f:
            json.dump(mock_data_2, f, indent=2)

        print(f"  Created: {mock_file_1}")
        print(f"  Created: {mock_file_2}")

        # Reload
        annotators = [load_annotator_file(mock_file_1), load_annotator_file(mock_file_2)]

    # Compute metrics
    metrics = {
        'pairwise_f1': {},
        'attribute_agreement': {},
        'krippendorff_alpha': {},
    }

    # Pairwise F1
    for ann1, ann2 in combinations(annotators, 2):
        pair_name = f"{ann1.annotator_id} vs {ann2.annotator_id}"
        metrics['pairwise_f1'][pair_name] = compute_pairwise_f1(ann1, ann2)

    # Attribute agreement
    for attr in ['belief_type', 'temporal_scope', 'polarity']:
        metrics['attribute_agreement'][attr] = compute_attribute_agreement(annotators, attr)
        metrics['krippendorff_alpha'][attr] = compute_krippendorff_alpha(annotators, attr)

    # Print summary
    print_summary_table(metrics)

    # Save detailed metrics
    output_file = Path("data/annotation_pilot/metrics_report.json")
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nDetailed metrics saved to: {output_file}")


if __name__ == '__main__':
    main()
