#!/usr/bin/env python3
"""
Export self_definition Experiences for manual annotation pilot.

Exports 20 experiences stratified by length for annotator labeling:
- 7 short (<50 chars)
- 7 medium (50-150 chars)
- 6 long (>150 chars)

Output format:
{
    "experience_id": "...",
    "text": "...",
    "expected_atoms": [],  // annotator fills this
    "notes": ""
}
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.raw_store import RawStore
from src.memory.models import ExperienceType


def get_self_definitions(raw_store: RawStore) -> list:
    """Get all self_definition experiences."""
    # Query experiences by type
    experiences = raw_store.query(
        types=[ExperienceType.SELF_DEFINITION.value],
        limit=10000  # Get all
    )
    return experiences


def stratify_by_length(experiences: list) -> dict:
    """
    Stratify experiences into short/medium/long buckets.

    Returns dict with:
    - short: experiences with text < 50 chars
    - medium: experiences with text 50-150 chars
    - long: experiences with text > 150 chars
    """
    short = []
    medium = []
    long = []

    for exp in experiences:
        text = exp.content.get('text', '') if hasattr(exp, 'content') else ''
        if isinstance(exp.content, dict):
            text = exp.content.get('text', '')
        else:
            text = str(exp.content)

        text_len = len(text)

        if text_len < 50:
            short.append(exp)
        elif text_len <= 150:
            medium.append(exp)
        else:
            long.append(exp)

    return {
        'short': short,
        'medium': medium,
        'long': long,
    }


def sample_experiences(stratified: dict, counts: dict, seed: int = 42) -> list:
    """
    Sample experiences from stratified buckets.

    Args:
        stratified: Dict with 'short', 'medium', 'long' lists
        counts: Dict with target counts per bucket
        seed: Random seed for reproducibility

    Returns:
        List of sampled experiences
    """
    random.seed(seed)
    sampled = []

    for bucket, count in counts.items():
        available = stratified.get(bucket, [])
        if len(available) <= count:
            sampled.extend(available)
        else:
            sampled.extend(random.sample(available, count))

    return sampled


def format_for_annotation(experiences: list) -> list:
    """
    Format experiences for annotation output.

    Output format per experience:
    {
        "experience_id": str,
        "text": str,
        "created_at": str (ISO format),
        "expected_atoms": [],  // annotator fills this
        "notes": ""
    }
    """
    output = []

    for exp in experiences:
        # Extract text from content
        if isinstance(exp.content, dict):
            text = exp.content.get('text', '')
        else:
            text = str(exp.content)

        entry = {
            "experience_id": exp.id,
            "text": text,
            "created_at": exp.created_at.isoformat() if hasattr(exp.created_at, 'isoformat') else str(exp.created_at),
            "expected_atoms": [
                # Example schema for annotators:
                # {
                #     "atom_text": "I [claim]",
                #     "belief_type": "TRAIT|PREFERENCE|VALUE|CAPABILITY_LIMIT|FEELING_STATE|META_BELIEF|RELATIONAL|BELIEF_ABOUT_SELF",
                #     "polarity": "affirm|deny",
                #     "temporal_scope": "state|ongoing|habitual|transitional|past|unknown",
                #     "confidence": 0.0-1.0
                # }
            ],
            "notes": "",
        }
        output.append(entry)

    return output


def main():
    """Main export function."""
    # Configuration
    output_dir = Path("data/annotation_pilot")
    output_dir.mkdir(parents=True, exist_ok=True)

    target_counts = {
        'short': 7,
        'medium': 7,
        'long': 6,
    }

    print("Loading self_definition experiences...")

    # Initialize raw store
    try:
        raw_store = RawStore()
    except Exception as e:
        print(f"Error initializing RawStore: {e}")
        print("Falling back to direct SQLite query...")

        # Fallback: direct SQLite query
        import sqlite3
        db_path = Path("data/raw_store.db")

        if not db_path.exists():
            print(f"Database not found: {db_path}")
            sys.exit(1)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, type, created_at, content
            FROM experience
            WHERE type = 'self_definition'
            ORDER BY created_at DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        # Convert to simple objects
        class SimpleExp:
            def __init__(self, row):
                self.id = row['id']
                self.type = row['type']
                self.created_at = row['created_at']
                try:
                    self.content = json.loads(row['content']) if row['content'] else {}
                except json.JSONDecodeError:
                    self.content = {'text': row['content']}

        experiences = [SimpleExp(r) for r in rows]
        print(f"Found {len(experiences)} self_definition experiences via direct query")

        # Stratify
        stratified = stratify_by_length(experiences)
        print(f"  Short (<50 chars): {len(stratified['short'])}")
        print(f"  Medium (50-150 chars): {len(stratified['medium'])}")
        print(f"  Long (>150 chars): {len(stratified['long'])}")

        # Sample
        sampled = sample_experiences(stratified, target_counts)
        print(f"\nSampled {len(sampled)} experiences for annotation")

        # Format output
        output = format_for_annotation(sampled)

        # Write output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"annotation_pilot_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nExported to: {output_file}")

        # Also create annotation schema reference
        schema_file = output_dir / "annotation_schema.md"
        schema_content = """# Annotation Schema for Self-Definition Experiences

## Expected Atoms Format

For each experience, annotators should extract atomic beliefs with the following fields:

### Required Fields

- **atom_text**: First-person atomic statement (must start with "I")
  - Example: "I value honesty"
  - Example: "I am introverted"

- **belief_type**: One of:
  - `TRAIT` - Personality characteristic (I am patient, I am creative)
  - `PREFERENCE` - Likes/dislikes (I prefer quiet, I enjoy reading)
  - `VALUE` - Core values (I value honesty, I believe in fairness)
  - `CAPABILITY_LIMIT` - Abilities/limitations (I can solve problems, I cannot sing)
  - `FEELING_STATE` - Emotional states (I feel anxious, I am happy)
  - `META_BELIEF` - Beliefs about beliefs (I think I know myself well)
  - `RELATIONAL` - About relationships (I am close to my family)
  - `BELIEF_ABOUT_SELF` - Self-perception (I am misunderstood)

- **polarity**: `affirm` or `deny`
  - affirm: positive assertion (I am, I like, I can)
  - deny: negative assertion (I am not, I don't like, I cannot)

- **temporal_scope**: One of:
  - `state` - Right now, at this moment
  - `ongoing` - Current but not permanent (default)
  - `habitual` - Repeated pattern (I always, I usually)
  - `transitional` - Changing state (I'm becoming, lately I)
  - `past` - Former state (I used to, I was)
  - `unknown` - Cannot determine

- **confidence**: 0.0-1.0
  - How confident the annotator is in this extraction
  - 1.0 = very certain this is correct
  - 0.5 = somewhat uncertain
  - < 0.3 = uncertain but included

## Notes Field

Use the notes field to:
- Flag ambiguous cases
- Note disagreements with schema
- Explain difficult extraction decisions
- Mark experiences that should be excluded

## Example Annotation

Input text: "I've always been a bit shy, but lately I'm becoming more confident. I really value authenticity."

Expected atoms:
```json
[
  {
    "atom_text": "I am shy",
    "belief_type": "TRAIT",
    "polarity": "affirm",
    "temporal_scope": "habitual",
    "confidence": 0.9
  },
  {
    "atom_text": "I am becoming more confident",
    "belief_type": "TRAIT",
    "polarity": "affirm",
    "temporal_scope": "transitional",
    "confidence": 0.95
  },
  {
    "atom_text": "I value authenticity",
    "belief_type": "VALUE",
    "polarity": "affirm",
    "temporal_scope": "ongoing",
    "confidence": 1.0
  }
]
```
"""
        with open(schema_file, 'w') as f:
            f.write(schema_content)

        print(f"Schema reference: {schema_file}")

        return

    # Normal path with RawStore
    experiences = get_self_definitions(raw_store)
    print(f"Found {len(experiences)} self_definition experiences")

    # Stratify
    stratified = stratify_by_length(experiences)
    print(f"  Short (<50 chars): {len(stratified['short'])}")
    print(f"  Medium (50-150 chars): {len(stratified['medium'])}")
    print(f"  Long (>150 chars): {len(stratified['long'])}")

    # Sample
    sampled = sample_experiences(stratified, target_counts)
    print(f"\nSampled {len(sampled)} experiences for annotation")

    # Format output
    output = format_for_annotation(sampled)

    # Write output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"annotation_pilot_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nExported to: {output_file}")


if __name__ == '__main__':
    main()
