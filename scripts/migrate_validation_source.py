#!/usr/bin/env python3
"""Migrate legacy SELF_DEFINITION experiences to include validation_source field.

This script patches experiences that were created by the claim extractor but lack
the validation_source metadata field. Only patches experiences that match the
extractor's payload shape.

Safety:
- Dry-run mode by default (use --apply to actually patch)
- Only patches SELF_DEFINITION experiences with extractor-shaped payloads
- Preserves data integrity by not patching non-extractor claims
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from sqlmodel import Session


def patch_validation_source(db_path: str, dry_run: bool = True):
    """Patch legacy SELF_DEFINITION experiences with validation_source.

    Args:
        db_path: Path to the SQLite database
        dry_run: If True, only report what would be patched
    """
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    with Session(engine) as session:
        # Find SELF_DEFINITION experiences without validation_source
        result = session.execute(
            text("""
                SELECT id, content
                FROM experience
                WHERE type = 'self_definition'
                AND json_extract(content, '$.structured.validation_source') IS NULL
            """)
        )

        experiences = result.fetchall()

        print(f"Found {len(experiences)} SELF_DEFINITION experiences without validation_source")

        if len(experiences) == 0:
            print("Nothing to patch.")
            return

        # Analyze payload shapes
        extractor_shaped = []
        non_extractor = []

        for exp_id, content_json in experiences:
            try:
                content = json.loads(content_json)
                structured = content.get("structured", {})

                # Heuristic: Extractor-shaped payloads have these fields:
                # Dissonance checker shape:
                #   - statement, source, confidence (string), context, source_experience_id, extracted_from_query
                # Ingestion pipeline shape:
                #   - trait_type, descriptor, topic, source_experience_id, source_prompt, source_response
                # Legacy extractor shape:
                #   - statement, trait_type, confidence (float), source

                # Accept all three shapes as legitimate claim extractor outputs
                is_dissonance_checker = (
                    "statement" in structured
                    and "source_experience_id" in structured
                    and "extracted_from_query" in structured
                )

                is_ingestion_pipeline = (
                    "trait_type" in structured
                    and "descriptor" in structured
                    and "source_experience_id" in structured
                )

                is_legacy_extractor = (
                    "statement" in structured
                    and "trait_type" in structured
                    and isinstance(structured.get("confidence"), (int, float))
                    and "descriptor" not in structured  # Distinguish from ingestion pipeline
                )

                is_extractor_shaped = is_dissonance_checker or is_ingestion_pipeline or is_legacy_extractor

                if is_extractor_shaped:
                    extractor_shaped.append((exp_id, content))
                else:
                    non_extractor.append((exp_id, content))

            except Exception as e:
                print(f"Warning: Failed to parse experience {exp_id}: {e}")
                non_extractor.append((exp_id, None))

        print(f"\nPayload analysis:")
        print(f"  Extractor-shaped: {len(extractor_shaped)}")
        print(f"  Non-extractor: {len(non_extractor)}")

        if non_extractor:
            print(f"\n⚠️  Warning: {len(non_extractor)} experiences don't match extractor shape")
            print("These will NOT be patched (safety measure)")
            print("\nSample non-extractor payloads:")
            for exp_id, content in non_extractor[:3]:
                if content:
                    print(f"\n  - {exp_id}:")
                    print(f"    structured keys: {list(content.get('structured', {}).keys())}")
                    print(f"    structured: {json.dumps(content.get('structured', {}), indent=6)[:300]}")

        if not extractor_shaped:
            print("\nNo extractor-shaped experiences to patch.")
            return

        if dry_run:
            print(f"\n🔍 DRY RUN: Would patch {len(extractor_shaped)} extractor-shaped experiences")
            print("Sample of what would be patched:")
            for exp_id, content in extractor_shaped[:3]:
                structured = content.get("structured", {})
                print(f"  - {exp_id}: {structured.get('statement', '')[:80]}...")
            print("\nRe-run with --apply to actually patch")
            return

        # Apply patches
        print(f"\n✏️  Patching {len(extractor_shaped)} experiences...")

        patched_count = 0
        for exp_id, content in extractor_shaped:
            # Add validation_source to structured
            content["structured"]["validation_source"] = "claim_extractor"

            # Update the experience
            session.execute(
                text("""
                    UPDATE experience
                    SET content = :content
                    WHERE id = :id
                """),
                {"id": exp_id, "content": json.dumps(content)}
            )
            patched_count += 1

        session.commit()
        print(f"✅ Successfully patched {patched_count} experiences")


def main():
    parser = argparse.ArgumentParser(description="Migrate validation_source field")
    parser.add_argument(
        "--db",
        default="data/raw_store.db",
        help="Path to SQLite database (default: data/raw_store.db)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply patches (default is dry-run)"
    )

    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Error: Database not found: {args.db}")
        sys.exit(1)

    patch_validation_source(args.db, dry_run=not args.apply)


if __name__ == "__main__":
    main()
