#!/usr/bin/env python3
"""Show recent self_definitions and their HTN decompositions."""

import sqlite3
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Show belief decomposition pipeline')
    parser.add_argument('-n', '--limit', type=int, default=10, help='Number of self_defs to show')
    parser.add_argument('--all', action='store_true', help='Show all, not just recent')
    args = parser.parse_args()

    db_path = Path(__file__).parent.parent / 'data' / 'raw_store.db'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        SELECT e.id, e.created_at, json_extract(e.content, '$.text') as text
        FROM experience e
        WHERE e.type = 'self_definition'
        ORDER BY e.created_at DESC
        LIMIT ?
    """, (args.limit,))

    for sd_id, created, text in c.fetchall():
        time = created[11:19]

        c.execute("""
            SELECT bn.canonical_text, bn.belief_type, bn.polarity
            FROM belief_occurrences bo
            JOIN belief_nodes bn ON bo.belief_id = bn.belief_id
            WHERE bo.source_experience_id = ?
        """, (sd_id,))
        atoms = c.fetchall()

        status = f"[{len(atoms)} atoms]" if atoms else "[NOT DECOMPOSED]"
        print(f"[{time}] {status}")
        print(f"  \"{text[:80]}{'...' if len(text or '') > 80 else ''}\"")

        for atext, atype, pol in atoms:
            p = "+" if pol == "affirm" else "-"
            print(f"    → [{atype}] {p} \"{atext}\"")
        print()

    conn.close()

if __name__ == '__main__':
    main()
