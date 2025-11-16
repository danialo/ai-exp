#!/usr/bin/env python3
"""Clean up broken research sessions with zero documents."""

import sqlite3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path("data/core.db")


def get_conn():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def find_broken_session_ids(conn):
    """Find sessions with anchors but no source docs."""
    cur = conn.cursor()

    # Sessions with anchors but no source docs
    cur.execute("""
        SELECT DISTINCT a.session_id
        FROM research_anchors a
        LEFT JOIN source_docs d ON d.session_id = a.session_id
        WHERE d.session_id IS NULL
    """)

    broken_with_anchors = {row[0] for row in cur.fetchall()}

    # Sessions marked completed but with zero docs
    cur.execute("""
        SELECT s.id
        FROM research_sessions s
        LEFT JOIN source_docs d ON d.session_id = s.id
        WHERE s.status = 'completed'
        GROUP BY s.id
        HAVING COUNT(d.id) = 0
    """)

    broken_completed = {row[0] for row in cur.fetchall()}

    return broken_with_anchors | broken_completed


def delete_broken_sessions(conn, broken_ids):
    """Delete broken sessions and related data."""
    if not broken_ids:
        return 0

    q_marks = ",".join("?" for _ in broken_ids)
    cur = conn.cursor()

    counts = {}

    # Delete anchors
    cur.execute(f"""
        DELETE FROM research_anchors
        WHERE session_id IN ({q_marks})
    """, tuple(broken_ids))
    counts['anchors'] = cur.rowcount

    # Delete belief updates tied to these sessions
    try:
        cur.execute(f"""
            DELETE FROM belief_updates
            WHERE session_id IN ({q_marks})
        """, tuple(broken_ids))
        counts['belief_updates'] = cur.rowcount
    except sqlite3.OperationalError:
        counts['belief_updates'] = 0

    # Delete source docs (defensive, should be none)
    cur.execute(f"""
        DELETE FROM source_docs
        WHERE session_id IN ({q_marks})
    """, tuple(broken_ids))
    counts['source_docs'] = cur.rowcount

    # Delete HTN tasks (if table exists)
    try:
        cur.execute(f"""
            DELETE FROM research_htn_tasks
            WHERE session_id IN ({q_marks})
        """, tuple(broken_ids))
        counts['htn_tasks'] = cur.rowcount
    except sqlite3.OperationalError:
        counts['htn_tasks'] = 0

    # Delete sessions
    cur.execute(f"""
        DELETE FROM research_sessions
        WHERE id IN ({q_marks})
    """, tuple(broken_ids))
    counts['sessions'] = cur.rowcount

    conn.commit()
    return counts


def main():
    """Main cleanup routine."""
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return 1

    conn = get_conn()

    try:
        broken_ids = find_broken_session_ids(conn)

        if not broken_ids:
            print("✓ No broken research sessions found.")
            return 0

        print(f"Found {len(broken_ids)} broken sessions:")
        for session_id in sorted(broken_ids):
            print(f"  - {session_id}")

        counts = delete_broken_sessions(conn, broken_ids)

        print(f"\nDeleted:")
        print(f"  - {counts['sessions']} sessions")
        print(f"  - {counts['anchors']} anchors")
        print(f"  - {counts['belief_updates']} belief updates")
        print(f"  - {counts['source_docs']} source docs")
        print(f"  - {counts['htn_tasks']} HTN tasks")
        print("\n✓ Cleanup complete.")

        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
