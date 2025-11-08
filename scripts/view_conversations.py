#!/usr/bin/env python3
"""View conversation history with Astra."""

import sqlite3
import json
from datetime import datetime
import sys

def view_conversations(limit=50, search_term=None):
    """View recent conversations from the raw store.

    Args:
        limit: Number of messages to show (default 50)
        search_term: Optional search term to filter messages
    """
    conn = sqlite3.connect('data/raw_store.db')
    cursor = conn.cursor()

    # Build query
    query = '''
        SELECT id, created_at, content
        FROM experience
        WHERE type = 'occurrence' AND provenance LIKE '%"actor": "user"%'
    '''

    params = []
    if search_term:
        query += ' AND content LIKE ?'
        params.append(f'%{search_term}%')

    query += ' ORDER BY created_at DESC LIMIT ?'
    params.append(limit)

    cursor.execute(query, params)

    print(f'\n{"="*80}')
    print(f'Conversation History (showing {limit} most recent)')
    if search_term:
        print(f'Filtered by: "{search_term}"')
    print(f'{"="*80}\n')

    for row in cursor.fetchall():
        exp_id, created_at, content = row
        content_obj = json.loads(content)
        text = content_obj.get('text', '')

        # Extract prompt and response
        if 'Prompt: ' in text:
            parts = text.split('\n\nResponse:', 1)
            user_msg = parts[0].replace('Prompt: ', '')
            agent_resp = parts[1].strip() if len(parts) > 1 else ''

            # Format timestamp
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            time_str = dt.strftime('%Y-%m-%d %H:%M:%S')

            print(f'[{time_str}]')
            print(f'You:   {user_msg}')
            print(f'Astra: {agent_resp[:200]}...' if len(agent_resp) > 200 else f'Astra: {agent_resp}')
            print()

    conn.close()
    print(f'{"="*80}\n')


if __name__ == '__main__':
    limit = 50
    search_term = None

    # Parse command line args
    if len(sys.argv) > 1:
        if sys.argv[1].isdigit():
            limit = int(sys.argv[1])
            if len(sys.argv) > 2:
                search_term = ' '.join(sys.argv[2:])
        else:
            search_term = ' '.join(sys.argv[1:])

    view_conversations(limit, search_term)
