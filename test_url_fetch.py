#!/usr/bin/env python3
"""Quick test of URL fetcher with new headers."""

import sys
sys.path.insert(0, '/home/d/git/ai-exp')

from src.services.url_fetcher_service import URLFetcherService

def test_fetch():
    """Test fetching a URL that was previously blocked."""
    fetcher = URLFetcherService(headless=True)

    # Test a URL that was getting 403 before
    test_url = "https://bambulab.com/en-us/compare"

    print(f"Testing URL fetch: {test_url}")
    result = fetcher.fetch_url(test_url)

    print(f"\nSuccess: {result.success}")
    print(f"Title: {result.title}")
    print(f"Content length: {len(result.main_content)} chars")

    if result.success:
        print(f"\nFirst 500 chars of content:")
        print(result.main_content[:500])
    else:
        print(f"\nError: {result.error_message}")

    fetcher.close()

if __name__ == "__main__":
    test_fetch()
