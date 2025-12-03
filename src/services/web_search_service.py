"""Web search service using SerpAPI.

Provides search functionality with result parsing and error handling.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

try:
    from serpapi import GoogleSearch  # type: ignore
except Exception:  # pragma: no cover - handled for test environments without serpapi
    GoogleSearch = None

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str
    position: int
    timestamp: datetime


class WebSearchService:
    """Service for performing web searches via SerpAPI."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the search service.

        Args:
            api_key: SerpAPI key. If None, uses settings.SERP_API_KEY
        """
        self.api_key = api_key or settings.SERP_API_KEY
        if not self.api_key:
            logger.warning("No SERP_API_KEY configured - search functionality will be limited")
        if GoogleSearch is None:
            logger.warning("serpapi package is unavailable - WebSearchService disabled")

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Perform a web search.

        Args:
            query: Search query string
            num_results: Maximum number of results to return (default: 5)

        Returns:
            List of SearchResult objects

        Raises:
            ValueError: If API key is not configured
            RuntimeError: If search API returns an error
        """
        if not self.api_key:
            raise ValueError("SERP_API_KEY not configured. Set it in .env file.")
        if GoogleSearch is None:
            raise RuntimeError("serpapi.GoogleSearch is not available; install serpapi package")

        logger.info(f"Searching for: {query} (max results: {num_results})")

        try:
            search_params = {
                "q": query,
                "api_key": self.api_key,
                "num": num_results,
                "engine": "google",
            }

            search = GoogleSearch(search_params)
            results = search.get_dict()

            if "error" in results:
                error_msg = results["error"]
                logger.error(f"Search API error: {error_msg}")
                raise RuntimeError(f"Search failed: {error_msg}")

            # Parse organic results
            organic_results = results.get("organic_results", [])
            parsed_results = []

            for idx, result in enumerate(organic_results[:num_results]):
                search_result = SearchResult(
                    title=result.get("title", ""),
                    url=result.get("link", ""),
                    snippet=result.get("snippet", ""),
                    position=idx + 1,
                    timestamp=datetime.now(),
                )
                parsed_results.append(search_result)

            logger.info(f"Found {len(parsed_results)} results for query: {query}")
            return parsed_results

        except Exception as e:
            logger.error(f"Search error for query '{query}': {str(e)}")
            raise RuntimeError(f"Search failed: {str(e)}") from e

    def is_available(self) -> bool:
        """Check if search service is available.

        Returns:
            True if API key is configured
        """
        return self.api_key is not None


def create_web_search_service() -> WebSearchService:
    """Factory function to create WebSearchService instance.

    Returns:
        Configured WebSearchService instance
    """
    return WebSearchService()
