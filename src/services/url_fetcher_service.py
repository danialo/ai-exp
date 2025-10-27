"""URL fetching service using Playwright for browser automation.

Handles fetching web pages with full JavaScript support.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import asyncio

from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup

from config.settings import settings

logger = logging.getLogger(__name__)

# Thread pool for running sync Playwright in async context
_thread_pool = ThreadPoolExecutor(max_workers=3)


@dataclass
class FetchedContent:
    """Content fetched from a URL."""

    url: str
    title: str
    text_content: str
    main_content: str  # Extracted main content (article text, etc.)
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None


class URLFetcherService:
    """Service for fetching web page content using browser automation."""

    def __init__(
        self,
        headless: bool = True,
        timeout_ms: int = 30000,
    ):
        """Initialize the URL fetcher.

        Args:
            headless: Run browser in headless mode (default: True)
            timeout_ms: Page load timeout in milliseconds (default: 30000)
        """
        self.headless = headless if headless is not None else settings.BROWSER_HEADLESS
        self.timeout_ms = timeout_ms if timeout_ms is not None else settings.BROWSER_TIMEOUT_MS
        self._playwright = None
        self._browser: Optional[Browser] = None

    def _ensure_browser(self):
        """Ensure browser is initialized."""
        if self._browser is None:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            logger.info(f"Browser launched (headless={self.headless})")

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML.

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            Extracted main content text
        """
        # Try common article/main content selectors
        main_selectors = [
            "article",
            "main",
            '[role="main"]',
            ".article-content",
            ".post-content",
            ".entry-content",
            "#content",
        ]

        for selector in main_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                # Remove script, style, nav, footer
                for tag in main_element.find_all(["script", "style", "nav", "footer", "aside"]):
                    tag.decompose()
                text = main_element.get_text(separator="\n", strip=True)
                if len(text) > 100:  # Ensure meaningful content
                    return text

        # Fallback: return body text
        body = soup.find("body")
        if body:
            for tag in body.find_all(["script", "style", "nav", "footer", "aside", "header"]):
                tag.decompose()
            return body.get_text(separator="\n", strip=True)

        return ""

    def _fetch_url_simple(self, url: str, timestamp: datetime) -> FetchedContent:
        """Fetch URL using simple HTTP request (no JavaScript).

        Args:
            url: URL to fetch
            timestamp: Timestamp for the fetch

        Returns:
            FetchedContent object with page data
        """
        import requests

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; AstraBot/1.0)'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Get title
            title_tag = soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else url

            # Get text content
            text_content = soup.get_text(separator="\n", strip=True)

            # Extract main content
            main_content = self._extract_main_content(soup)

            # Limit content length
            max_length = settings.WEB_CONTENT_MAX_LENGTH
            if len(text_content) > max_length:
                text_content = text_content[:max_length] + "..."
            if len(main_content) > max_length:
                main_content = main_content[:max_length] + "..."

            logger.info(f"Successfully fetched {url} via HTTP - {len(text_content)} chars")

            return FetchedContent(
                url=url,
                title=title,
                text_content=text_content,
                main_content=main_content,
                timestamp=timestamp,
                success=True,
            )

        except requests.Timeout:
            error_msg = "Request timeout (>30s)"
            logger.warning(f"{error_msg} - {url}")
            return FetchedContent(
                url=url,
                title="",
                text_content="",
                main_content="",
                timestamp=timestamp,
                success=False,
                error_message=error_msg,
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error fetching {url} via HTTP: {error_msg}")
            return FetchedContent(
                url=url,
                title="",
                text_content="",
                main_content="",
                timestamp=timestamp,
                success=False,
                error_message=error_msg,
            )

    def _fetch_url_sync(self, url: str, timestamp: datetime) -> FetchedContent:
        """Internal synchronous fetch method using Playwright.

        Args:
            url: URL to fetch
            timestamp: Timestamp for the fetch

        Returns:
            FetchedContent object with page data
        """

        try:
            self._ensure_browser()
            page: Page = self._browser.new_page()

            try:
                # Navigate to URL
                response = page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")

                if not response or not response.ok:
                    status = response.status if response else "unknown"
                    error_msg = f"Failed to load page (status: {status})"
                    logger.warning(f"{error_msg} - {url}")
                    return FetchedContent(
                        url=url,
                        title="",
                        text_content="",
                        main_content="",
                        timestamp=timestamp,
                        success=False,
                        error_message=error_msg,
                    )

                # Wait a bit for dynamic content
                page.wait_for_timeout(1000)  # 1 second

                # Get page title
                title = page.title()

                # Get HTML content
                html = page.content()
                soup = BeautifulSoup(html, "html.parser")

                # Extract text content
                text_content = soup.get_text(separator="\n", strip=True)

                # Extract main content
                main_content = self._extract_main_content(soup)

                # Limit content length
                max_length = settings.WEB_CONTENT_MAX_LENGTH
                if len(text_content) > max_length:
                    text_content = text_content[:max_length] + "..."
                if len(main_content) > max_length:
                    main_content = main_content[:max_length] + "..."

                logger.info(f"Successfully fetched {url} - {len(text_content)} chars")

                return FetchedContent(
                    url=url,
                    title=title,
                    text_content=text_content,
                    main_content=main_content,
                    timestamp=timestamp,
                    success=True,
                )

            finally:
                page.close()

        except PlaywrightTimeout:
            error_msg = f"Timeout loading page (>{self.timeout_ms}ms)"
            logger.warning(f"{error_msg} - {url}")
            return FetchedContent(
                url=url,
                title="",
                text_content="",
                main_content="",
                timestamp=timestamp,
                success=False,
                error_message=error_msg,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error fetching {url}: {error_msg}")
            return FetchedContent(
                url=url,
                title="",
                text_content="",
                main_content="",
                timestamp=timestamp,
                success=False,
                error_message=error_msg,
            )

    def fetch_url(self, url: str) -> FetchedContent:
        """Fetch content from a URL.

        Tries simple HTTP first (fast, no dependencies), falls back to Playwright if needed.

        Args:
            url: URL to fetch

        Returns:
            FetchedContent object with page data
        """
        logger.info(f"Fetching URL: {url}")
        timestamp = datetime.now()

        # Try simple HTTP first (works for most sites, no system deps needed)
        result = self._fetch_url_simple(url, timestamp)

        if result.success:
            return result

        # If simple HTTP failed, try Playwright (for JS-heavy sites)
        logger.info(f"Simple HTTP fetch failed for {url}, trying Playwright...")

        # Run sync Playwright in thread pool to avoid asyncio conflicts
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, use thread pool
                import concurrent.futures
                future = _thread_pool.submit(self._fetch_url_sync, url, timestamp)
                return future.result(timeout=self.timeout_ms / 1000 + 10)
        except RuntimeError:
            # No event loop, we can run directly
            pass

        # Fallback to direct call
        return self._fetch_url_sync(url, timestamp)

    def close(self):
        """Close browser and cleanup resources."""
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
        logger.info("Browser closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_url_fetcher_service() -> URLFetcherService:
    """Factory function to create URLFetcherService instance.

    Returns:
        Configured URLFetcherService instance
    """
    return URLFetcherService(
        headless=settings.BROWSER_HEADLESS,
        timeout_ms=settings.BROWSER_TIMEOUT_MS,
    )
