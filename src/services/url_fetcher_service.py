"""URL fetching service using Playwright for browser automation.

Handles fetching web pages with full JavaScript support.
Uses trafilatura for intelligent content extraction.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import asyncio

from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup
import trafilatura

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
    screenshot_path: Optional[str] = None  # Path to screenshot if captured


class URLFetcherService:
    """Service for fetching web page content using browser automation."""

    def __init__(
        self,
        headless: bool = True,
        timeout_ms: int = 30000,
        capture_screenshots: bool = False,
        screenshots_dir: Optional[Path] = None,
    ):
        """Initialize the URL fetcher.

        Args:
            headless: Run browser in headless mode (default: True)
            timeout_ms: Page load timeout in milliseconds (default: 30000)
            capture_screenshots: Whether to capture screenshots (default: False)
            screenshots_dir: Directory to save screenshots (default: from settings)
        """
        self.headless = headless if headless is not None else settings.BROWSER_HEADLESS
        self.timeout_ms = timeout_ms if timeout_ms is not None else settings.BROWSER_TIMEOUT_MS
        self.capture_screenshots = capture_screenshots if capture_screenshots is not None else settings.BROWSER_SCREENSHOTS_ENABLED
        self.screenshots_dir = screenshots_dir or Path(settings.BROWSER_SCREENSHOTS_PATH)
        self._playwright = None
        self._browser: Optional[Browser] = None

        # Create screenshots directory if needed
        if self.capture_screenshots:
            self.screenshots_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Screenshots enabled - saving to {self.screenshots_dir}")

    def _ensure_browser(self):
        """Ensure browser is initialized."""
        if self._browser is None:
            self._playwright = sync_playwright().start()
            # Launch with realistic browser args to avoid detection
            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox'
                ]
            )
            logger.info(f"Browser launched (headless={self.headless})")

    def _extract_main_content(self, html: str, url: str) -> str:
        """Extract main content from HTML using trafilatura.

        Args:
            html: Raw HTML content
            url: URL of the page (for context)

        Returns:
            Extracted main content text
        """
        # Use trafilatura for intelligent content extraction
        # It automatically removes ads, navigation, footers, etc.
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,  # Use fallback methods if needed
            favor_precision=False,  # Favor recall to get more content
        )

        if extracted and len(extracted) > 100:
            return extracted

        # Fallback to basic text extraction if trafilatura fails
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["script", "style", "nav", "footer", "aside", "header"]):
            tag.decompose()

        body = soup.find("body")
        if body:
            return body.get_text(separator="\n", strip=True)

        return soup.get_text(separator="\n", strip=True)

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
            # Use realistic browser headers to avoid 403 blocks
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

            # Get title
            title_tag = soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else url

            # Get text content
            text_content = soup.get_text(separator="\n", strip=True)

            # Extract main content using trafilatura
            main_content = self._extract_main_content(html, url)

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

            # Create context with realistic headers to avoid bot detection
            context = self._browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1',
                    'Cache-Control': 'max-age=0'
                }
            )
            page: Page = context.new_page()

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

                # Capture screenshot if enabled
                screenshot_path = None
                if self.capture_screenshots:
                    try:
                        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                        # Sanitize URL for filename
                        safe_url = url.replace("://", "_").replace("/", "_")[:50]
                        screenshot_filename = f"{timestamp_str}_{safe_url}.png"
                        screenshot_path = str(self.screenshots_dir / screenshot_filename)
                        page.screenshot(path=screenshot_path, full_page=False)
                        logger.info(f"Screenshot captured: {screenshot_path}")
                    except Exception as e:
                        logger.warning(f"Failed to capture screenshot: {e}")
                        screenshot_path = None

                # Get HTML content
                html = page.content()
                soup = BeautifulSoup(html, "html.parser")

                # Extract text content
                text_content = soup.get_text(separator="\n", strip=True)

                # Extract main content using trafilatura
                main_content = self._extract_main_content(html, url)

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
                    screenshot_path=screenshot_path,
                )

            finally:
                page.close()
                context.close()

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

        If screenshots are enabled, always uses Playwright.
        Otherwise, tries simple HTTP first, falls back to Playwright if needed.

        Args:
            url: URL to fetch

        Returns:
            FetchedContent object with page data
        """
        logger.info(f"Fetching URL: {url}")
        timestamp = datetime.now()

        # If screenshots enabled, must use Playwright
        if self.capture_screenshots:
            logger.info("Screenshots enabled - using Playwright")
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
        capture_screenshots=settings.BROWSER_SCREENSHOTS_ENABLED,
        screenshots_dir=Path(settings.BROWSER_SCREENSHOTS_PATH),
    )
