"""
Stage 1 of WebFetch: HTML fetching and Markdown conversion.

This module handles:
- Fetching raw HTML from URLs via httpx
- HTTP → HTTPS auto-upgrade
- Redirect detection (same-host follows, cross-host returns redirect URL)
- HTML → clean Markdown conversion using trafilatura + markdownify
- Content size limiting
"""

import httpx
from urllib.parse import urlparse
from typing import Optional, Tuple

from .config import Config

# Try trafilatura first (better at extracting article content),
# fall back to markdownify (general-purpose HTML→MD)
try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

try:
    from markdownify import markdownify as md
    HAS_MARKDOWNIFY = True
except ImportError:
    HAS_MARKDOWNIFY = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


class HTMLToMarkdown:
    """
    Fetches a URL and converts its HTML content to clean Markdown.

    Behavior mirrors Claude's WebFetch Stage 1:
    - HTTP URLs automatically upgraded to HTTPS
    - Same-host redirects are followed
    - Cross-host redirects return the redirect URL (not followed)
    - Content is cleaned and converted to Markdown
    - Large content is truncated to MAX_CONTENT_LENGTH
    """

    def __init__(self):
        self.timeout = Config.FETCH_TIMEOUT
        self.max_length = Config.FETCH_MAX_CONTENT_LENGTH
        self.user_agent = Config.USER_AGENT

    async def fetch_and_convert(self, url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Fetch URL and convert to Markdown.

        Returns:
            Tuple of (markdown_content, redirect_url, error)
            - If successful: (markdown, None, None)
            - If cross-host redirect: (None, redirect_url, None)
            - If error: (None, None, error_message)
        """
        # Stage 1a: Upgrade HTTP to HTTPS
        url = self._upgrade_to_https(url)

        try:
            # Stage 1b: Fetch with redirect handling
            html, redirect_url = await self._fetch_html(url)

            if redirect_url:
                return None, redirect_url, None

            if not html:
                return None, None, "Empty response from server"

            # Stage 1c: Convert HTML to Markdown
            markdown = self._convert_to_markdown(html, url)

            if not markdown:
                return None, None, "Failed to extract content from page"

            # Stage 1d: Truncate if too large
            if len(markdown) > self.max_length:
                markdown = markdown[:self.max_length] + "\n\n[Content truncated due to size]"

            return markdown, None, None

        except httpx.ConnectError:
            return None, None, f"Cannot connect to {url}"
        except httpx.TimeoutException:
            return None, None, f"Request timed out after {self.timeout}s"
        except httpx.HTTPStatusError as e:
            return None, None, f"HTTP {e.response.status_code}: {e.response.reason_phrase}"
        except Exception as e:
            return None, None, f"Fetch error: {str(e)}"

    async def _fetch_html(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Fetch HTML with smart redirect handling.

        Cross-host redirects are NOT followed — instead the redirect URL
        is returned so the caller can decide (matching Claude's behavior).

        Returns:
            Tuple of (html_content, redirect_url)
        """
        original_host = urlparse(url).hostname

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=False,  # Handle redirects manually
            headers={"User-Agent": self.user_agent},
        ) as client:
            response = await client.get(url)

            # Handle redirects
            redirect_count = 0
            while response.is_redirect and redirect_count < 10:
                redirect_url = str(response.headers.get("location", ""))

                # Make relative URLs absolute
                if redirect_url.startswith("/"):
                    parsed = urlparse(url)
                    redirect_url = f"{parsed.scheme}://{parsed.hostname}{redirect_url}"

                redirect_host = urlparse(redirect_url).hostname

                # Cross-host redirect → return the URL, don't follow
                if redirect_host and redirect_host != original_host:
                    return None, redirect_url

                # Same-host redirect → follow it
                url = redirect_url
                response = await client.get(url)
                redirect_count += 1

            response.raise_for_status()
            return response.text, None

    def _convert_to_markdown(self, html: str, source_url: str) -> Optional[str]:
        """
        Convert HTML to clean Markdown.

        Strategy (4 tiers, most precise → most aggressive):
        1. Try trafilatura (best for article/content extraction)
        2. Fall back to markdownify (general HTML→MD)
        3. Fall back to BeautifulSoup text extraction
        4. Last resort: regex strip all tags from raw HTML
        """
        markdown = None

        # Strategy 1: trafilatura (content-focused extraction)
        if HAS_TRAFILATURA:
            try:
                extracted = trafilatura.extract(
                    html,
                    include_links=True,
                    include_tables=True,
                    include_images=False,
                    output_format="txt",
                    url=source_url,
                )
                if extracted and len(extracted.strip()) > 50:
                    markdown = extracted
            except Exception:
                pass

        # Strategy 2: markdownify (full HTML→MD conversion)
        if not markdown and HAS_MARKDOWNIFY:
            try:
                # First clean with BS4 if available
                if HAS_BS4:
                    soup = BeautifulSoup(html, "html.parser")
                    for tag in soup(["script", "style", "nav", "footer",
                                     "header", "aside", "noscript"]):
                        tag.decompose()
                    html_clean = str(soup)
                else:
                    html_clean = html

                result = md(
                    html_clean,
                    heading_style="ATX",
                    strip=["img", "script", "style"],
                )
                if result and len(result.strip()) > 20:
                    markdown = result
            except Exception:
                pass

        # Strategy 3: Plain text extraction via BS4
        if not markdown and HAS_BS4:
            try:
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                if text and len(text.strip()) > 20:
                    markdown = text
            except Exception:
                pass

        # Strategy 4: Last resort — regex strip all HTML tags
        if not markdown:
            try:
                import re
                text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                if text and len(text) > 20:
                    markdown = text
            except Exception:
                pass

        return markdown.strip() if markdown else None

    @staticmethod
    def _upgrade_to_https(url: str) -> str:
        """Auto-upgrade HTTP URLs to HTTPS (matching Claude's behavior)."""
        if url.startswith("http://"):
            return "https://" + url[7:]
        if not url.startswith("https://"):
            return "https://" + url
        return url
