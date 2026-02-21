# Web Search Tool
"""
Search the web for project recommendations and documentation.
Uses DuckDuckGo (no API key required) for search.
"""

import httpx
import logging
import re
from typing import Dict, Any, List, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    Web search tool for finding project recommendations.
    Uses DuckDuckGo HTML search (no API required).
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    async def search(
        self,
        query: str,
        max_results: int = 5,
        site_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the web for relevant results.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            site_filter: Optional site to filter (e.g., "github.com")
            
        Returns:
            List of search results with title, url, snippet
        """
        try:
            # Build search query
            search_query = query
            if site_filter:
                search_query = f"site:{site_filter} {query}"
            
            # Use DuckDuckGo HTML search
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(search_query)}"
            
            response = await self.client.get(url, headers=self.headers)
            
            if response.status_code == 200:
                results = self._parse_ddg_results(response.text, max_results)
                return results
            else:
                logger.warning(f"Search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    def _parse_ddg_results(self, html: str, max_results: int) -> List[Dict[str, Any]]:
        """Parse DuckDuckGo HTML results."""
        results = []
        
        # Simple regex parsing for result links
        # Pattern matches result entries in DuckDuckGo HTML
        link_pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>'
        snippet_pattern = r'<a class="result__snippet"[^>]*>([^<]+(?:<[^>]+>[^<]*</[^>]+>)*[^<]*)</a>'
        
        links = re.findall(link_pattern, html)
        snippets = re.findall(snippet_pattern, html)
        
        for i, (url, title) in enumerate(links[:max_results]):
            # Clean up URL (DuckDuckGo wraps URLs)
            if "uddg=" in url:
                # Extract actual URL from redirect
                match = re.search(r'uddg=([^&]+)', url)
                if match:
                    from urllib.parse import unquote
                    url = unquote(match.group(1))
            
            result = {
                "title": self._clean_html(title),
                "url": url,
                "snippet": self._clean_html(snippets[i]) if i < len(snippets) else ""
            }
            results.append(result)
        
        return results
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and clean text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&quot;', '"').replace('&#39;', "'")
        # Clean whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    async def search_github_projects(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search specifically for GitHub projects."""
        return await self.search(query, max_results, site_filter="github.com")
    
    async def search_documentation(
        self,
        tool: str,
        topic: str,
        max_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Search for tool documentation."""
        query = f"{tool} documentation {topic}"
        return await self.search(query, max_results)
    
    async def find_alternatives(
        self,
        tool: str,
        purpose: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Find alternative tools for a specific purpose."""
        query = f"best {purpose} tools alternatives to {tool} open source"
        return await self.search(query, max_results)
    
    async def search_with_summary(
        self,
        query: str,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search and generate a summary of findings.
        
        Returns:
            Dict with results and AI-generated summary
        """
        results = await self.search(query, max_results)
        
        if not results:
            return {
                "results": [],
                "summary": "No results found for your query."
            }
        
        # Generate simple summary
        summary_parts = [f"Found {len(results)} relevant results:"]
        for i, r in enumerate(results, 1):
            summary_parts.append(f"{i}. **{r['title']}**")
            if r['snippet']:
                summary_parts.append(f"   {r['snippet'][:100]}...")
        
        return {
            "results": results,
            "summary": "\n".join(summary_parts)
        }


# Singleton instance
web_search = WebSearchTool()
