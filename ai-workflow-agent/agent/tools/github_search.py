# GitHub Search Tool
"""
Search GitHub repositories and provide recommendations.
Simple keyword search with README summarization.
"""

import httpx
import logging
from typing import List, Dict, Any, Optional

from config import settings

logger = logging.getLogger(__name__)


class GitHubSearchTool:
    """
    GitHub repository search tool.
    Searches for relevant projects and provides recommendations.
    """
    
    def __init__(self):
        self.api_base = "https://api.github.com"
        self.token = settings.GITHUB_TOKEN
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Set headers
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Workflow-Agent"
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
    
    async def search(
        self,
        keywords: str,
        max_results: int = 3,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search GitHub repositories by keywords.
        
        Args:
            keywords: Search terms
            max_results: Maximum number of results (default 3)
            language: Optional language filter
            
        Returns:
            List of repository info dictionaries
        """
        try:
            # Build search query
            query = keywords
            if language:
                query += f" language:{language}"
            
            # Search repositories
            response = await self.client.get(
                f"{self.api_base}/search/repositories",
                params={
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": max_results
                },
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                repos = data.get("items", [])
                
                # Extract relevant info
                results = []
                for repo in repos[:max_results]:
                    repo_info = await self._extract_repo_info(repo)
                    results.append(repo_info)
                
                return results
            else:
                logger.error(f"GitHub search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"GitHub search error: {e}")
            return []
    
    async def _extract_repo_info(self, repo: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant information from repository data."""
        # Get README summary
        readme_summary = await self._get_readme_summary(
            repo.get("owner", {}).get("login", ""),
            repo.get("name", "")
        )
        
        return {
            "name": repo.get("name", ""),
            "full_name": repo.get("full_name", ""),
            "url": repo.get("html_url", ""),
            "clone_url": repo.get("clone_url", ""),
            "description": repo.get("description", "No description"),
            "stars": repo.get("stargazers_count", 0),
            "forks": repo.get("forks_count", 0),
            "language": repo.get("language", "Unknown"),
            "topics": repo.get("topics", []),
            "updated_at": repo.get("updated_at", ""),
            "has_docker": await self._check_has_docker(
                repo.get("owner", {}).get("login", ""),
                repo.get("name", "")
            ),
            "readme_summary": readme_summary
        }
    
    async def _get_readme_summary(self, owner: str, repo: str) -> str:
        """Fetch and summarize repository README."""
        try:
            response = await self.client.get(
                f"{self.api_base}/repos/{owner}/{repo}/readme",
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                # README is base64 encoded
                import base64
                content = base64.b64decode(data.get("content", "")).decode("utf-8")
                
                # Simple summary: first 500 chars
                summary = content[:500].replace("\n", " ").strip()
                if len(content) > 500:
                    summary += "..."
                return summary
            return "README not available"
            
        except Exception as e:
            logger.warning(f"README fetch error: {e}")
            return "README not available"
    
    async def _check_has_docker(self, owner: str, repo: str) -> bool:
        """Check if repository has Dockerfile or docker-compose."""
        try:
            # Check for Dockerfile
            response = await self.client.get(
                f"{self.api_base}/repos/{owner}/{repo}/contents/Dockerfile",
                headers=self.headers
            )
            if response.status_code == 200:
                return True
            
            # Check for docker-compose
            response = await self.client.get(
                f"{self.api_base}/repos/{owner}/{repo}/contents/docker-compose.yml",
                headers=self.headers
            )
            if response.status_code == 200:
                return True
                
            # Check for docker-compose.yaml
            response = await self.client.get(
                f"{self.api_base}/repos/{owner}/{repo}/contents/docker-compose.yaml",
                headers=self.headers
            )
            return response.status_code == 200
            
        except Exception:
            return False
    
    async def generate_recommendation(self, repos: List[Dict[str, Any]]) -> str:
        """
        Generate a recommendation summary for found repositories.
        
        Args:
            repos: List of repository info dictionaries
            
        Returns:
            Recommendation text
        """
        if not repos:
            return "No repositories found. Try different keywords."
        
        # Build recommendation
        lines = ["📦 **Repository Recommendations:**\n"]
        
        for i, repo in enumerate(repos, 1):
            stars = repo.get("stars", 0)
            docker_status = "✅ Docker" if repo.get("has_docker") else "❌ No Docker"
            
            lines.append(f"**{i}. {repo.get('name', 'Unknown')}** ⭐ {stars}")
            lines.append(f"   {repo.get('description', 'No description')}")
            lines.append(f"   Language: {repo.get('language', 'Unknown')} | {docker_status}")
            lines.append(f"   URL: {repo.get('url', '')}")
            lines.append("")
        
        # Add best pick recommendation
        if repos:
            best = max(repos, key=lambda r: r.get("stars", 0))
            if best.get("has_docker"):
                lines.append(f"💡 **Recommended:** {best.get('name')} (most stars + Docker support)")
            else:
                docker_repos = [r for r in repos if r.get("has_docker")]
                if docker_repos:
                    best_docker = max(docker_repos, key=lambda r: r.get("stars", 0))
                    lines.append(f"💡 **Recommended:** {best_docker.get('name')} (has Docker support)")
                else:
                    lines.append(f"💡 **Recommended:** {best.get('name')} (most stars, but needs Docker setup)")
        
        return "\n".join(lines)
    
    async def get_repo_details(self, repo_url: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific repository.
        
        Args:
            repo_url: GitHub repository URL or owner/repo format
            
        Returns:
            Repository details dictionary
        """
        try:
            # Parse owner/repo from URL or direct format
            if "github.com" in repo_url:
                parts = repo_url.rstrip("/").split("/")
                owner = parts[-2]
                repo = parts[-1]
            else:
                owner, repo = repo_url.split("/")
            
            response = await self.client.get(
                f"{self.api_base}/repos/{owner}/{repo}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                return await self._extract_repo_info(response.json())
            else:
                return {"error": f"Repository not found: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Repo details error: {e}")
            return {"error": str(e)}
