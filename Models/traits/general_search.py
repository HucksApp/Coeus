import asyncio
import aiohttp
import os
import json
from bs4 import BeautifulSoup
import requests
import re
from concurrent.futures import ThreadPoolExecutor


class CoeusSearch:
    def __init__(self, save_dir, title=None):
        self.save_dir = os.path.join(save_dir, title) if title else save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.cache_file = os.path.join(self.save_dir, "search_results.json")
        self.search_results = []

        # Load cached results
        self._load_cache()

    def _load_cache(self):
        """Load cached search results."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as file:
                self.search_results = json.load(file)

    def _save_cache(self):
        """Save search results to cache."""
        with open(self.cache_file, "w", encoding="utf-8") as file:
            json.dump(self.search_results, file, indent=4)

    async def _fetch_url(self, session, url):
        """Fetch a single URL asynchronously."""
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    return await response.text()
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        return None

    async def _search_bing(self, query, num_results=10):
        """Search Bing and return parsed results."""
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://www.bing.com/search?q={query}&count={num_results}"
        async with aiohttp.ClientSession(headers=headers) as session:
            html = await self._fetch_url(session, url)
            if html:
                return self._parse_search_results(html, source="Bing")
        return []

    async def _search_duckduckgo(self, query, num_results=10):
        """Search DuckDuckGo and return parsed results."""
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://duckduckgo.com/html?q={query}&num={num_results}"
        async with aiohttp.ClientSession(headers=headers) as session:
            html = await self._fetch_url(session, url)
            if html:
                return self._parse_search_results(html, source="DuckDuckGo")
        return []

    def _parse_search_results(self, html, source):
        """Parse search results using BeautifulSoup."""
        soup = BeautifulSoup(html, "lxml")
        results = []
        for link in soup.find_all("a", href=True):
            if "http" in link["href"]:
                results.append(
                    {
                        "title": link.get_text(strip=True),
                        "url": link["href"],
                        "source": source,
                    }
                )
        return results

    async def _run_searches(self, query, num_results):
        """Run multiple searches asynchronously."""
        tasks = [
            self._search_bing(query, num_results),
            self._search_duckduckgo(query, num_results),
        ]
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

    def search(self, query, num_results=10):
        """Search the internet for related articles and websites."""
        # Run async searches
        results = asyncio.run(self._run_searches(query, num_results))
        self.search_results = results
        self._save_cache()
        return results

    def filter_results(self, keywords):
        """Filter results based on keywords."""
        keyword_pattern = re.compile(r"|".join(map(re.escape, keywords)), re.IGNORECASE)
        filtered_results = [
            result
            for result in self.search_results
            if keyword_pattern.search(result["title"])
            or keyword_pattern.search(result["url"])
        ]
        return filtered_results

    def render_results(self):
        """Print search results."""
        for result in self.search_results:
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Source: {result['source']}")
            print("-" * 40)

    def download_articles(self):
        """Download articles concurrently for offline reading."""
        def download_article(result):
            try:
                response = requests.get(result["url"], timeout=10)
                if response.status_code == 200:
                    file_path = os.path.join(self.save_dir, f"{result['title'][:50]}.html")
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(response.text)
                    print(f"Downloaded: {file_path}")
            except Exception as e:
                print(f"Error downloading {result['url']}: {e}")

        with ThreadPoolExecutor() as executor:
            executor.map(download_article, self.search_results)

