import asyncio
import aiohttp
import os
import json
import requests
from bs4 import BeautifulSoup
import re
import time
import googleapiclient.discovery
from concurrent.futures import ThreadPoolExecutor
from aiofiles import open as aio_open  # For non-blocking file writes

class CoeusDataCollector:
    def __init__(self, save_dir, title=None, google_api_key=None, google_cx=None, cache_size=5000):
        self.save_dir = os.path.join(save_dir, title) if title else save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.cache_file = os.path.join(self.save_dir, "collected_data.json")
        self.collected_data = []

        # Google Search API credentials (if provided)
        self.google_api_key = google_api_key
        self.google_cx = google_cx

        # Caching for large datasets
        self.cache_size = cache_size  # Define how many items we want to keep in cache
        self._load_cache()

    def _load_cache(self):
        """Load cached collected data."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as file:
                self.collected_data = json.load(file)
        else:
            self.collected_data = []

    def _save_cache(self):
        """Save collected data to cache."""
        if len(self.collected_data) > self.cache_size:
            self.collected_data = self.collected_data[-self.cache_size:]  # Keep cache size manageable
        with open(self.cache_file, "w", encoding="utf-8") as file:
            json.dump(self.collected_data, file, indent=4)

    async def _fetch_url(self, session, url):
        """Fetch a single URL asynchronously with retries."""
        retries = 3
        for _ in range(retries):
            try:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        return await response.text()
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                await asyncio.sleep(1)  # wait before retrying
        return None

    async def _search_bing(self, query, num_results=10):
        """Search Bing for images and return parsed results."""
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://www.bing.com/images/search?q={query}&count={num_results}"
        async with aiohttp.ClientSession(headers=headers) as session:
            html = await self._fetch_url(session, url)
            if html:
                return self._parse_image_results(html, source="Bing")
        return []

    async def _search_duckduckgo(self, query, num_results=10):
        """Search DuckDuckGo for images and return parsed results."""
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://duckduckgo.com/html?q={query}&num={num_results}"
        async with aiohttp.ClientSession(headers=headers) as session:
            html = await self._fetch_url(session, url)
            if html:
                return self._parse_image_results(html, source="DuckDuckGo")
        return []

    def _search_google_images(self, query, num_results=10):
        """Search Google for images using the Google Custom Search API."""
        if not self.google_api_key or not self.google_cx:
            print("Google API key or Custom Search Engine ID not provided.")
            return []

        service = googleapiclient.discovery.build("customsearch", "v1", developerKey=self.google_api_key)
        try:
            res = service.cse().list(q=query, cx=self.google_cx, searchType="image", num=num_results).execute()
            return [
                {"title": item["title"], "url": item["link"], "source": "Google"}
                for item in res.get("items", [])
            ]
        except Exception as e:
            print(f"Error during Google search: {e}")
            return []

    def _parse_image_results(self, html, source):
        """Parse image results from HTML."""
        soup = BeautifulSoup(html, "lxml")
        results = []
        for img_tag in soup.find_all("img", src=True):
            img_url = img_tag["src"]
            if img_url.startswith("http"):
                results.append({"title": img_tag.get("alt", "No title"), "url": img_url, "source": source})
        return results

    async def _run_searches(self, query, num_results):
        """Run multiple searches asynchronously with performance optimization."""
        tasks = [
            self._search_bing(query, num_results),
            self._search_duckduckgo(query, num_results),
        ]

        if self.google_api_key and self.google_cx:
            tasks.append(asyncio.to_thread(self._search_google_images, query, num_results))  # Run Google in thread
        
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

    def collect_data(self, query, num_results=10):
        """Search the internet for related data."""
        start_time = time.time()
        results = asyncio.run(self._run_searches(query, num_results))
        self.collected_data = results
        self._save_cache()
        print(f"Collected {len(results)} results in {time.time() - start_time:.2f} seconds")
        return results

    async def _download_image(self, result):
        """Download a single image concurrently."""
        try:
            img_data = await asyncio.to_thread(requests.get, result["url"], timeout=10)  # Asyncify requests
            file_path = os.path.join(self.save_dir, f"{result['title'][:50]}.jpg")
            async with aio_open(file_path, "wb") as file:
                await file.write(img_data.content)  # Asynchronous write to disk
            print(f"Downloaded: {file_path}")
        except Exception as e:
            print(f"Error downloading {result['url']}: {e}")

    def download_images(self):
        """Download images concurrently for training data."""
        loop = asyncio.get_event_loop()
        tasks = [self._download_image(result) for result in self.collected_data]
        loop.run_until_complete(asyncio.gather(*tasks))

    def render_data(self):
        """Print collected data."""
        for result in self.collected_data:
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Source: {result['source']}")
            print("-" * 40)
