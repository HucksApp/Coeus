import asyncio
import aiohttp
import os
import json
import re
import time
from bs4 import BeautifulSoup
import googleapiclient.discovery
from concurrent.futures import ThreadPoolExecutor
from aiofiles import open as aio_open  # For non-blocking file writes
from duckduckgo_search import DDGS  # New DuckDuckGo integration


class CoeusImagesdataCollector:
    def __init__(self, save_dir, title=None, google_api_key=None, google_cx=None, bing_api_key=None, cache_size=5000):
        self.save_dir = os.path.join(save_dir, title) if title else save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.cache_file = os.path.join(self.save_dir, "collected_data.json")
        self.collected_data = []

        # Google Search API credentials (if provided)
        self.google_api_key = google_api_key
        self.google_cx = google_cx

        # Bing API credentials (if provided)
        self.bing_api_key = bing_api_key

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
            # Keep cache size manageable
            self.collected_data = self.collected_data[-self.cache_size:]
        with open(self.cache_file, "w", encoding="utf-8") as file:
            json.dump(self.collected_data, file, indent=4)

    def _is_duplicate(self, url):
        """Check if a URL already exists in the collected data."""
        return any(item["url"] == url for item in self.collected_data)

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
        """Search Bing for images using the Bing Image Search API."""
        if not self.bing_api_key:
            print("Bing API key not provided.")
            return []

        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}
        url = f"https://api.cognitive.microsoft.com/bing/v7.0/images/search?q={
            query}&count={num_results}"

        async with aiohttp.ClientSession(headers=headers) as session:
            response = await self._fetch_url(session, url)
            if response:
                return self._parse_bing_results(response)
        return []

    async def _search_bing_scrape(self, query, num_results=10):
        """Search Bing for images and return parsed results."""
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://www.bing.com/images/search?q={
            query}&count={num_results}"
        async with aiohttp.ClientSession(headers=headers) as session:
            html = await self._fetch_url(session, url)
            if html:
                return self._parse_image_results_scrape(html, source="Bing")
        return []

    def _parse_image_results_scrape(self, html, source):
        """Parse image results from HTML."""
        soup = BeautifulSoup(html, "lxml")
        results = []
        for img_tag in soup.find_all("img", src=True):
            img_url = img_tag["src"]
            if img_url.startswith("http") and not self._is_duplicate(img_url):
                results.append({"title": img_tag.get(
                    "alt", "No title"), "url": img_url, "source": source})
        return results

    def _parse_bing_results(self, response):
        """Parse Bing image search API results."""
        results = []
        try:
            data = response.json()
            for item in data.get("value", []):
                if not self._is_duplicate(item.get("contentUrl", "")):
                    results.append({
                        "title": item.get("name", "No title"),
                        "url": item.get("contentUrl", ""),
                        "width": item.get("width", ""),
                        "height": item.get("height", ""),
                        "source": "Bing"
                    })
        except Exception as e:
            print(f"Error parsing Bing results: {e}")
        return results

    async def _search_duckduckgo(self, query, num_results=10):
        """Search DuckDuckGo for images and return parsed results."""
        results = DDGS().images(query, safesearch='off', max_results=num_results)
        return [
            {"title": result['title'],  "height": result['height'],
                "width": result['width'], "url": result['url'], "source": "DuckDuckGo"}
            for result in results if not self._is_duplicate(result["url"])
        ]

    def _search_google_images(self, query, num_results=10):
        """Search Google for images using the Google Custom Search API."""
        if not self.google_api_key or not self.google_cx:
            print("Google API key or Custom Search Engine ID not provided.")
            return []

        service = googleapiclient.discovery.build(
            "customsearch", "v1", developerKey=self.google_api_key)
        try:
            res = service.cse().list(q=query, cx=self.google_cx,
                                     searchType="image", num=num_results).execute()
            print(res.get("items")[0])
            return [
                {"title": item["title"], "url": item["link"],   "height": item['image']
                    ['height'], "width": item['image']['width'], "source": "Google"}
                for item in res.get("items", []) if not self._is_duplicate(item["link"])
            ]
        except Exception as e:
            print(f"Error during Google search: {e}")
            return []

    async def _run_searches(self, query, num_results):
        """Run multiple searches asynchronously with performance optimization."""
        tasks = [
                self._search_duckduckgo(query, num_results),
        ]

        if self.bing_api_key:
            tasks.append(self._search_bing(query, num_results))
        else:
            tasks.append(self._search_bing_scrape(query, num_results))
        if self.google_api_key and self.google_cx:
            # Run Google in thread
            tasks.append(asyncio.to_thread(
                self._search_google_images, query, num_results))

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out exceptions that might occur during the async tasks
            return [item for sublist in results if not isinstance(sublist, Exception) for item in sublist]
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def collect_data(self, query, num_results=10):
        """Search the internet for related data."""
        start_time = time.time()
        results = asyncio.run(self._run_searches(query, num_results))
        new_results = [
            item for item in results if not self._is_duplicate(item["url"])]
        self.collected_data.extend(new_results)
        self._save_cache()
        print(f"Collected {len(results)} results in {
              time.time() - start_time:.2f} seconds")
        return results

    async def _download_image(self, result):
        """Download a single image."""
        try:
            file_name = f"{result['title'][:50]}.jpg".replace(
                "/", "_").replace("\\", "_")
            file_path = os.path.join(self.save_dir, file_name)

            # Skip download if file already exists
            if os.path.exists(file_path):
                print(f"File already exists: {file_path}")
                return

            async with aiohttp.ClientSession() as session:
                async with session.get(result["url"], timeout=10) as response:
                    if response.status == 200:
                        img_data = await response.read()
                        async with aio_open(file_path, "wb") as file:
                            await file.write(img_data)
                        print(f"Downloaded: {file_path}")
                    else:
                        print(f"Failed to download {
                              result['url']}: Status {response.status}")
        except Exception as e:
            print(f"Error downloading {result['url']}: {e}")

    def download_images(self):
        """Download images concurrently for training data."""
        async def download_all():
            tasks = [self._download_image(result)
                     for result in self.collected_data]
            await asyncio.gather(*tasks)
        # Check if an event loop is already running
        try:
            # If an event loop is already running, use await directly
            loop = asyncio.get_event_loop()
            loop.create_task(download_all())  # Use create_task instead of run
        except RuntimeError:  # No event loop is running, so we create one
            asyncio.run(download_all())
    # Ensure the event loop is properly closed

    def render_data(self):
        """Print collected data."""
        for result in self.collected_data:
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Source: {result['source']}")
            print("-" * 40)
