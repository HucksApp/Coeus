import os
import json
import re
import time
import asyncio
import aiohttp
from aiofiles import open as aio_open
import googleapiclient.discovery
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


class CoeusVideodataCollector:
    def __init__(self, save_dir, title=None, google_api_key=None, google_cx=None, youtube_api_key=None, bing_api_key=None, cache_size=5000):
        self.save_dir = os.path.join(save_dir, title) if title else save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.cache_file = os.path.join(self.save_dir, "collected_videos.json")
        self.collected_videos = []

        # API keys
        self.google_api_key = google_api_key
        self.youtube_api_key = youtube_api_key
        self.bing_api_key = bing_api_key
        self.google_cx = google_cx

        # Caching
        self.cache_size = cache_size
        self._load_cache()

    def _load_cache(self):
        """Load cached video data."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as file:
                self.collected_videos = json.load(file)
        else:
            self.collected_videos = []

    def _save_cache(self):
        """Save video data to cache, extending if the cache exists."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as file:
                    existing_data = json.load(file)
                # Merge existing data with new data, avoiding duplicates
                url_set = {item["url"] for item in existing_data}
                new_data = [
                    video for video in self.collected_videos if video["url"] not in url_set]
                existing_data.extend(new_data)

                # Ensure the cache size limit is respected
                if len(existing_data) > self.cache_size:
                    existing_data = existing_data[-self.cache_size:]

                # Write the extended data back to the file
                with open(self.cache_file, "w", encoding="utf-8") as file:
                    json.dump(existing_data, file, indent=4)
            else:
                # If the file does not exist, create and write the current data
                with open(self.cache_file, "w", encoding="utf-8") as file:
                    json.dump(self.collected_videos, file, indent=4)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def _search_youtube(self, query, max_results=10):
        """Search YouTube videos using the YouTube Data API."""
        if not self.youtube_api_key:
            print("YouTube API key is not provided.")
            return []

        service = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=self.youtube_api_key)
        try:
            response = service.search().list(q=query, part="snippet", type="video",
                                             maxResults=max_results).execute()
            return [
                {
                    "title": item["snippet"]["title"],
                    "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                    "source": "YouTube",
                }
                for item in response.get("items", [])
            ]
        except Exception as e:
            print(f"Error during YouTube search: {e}")
            return []

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
                await asyncio.sleep(1)  # Wait before retrying
        return None

    async def _search_bing_videos_scraping(self, query, num_results=10):
        """Scrape Bing search results for videos."""
        search_url = f"https://www.bing.com/videos/search?q={
            query}&count={num_results}"
        headers = {"User-Agent": "Mozilla/5.0"}

        async with aiohttp.ClientSession(headers=headers) as session:
            html = await self._fetch_url(session, search_url)
            if html:
                return self._parse_bing_video_results(html)
        return []

    def _parse_bing_video_results(self, html):
        """Parse the HTML of Bing's video search results page."""
        soup = BeautifulSoup(html, "lxml")
        results = []

        # Find video links on the page
        for video in soup.find_all("a", href=True):
            title = video.get_text(strip=True)
            url = video["href"]

            # Ensure the link is a video link and not a regular webpage link
            if "video" in url and title:
                results.append({
                    "title": title,
                    "url": url,
                    "source": "Bing",
                })

        # Filter out duplicate URLs
        return [video for video in results if not self._is_duplicate(video["url"])]

    async def _search_google_custom(self, query, num_results=10):
        """Search for videos using Google Custom Search and filter for video results."""
        if not self.google_api_key:
            print("Google API key is not provided.")
            return []

        # Customize your Custom Search Engine (CSE) to include video sites only.
        url = f"https://www.googleapis.com/customsearch/v1?q={query} video&cx={
            self.google_cx}&key={self.google_api_key}&num={num_results}"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            {
                                "title": item.get("title", "No title"),
                                "url": item.get("link", ""),
                                "source": "Google Custom Search",
                            }
                            for item in data.get("items", [])
                            # Example video domains
                            if "youtube.com" not in item.get("link", "")
                        ]
            except Exception as e:
                print(f"Error fetching Google Custom Search results: {e}")
        return []

    async def _search_bing_videos(self, query, num_results=10):
        """Search Bing videos using the Bing API."""
        if not self.bing_api_key:
            print("Bing API key not provided.")
            return []

        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}
        url = f"https://api.cognitive.microsoft.com/bing/v7.0/videos/search?q={
            query}&count={num_results}"

        async with aiohttp.ClientSession(headers=headers) as session:
            try:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            {
                                "title": item.get("name", "No title"),
                                "url": item.get("contentUrl", ""),
                                "source": "Bing",
                            }
                            for item in data.get("value", []) if not self._is_duplicate(item.get("contentUrl", ""))
                        ]
            except Exception as e:
                print(f"Error fetching Bing videos: {e}")
        return []

    async def _search_duckduckgo_videos(self, query, num_results=10, duration=None):
        """Search DuckDuckGo for videos."""
        try:
            ddgs = DDGS()
            results = ddgs.videos(query, safesearch="off",
                                  duration=duration, max_results=num_results)
            return [
                {"title": result.get("title", "No title"), "url": result.get(
                    "content", ""), "duration": result.get("duration", ""), "source": "DuckDuckGo"}
                for result in results if not self._is_duplicate(result.get("content", ""))
            ]
        except Exception as e:
            print(f"Error fetching DuckDuckGo videos: {e}")
            return []

    def _is_duplicate(self, url):
        """Check if a URL already exists in the collected data."""
        return any(item["url"] == url for item in self.collected_videos)

    async def _run_video_searches(self, query, max_results, duration=None):
        """Run video searches asynchronously."""
        tasks = [self._search_duckduckgo_videos(
            query, max_results, duration=duration)]

        if self.bing_api_key:
            tasks.append(self._search_bing_videos(query, max_results))
        else:
            tasks.append(self._search_bing_videos_scraping(query, max_results))
        if self.youtube_api_key:
            tasks.append(asyncio.to_thread(
                self._search_youtube, query, max_results))
        if self.google_api_key and self.google_cx:
            tasks.append(asyncio.to_thread(
                self._search_google_custom, query, max_results))
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out exceptions that might occur during the async tasks
            return [item for sublist in results if not isinstance(sublist, Exception) for item in sublist]
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def collect_videos(self, query, max_results=10, duration=None):
        """Collect video data for training."""
        start_time = time.time()
        results = asyncio.run(self._run_video_searches(
            query, max_results, duration))
        self.collected_videos.extend(results)
        self._save_cache()
        print(f"Collected {len(results)} videos in {
              time.time() - start_time:.2f} seconds")
        return results

    async def _download_video_metadata(self, video):
        """Save video metadata."""
        try:
            metadata_path = os.path.join(
                self.save_dir, f"{video['title'][:50]}.json")
            # Check if file already exists
            if not os.path.exists(metadata_path):
                async with aio_open(metadata_path, "w", encoding="utf-8") as file:
                    await file.write(json.dumps(video, indent=4))
                print(f"Saved metadata: {metadata_path}")
            else:
                print(f"Metadata already exists for {video['title'][:50]}")
        except Exception as e:
            print(f"Error saving metadata for {video['url']}: {e}")

    def download_videos(self):
        """Download video metadata."""
        loop = asyncio.get_event_loop()
        tasks = [self._download_video_metadata(
            video) for video in self.collected_videos]
        loop.run_until_complete(asyncio.gather(*tasks))

    def render_videos(self):
        """Display collected video data."""
        for video in self.collected_videos:
            print(f"Title: {video['title']}")
            print(f"URL: {video['url']}")
            print(f"Source: {video['source']}")
            print("-" * 40)
