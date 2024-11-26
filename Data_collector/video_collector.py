import os
import json
import time
import asyncio
import aiohttp
from aiofiles import open as aio_open
from bs4 import BeautifulSoup
import googleapiclient.discovery


class CoeusVideodataCollector:
    def __init__(self, save_dir, title=None, google_api_key=None, youtube_api_key=None, cache_size=5000):
        self.save_dir = os.path.join(save_dir, title) if title else save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.cache_file = os.path.join(self.save_dir, "collected_videos.json")
        self.collected_videos = []

        # API keys
        self.google_api_key = google_api_key
        self.youtube_api_key = youtube_api_key

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
        """Save video data to cache."""
        if len(self.collected_videos) > self.cache_size:
            self.collected_videos = self.collected_videos[-self.cache_size:]
        with open(self.cache_file, "w", encoding="utf-8") as file:
            json.dump(self.collected_videos, file, indent=4)

    async def _fetch_url(self, session, url):
        """Fetch a URL asynchronously with retries."""
        retries = 3
        for _ in range(retries):
            try:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        return await response.text()
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                await asyncio.sleep(1)  # Retry delay
        return None

    def _search_youtube(self, query, max_results=10):
        """Search YouTube videos using the YouTube Data API."""
        if not self.youtube_api_key:
            print("YouTube API key is not provided.")
            return []

        service = googleapiclient.discovery.build("youtube", "v3", developerKey=self.youtube_api_key)
        try:
            response = service.search().list(q=query, part="snippet", type="video", maxResults=max_results).execute()
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

    async def _search_bing_videos(self, query, num_results=10):
        """Search Bing for videos."""
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://www.bing.com/videos/search?q={query}&count={num_results}"
        async with aiohttp.ClientSession(headers=headers) as session:
            html = await self._fetch_url(session, url)
            if html:
                return self._parse_video_results(html, source="Bing")
        return []

    def _parse_video_results(self, html, source):
        """Parse video results from HTML."""
        soup = BeautifulSoup(html, "lxml")
        results = []
        for video_tag in soup.find_all("a", href=True):
            video_url = video_tag["href"]
            if video_url.startswith("http"):
                title = video_tag.get("title", "No title")
                results.append({"title": title, "url": video_url, "source": source})
        return results

    async def _run_video_searches(self, query, max_results):
        """Run video searches asynchronously."""
        tasks = [self._search_bing_videos(query, max_results)]

        if self.youtube_api_key:
            tasks.append(asyncio.to_thread(self._search_youtube, query, max_results))

        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

    def collect_videos(self, query, max_results=10):
        """Collect video data for training."""
        start_time = time.time()
        results = asyncio.run(self._run_video_searches(query, max_results))
        self.collected_videos.extend(results)
        self._save_cache()
        print(f"Collected {len(results)} videos in {time.time() - start_time:.2f} seconds")
        return results

    async def _download_video(self, video):
        """Download video metadata or video file."""
        try:
            metadata_path = os.path.join(self.save_dir, f"{video['title'][:50]}.json")
            async with aio_open(metadata_path, "w", encoding="utf-8") as file:
                await file.write(json.dumps(video, indent=4))
            print(f"Saved metadata: {metadata_path}")
        except Exception as e:
            print(f"Error saving metadata for {video['url']}: {e}")

    def download_videos(self):
        """Download video metadata or files."""
        loop = asyncio.get_event_loop()
        tasks = [self._download_video(video) for video in self.collected_videos]
        loop.run_until_complete(asyncio.gather(*tasks))

    def render_videos(self):
        """Display collected video data."""
        for video in self.collected_videos:
            print(f"Title: {video['title']}")
            print(f"URL: {video['url']}")
            print(f"Source: {video['source']}")
            print("-" * 40)
