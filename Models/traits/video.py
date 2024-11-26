import os
import json
import requests
import threading
import re
import gzip


class CoeusVideos:
    def __init__(self, api_key, save_dir, title=None):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.save_dir = os.path.join(save_dir, title) if title else save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Storage for video metadata
        self.video_metadata = []

    def search_videos(self, query, max_results=10):
        """Search for videos using the YouTube Data API with optimized request handling."""
        search_url = f"{self.base_url}/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "key": self.api_key,
        }
        response = requests.get(search_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            self.video_metadata = self._process_video_data(data["items"])
            self._save_metadata()
            return self.video_metadata
        else:
            raise Exception(f"Error fetching videos: {response.json()}")

    def _process_video_data(self, items):
        """Efficiently process raw video data into structured metadata."""
        return [
            {
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "video_id": item["id"]["videoId"],
                "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
                "publish_time": item["snippet"]["publishTime"],
                "link": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
            }
            for item in items
        ]

    def categorize_videos(self):
        """Categorize videos using pre-compiled regex patterns."""
        categories = {"installation": [], "repair": [], "maintenance": []}
        patterns = {
            "installation": re.compile(r"\b(install|setup|configure|mount)\b", re.IGNORECASE),
            "repair": re.compile(r"\b(repair|fix|troubleshoot|fault|issue)\b", re.IGNORECASE),
            "maintenance": re.compile(r"\b(maintain|service|routine|schedule)\b", re.IGNORECASE),
        }

        for video in self.video_metadata:
            for category, pattern in patterns.items():
                if pattern.search(video["title"]) or pattern.search(video["description"]):
                    categories[category].append(video)
                    break  # Assign to the first matching category to avoid duplication

        return categories

    def _save_metadata(self):
        """Save video metadata to a compressed JSON file."""
        metadata_path = os.path.join(self.save_dir, "video_metadata.json.gz")
        with gzip.open(metadata_path, "wt", encoding="utf-8") as gz_file:
            json.dump(self.video_metadata, gz_file, indent=4)

    def _load_metadata(self):
        """Load video metadata from a compressed JSON file."""
        metadata_path = os.path.join(self.save_dir, "video_metadata.json.gz")
        if os.path.exists(metadata_path):
            with gzip.open(metadata_path, "rt", encoding="utf-8") as gz_file:
                self.video_metadata = json.load(gz_file)

    def render_videos(self, category=None):
        """Render video information with optional category filtering."""
        if category:
            categorized_videos = self.categorize_videos().get(category, [])
            videos = categorized_videos
        else:
            videos = self.video_metadata

        for video in videos:
            print(f"Title: {video['title']}")
            print(f"Description: {video['description']}")
            print(f"Link: {video['link']}")
            print(f"Thumbnail: {video['thumbnail']}")
            print(f"Published: {video['publish_time']}")
            print("-" * 40)

    def download_thumbnails(self):
        """Download all thumbnails using multithreading for speed."""
        def download_thumbnail(video):
            response = requests.get(video["thumbnail"], stream=True, timeout=10)
            if response.status_code == 200:
                thumbnail_path = os.path.join(self.save_dir, f"{video['video_id']}.jpg")
                with open(thumbnail_path, "wb") as img_file:
                    for chunk in response.iter_content(1024):
                        img_file.write(chunk)
                print(f"Downloaded: {thumbnail_path}")

        threads = []
        for video in self.video_metadata:
            thread = threading.Thread(target=download_thumbnail, args=(video,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
