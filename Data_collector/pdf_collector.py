import asyncio
import aiohttp
import os
import json
import re
import requests
from bs4 import BeautifulSoup
import time
import googleapiclient.discovery
from aiofiles import open as aio_open
from duckduckgo_search import DDGS


class CoeusPdfCollector:
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
        self.cache_size = cache_size
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
        # Remove duplicates before saving
        seen_urls = set()
        unique_data = []
        for item in self.collected_data:
            if item["url"] not in seen_urls:
                unique_data.append(item)
                seen_urls.add(item["url"])
        self.collected_data = unique_data

        # Save to cache file
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
                await asyncio.sleep(1)  # Wait before retrying
        return None

    async def _search_bing(self, query, num_results=10):
        """Search Bing for PDFs using the Bing API."""
        if not self.bing_api_key:
            print("Bing API key not provided.")
            return []

        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}
        url = f"https://api.cognitive.microsoft.com/bing/v7.0/search?q={
            query}+filetype:pdf&count={num_results}"
        async with aiohttp.ClientSession(headers=headers) as session:
            html = await self._fetch_url(session, url)
            if html:
                return self._parse_pdf_results(html, source="Bing")
        return []

    async def _search_bing_scraping(self, query, num_results=10):
        """Search Bing for PDFs using web scraping."""
        # Construct the Bing search URL for PDFs
        url = f"https://www.bing.com/search?q={query}+filetype:pdf"

        # Fetch the HTML content of the Bing search results page
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        return self._parse_pdf_results(html, source="Bing")
                    else:
                        print(f"Failed to fetch Bing search results. Status: {
                              response.status}")
            except Exception as e:
                print(f"Error fetching Bing search results: {e}")
        return []

    def _search_google_pdfs(self, query, num_results=10):
        """Search Google for PDFs using the Custom Search API."""
        if not self.google_api_key or not self.google_cx:
            print("Google API key or Custom Search Engine ID not provided.")
            return []

        service = googleapiclient.discovery.build(
            "customsearch", "v1", developerKey=self.google_api_key)
        try:
            res = service.cse().list(
                q=f"{query} filetype:pdf", cx=self.google_cx, num=num_results).execute()
            return [
                {"title": item["title"],
                    "url": item["link"], "source": "Google"}
                for item in res.get("items", []) if not self._is_duplicate(item["link"])
            ]
        except Exception as e:
            print(f"Error during Google search: {e}")
            return []

    async def _search_duckduckgo(self, query, num_results=10):
        """Search DuckDuckGo for PDFs."""
        ddgs = DDGS()
        results = ddgs.text(f'{query} filetype:pdf',
                            safesearch='off', max_results=num_results)
        return [
            {"title": result["title"], "url": result["href"],
                "source": "DuckDuckGo"}
            for result in results
            if result["href"].endswith(".pdf") and not self._is_duplicate(result["href"])
        ]

    def _is_duplicate(self, url):
        """Check if a URL already exists in the collected data."""
        return any(item["url"] == url for item in self.collected_data)

    def _parse_pdf_results(self, html, source):
        """Parse PDF results from HTML."""
        soup = BeautifulSoup(html, "lxml")
        results = []
        for a_tag in soup.find_all("a", href=True):
            pdf_url = a_tag["href"]
            if pdf_url.endswith(".pdf"):
                if not self._is_duplicate(pdf_url):
                    results.append({"title": a_tag.get_text(
                        strip=True), "url": pdf_url, "source": source})
        return results

    async def _run_searches(self, query, num_results):
        """Run multiple searches asynchronously with performance optimization."""
        tasks = [
            self._search_duckduckgo(query, num_results),
        ]  # DuckDuckGo
        if self.bing_api_key:
            tasks.append(self._search_bing(query, num_results))
        else:
            print("No bing API key proceeding with bing scraping")
            tasks.append(self._search_bing_scraping(query, num_results))
        if self.google_api_key and self.google_cx:
            # Google Search in thread
            tasks.append(asyncio.to_thread(
                self._search_google_pdfs, query, num_results))

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out exceptions that might occur during the async tasks
            return [item for sublist in results if not isinstance(sublist, Exception) for item in sublist]
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def collect_data(self, query, num_results=10):
        """Search the internet for related PDFs."""
        start_time = time.time()
        results = asyncio.run(self._run_searches(query, num_results))

        # Remove duplicates by URL
        existing_urls = {item["url"]
                         for item in self.collected_data}  # URLs from cache
        unique_results = [
            res for res in results if res["url"] not in existing_urls]

        # Add only unique results to the collected data
        self.collected_data.extend(unique_results)

        self._save_cache()
        print(f"Collected {len(unique_results)} new results "
              f"(total {len(self.collected_data)}) in {time.time() - start_time:.2f} seconds")
        return unique_results

    async def _download_pdf(self, result):
        """Download a single PDF concurrently."""
        try:
            # Asyncify requests
            pdf_data = await asyncio.to_thread(requests.get, result["url"], timeout=10)

            # Sanitize title for safe file naming
            sanitized_title = re.sub(
                r'[^\w\-_\. ]', '_', result['title']).strip()[:50]
            if not sanitized_title:  # Fallback if title is empty after sanitization
                sanitized_title = "document"

            file_path = os.path.join(self.save_dir, f"{sanitized_title}.pdf")

            if os.path.exists(file_path):
                print(f"File already exists: {file_path}")
                return  # Skip downloading if the file exists

            async with aio_open(file_path, "wb") as file:
                # Asynchronous write to disk
                await file.write(pdf_data.content)
            print(f"Downloaded: {file_path}")
        except Exception as e:
            print(f"Error downloading {result['url']}: {e}")

    def download_pdfs(self):
        """Download PDFs concurrently."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            tasks = [self._download_pdf(result)
                     for result in self.collected_data]
            loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            loop.close()

    def render_data(self):
        """Print collected data."""
        for result in self.collected_data:
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Source: {result['source']}")
            print("-" * 40)

