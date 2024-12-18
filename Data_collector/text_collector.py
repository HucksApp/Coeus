import asyncio
import aiohttp
import os
import json
import re
import time
from bs4 import BeautifulSoup
from aiofiles import open as aio_open  # For non-blocking file writes
from concurrent.futures import ThreadPoolExecutor
import googleapiclient.discovery
from duckduckgo_search import DDGS
from transformers import pipeline
import spacy


class CoeusTextdataCollector:
    def __init__(self, save_dir, title=None, cache_size=5000, google_api_key=None, google_cx=None, bing_api_key=None):
        self.save_dir = os.path.join(save_dir, title) if title else save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.cache_file = os.path.join(self.save_dir, "collected_data.json")
        self.collected_data = []

        # Cache management
        self.cache_size = cache_size
        self.google_api_key = google_api_key
        self.google_cx = google_cx
        # Bing API credentials (if provided)
        self.bing_api_key = bing_api_key

        self.nlp = spacy.load("en_core_web_sm")
        self.qa_pipeline = pipeline(
            "question-answering", model="deepset/roberta-base-squad2")
        self.summarizer = pipeline(
            "summarization", model="facebook/bart-large-cnn")
        
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
            # Trim cache to size
            self.collected_data = self.collected_data[-self.cache_size:]
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
        """Search Bing for textual content and return parsed results."""
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://www.bing.com/search?q={query}&count={num_results}"
        async with aiohttp.ClientSession(headers=headers) as session:
            html = await self._fetch_url(session, url)
            if html:
                return self._parse_text_results(html, source="Bing")
        return []

    async def _search_duckduckgo(self, query, num_results=10):
        """Search DuckDuckGo for textual content, targeting blogs and Q&A platforms."""
        results = []
        try:
            with DDGS() as ddgs:
                search_query = f"{
                    query} site:quora.com OR site:stackoverflow.com OR site:reddit.com"
                for result in ddgs.text(search_query, max_results=num_results):
                    if not self._is_duplicate(result["href"]):
                        results.append({
                            "title": result["title"],
                            "url": result["href"],
                            "source": "DuckDuckGo"
                        })
        except Exception as e:
            print(f"Error searching DuckDuckGo: {e}")
        return results

    async def _search_bing_scraping(self, query, num_results=10):
        """Search Bing for textual content using web scraping."""
        # Construct the Bing search URL for the query
        url = f"https://www.bing.com/search?q={query}&count={num_results}"
        headers = {"User-Agent": "Mozilla/5.0"}
        async with aiohttp.ClientSession(headers=headers) as session:
            try:
                # Fetch the page HTML
                html = await self._fetch_url(session, url)
                if html:
                    return self._parse_text_results(html, source="Bing")
            except Exception as e:
                print(f"Error scraping Bing: {e}")

        return []

    async def _search_google(self, query, num_results=10):
        """Search Google using the Custom Search JSON API."""
        if not self.google_api_key or not self.google_cx:
            print("Google API key or Custom Search Engine ID not provided.")
            return []

        try:
            service = googleapiclient.discovery.build(
                "customsearch", "v1", developerKey=self.google_api_key)
            res = service.cse().list(q=query, cx=self.google_cx, num=num_results).execute()
            return [
                {"title": item["title"],
                    "url": item["link"], "source": "Google"}
                for item in res.get("items", []) if not self._is_duplicate(item["link"])
            ]
        except Exception as e:
            print(f"Error during Google search: {e}")
            return []

    def _is_duplicate(self, url):
        """Check if a URL already exists in the collected data."""
        return any(item["url"] == url for item in self.collected_data)

    def _parse_text_results(self, html, source):
        """Parse textual search results from HTML."""
        soup = BeautifulSoup(html, "lxml")
        results = []
        for link in soup.find_all("a", href=True):
            title = link.get_text(strip=True)
            # Ensure valid titles and links
            if title and "http" in link["href"] and not self._is_duplicate(link["href"]):
                results.append(
                    {"title": title, "url": link["href"], "source": source})
        return results

    async def _run_searches(self, query, num_results):
        """Run multiple searches asynchronously with performance optimization."""
        tasks = [
            self._search_duckduckgo(query, num_results),
        ]

        if self.bing_api_key:
            tasks.append(self._search_bing(query, num_results))
        else:
            print("No bing API key proceeding with bing scraping")
            tasks.append(self._search_bing_scraping(query, num_results))
        if self.google_api_key and self.google_cx:
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
        """Search the internet for related data."""
        start_time = time.time()
        results = asyncio.run(self._run_searches(query, num_results))
        self.collected_data.extend(results)
        self._save_cache()
        print(f"Collected {len(results)} results in {
              time.time() - start_time:.2f} seconds")
        return results

    async def _fetch_full_content(self, result):
        """Fetch and extract full textual content from a single result."""
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                html = await self._fetch_url(session, result["url"])
                if html:
                    soup = BeautifulSoup(html, "lxml")
                    paragraphs = soup.find_all("p")
                    full_text = "\n".join(
                        [p.get_text(strip=True) for p in paragraphs])
                    result["full_text"] = full_text
                    print(f"Extracted content from: {result['url']}")
        except Exception as e:
            print(f"Error fetching content for {result['url']}: {e}")

    def fetch_full_text(self):
        """Download full text for all collected data concurrently."""
        loop = asyncio.get_event_loop()
        tasks = [self._fetch_full_content(result)
                 for result in self.collected_data]
        loop.run_until_complete(asyncio.gather(*tasks))
        self._save_cache()

    def process_results_to_dataset(self):
        """
        Process the collected data and generate question-answer pairs and problem-solution mappings
        for training a generative AI model like GPT-4.
        """
        qa_pairs = []
        problem_solution_pairs = []
        for result in self.collected_data:
            # Step 1: Extract full text if not already done
            if "full_text" not in result:
                # Fetch full text asynchronously
                asyncio.run(self._fetch_full_content(result))
            
            # Step 2: Use QA pipeline to generate question-answer pairs
            if result.get("full_text"):
                text = result["full_text"]
                question = f"What is the main idea of the article at {result['url']}?"
                answer = self._generate_answer(question, text)
                qa_pairs.append({"question": question, "answer": answer})

                # Step 3: Identify problems and solutions from the text (using problem-solution map)
                problem_solution = self._extract_problem_solution(text)
                if problem_solution:
                    problem_solution_pairs.append(problem_solution)
        
        # Step 4: Save the generated data for AI training
        self._save_for_training(qa_pairs, problem_solution_pairs)
        return qa_pairs, problem_solution_pairs

    def _generate_answer(self, question, text):
        """Use the QA pipeline to generate an answer from the full text."""
        try:
            return self.qa_pipeline(question=question, context=text)['answer']
        except Exception as e:
            print(f"Error during question-answering: {e}")
            return "No answer found"

    def _extract_problem_solution(self, text):
        """Extract problem and solution pairs from the text (basic example)."""
        # You can enhance this method with more advanced NLP or pattern matching
        problem_keywords = ["problem", "issue", "challenge"]
        solution_keywords = ["solution", "fix", "resolution"]
        problem = None
        solution = None
        
        # Simple extraction approach (can be expanded with NLP or regex)
        for sentence in text.split("\n"):
            if any(keyword in sentence.lower() for keyword in problem_keywords):
                problem = sentence.strip()
            if any(keyword in sentence.lower() for keyword in solution_keywords):
                solution = sentence.strip()
        
        if problem and solution:
            return {"problem": problem, "solution": solution}
        return None

    def _save_to_dataset(self, qa_pairs, problem_solution_pairs):
        """Save the QA pairs and problem-solution data for AI model training."""
        # Example: Save to JSON file
        training_data = {
            "qa_pairs": qa_pairs,
            "problem_solution_pairs": problem_solution_pairs
        }
        training_data_file = os.path.join(self.save_dir, "processed_data.json")
        with open(training_data_file, "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=4)
        print(f"Training data saved to {training_data_file}")

    def render_data(self):
        """Print collected data."""
        for result in self.collected_data:
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Source: {result['source']}")
            print(f"Full Text: {result.get(
                'full_text', 'Not yet fetched')[:300]}...")
            print("-" * 40)
