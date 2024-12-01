from data_collector.text_collector import CoeusTextdataCollector
import os
from dotenv import load_dotenv

load_dotenv()
# Specify the directory where the data will be saved
save_dir = "./uprocessed"
# Optional: Provide Google API Key and CX if using Google Search
google_api_key = os.getenv('API_KEY')
google_cx = os.getenv('CX')



# Initialize the collector
collector = CoeusTextdataCollector(save_dir=save_dir, title="Tech Articles", google_api_key=google_api_key, google_cx=google_cx)

# Collect data using a query
query = "Artificial Intelligence in Healthcare"
collected_results = collector.collect_data(query, num_results=5)

# Fetch full text for the collected data
collector.fetch_full_text()

# Process results to generate QA pairs and problem-solution mappings
qa_pairs, problem_solution_pairs = collector.process_results_to_dataset()

# Render the collected data for inspection
collector.render_data()
#collector._save_to_dataset(qa_pairs, problem_solution_pairs)