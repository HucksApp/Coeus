from data_collector.pdf_collector import CoeusPdfCollector
from data_collector.pdf_processor import CoeusPdfProcessor
import os
from dotenv import load_dotenv

load_dotenv()
# Specify the directory where the data will be saved
save_dir = "./uprocessed"
# Optional: Provide Google API Key and CX if using Google Search
google_api_key = os.getenv('API_KEY')
google_cx = os.getenv('CX')

# Initialize the collector
collector = CoeusPdfCollector(
    save_dir=save_dir,
    title="manual",
    google_api_key=google_api_key,
    google_cx=google_cx,
    cache_size=5000  # Optional: Adjust cache size
)

results = collector.collect_data(query="Toyota service", num_results=100)
collector.render_data()
collector.download_pdfs()


# processor = CoeusPdfProcessor(
#     pdf_path=f"{save_dir}/manual/headfirst_js.pdf",
#     save_dir=f"{save_dir}/extracted",
#     title="manual",
#     to_dataset=True,
#     use_spacy=False
# )
#processor.process_directory(f"{save_dir}/manual")