import os
from data_collector.image_collector import CoeusImagesdataCollector
from dotenv import load_dotenv

load_dotenv()
# Specify the directory where the data will be saved
save_dir = "./uprocessed"
# Optional: Provide Google API Key and CX if using Google Search
google_api_key = os.getenv('API_KEY')
google_cx = os.getenv('CX')
print(google_api_key, google_cx)

# Initialize the collector
collector = CoeusImagesdataCollector(
    save_dir=save_dir,
    title="toyota",  #
    google_api_key=google_api_key,
    google_cx=google_cx
)

# Search for images
query = "toyota engine"
num_results = 5
collector.collect_data(query, num_results)
# Display results
collector.render_data()
# Download images
collector.download_images()
