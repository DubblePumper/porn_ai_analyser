import os
import json
import signal
from bs4 import BeautifulSoup
import cloudscraper

# Constants
BASE_URL_ALL = "https://nl.pornhub.com/pornstars?page="
BASE_URL_TYPE = "https://nl.pornhub.com/pornstars?performerType=pornstar&page="
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'all_performers.json')
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Utility functions
def log_message(message):
    """Log messages with a consistent format."""
    print(f"--- {message} ---")

def save_data(json_path, data):
    """Save data to JSON file."""
    try:
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        log_message(f"Data successfully saved to {json_path}.")
    except Exception as e:
        log_message(f"Error saving data to {json_path}: {e}")
        raise

def performer_exists(name, performers):
    """Check if the performer already exists in the data."""
    exists = any(p['name'] == name for p in performers)
    if exists:
        log_message(f"Duplicate performer detected: {name}. Skipping...")
    return exists

# Signal handler for graceful exit
def signal_handler(sig, frame):
    log_message("=== Scraping process interrupted. Saving data and exiting... ===")
    save_data(OUTPUT_FILE, performers_by_page)
    log_message("=== Data saved. Exiting now. ===")
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Scraper function
def scrape_performers(base_url, start_page, end_page, existing_performers):
    def create_scraper():
        return cloudscraper.create_scraper(browser={
            'browser': 'chrome',
            'platform': 'windows',
            'mobile': False
        })

    scraper = create_scraper()
    all_performers = existing_performers.copy()
    log_message(f"Starting scrape from {base_url}{start_page} to {base_url}{end_page}.")

    global performers_by_page
    performers_by_page = {}
    for page in range(start_page, end_page + 1):
        log_message(f"Processing page {page}.")
        try:
            response = scraper.get(f"{base_url}{page}")
            if response.status_code != 200:
                log_message(f"Failed to fetch page {page}. HTTP status code: {response.status_code}")
                continue

            soup = BeautifulSoup(response.content, 'html.parser')
            performer_cards = soup.find_all('li', class_=['pornstarLi performerCard', 'modelLi performerCard', 'pornstarWrap performerCard'])
            page_performers = []
            if performer_cards:
                for card in performer_cards:
                    name_tag = card.find('span', class_=['pornStarName performerCardName', 'modelName performerCardName'])
                    if name_tag:
                        name = name_tag.get_text().strip()
                        if not performer_exists(name, all_performers):
                            all_performers.append({"name": name})
                            page_performers.append({"name": name})
                            log_message(f"New performer added: {name}.")
            else:
                log_message(f"No performer cards found on page {page}. Changing user agent and retrying...")
                scraper = create_scraper()
                continue  # Retry the same page with a new user agent

            # Save performers for the current page
            performers_by_page[f"page_{page}"] = page_performers
            save_data(OUTPUT_FILE, performers_by_page)
        except Exception as e:
            log_message(f"Error processing page {page}: {e}")

        log_message(f"Finished processing page {page}. Total performers collected so far: {len(all_performers)}.")

    return performers_by_page

# Execution
if __name__ == "__main__":
    log_message("=== Starting the scraping process ===")
    # Step 1: Scrape all performers
    log_message("Step 1: Scraping all performers from general list...")
    performers_by_page = scrape_performers(BASE_URL_ALL, start_page=1, end_page=1695, existing_performers=[])

    # Step 2: Scrape performers of type 'pornstar'
    log_message("Step 2: Scraping performers with type 'pornstar'...")
    performers_by_page.update(scrape_performers(BASE_URL_TYPE, start_page=1, end_page=333, existing_performers=[]))

    # Step 3: Save all performers to a single file
    log_message("Step 3: Saving all performers to a single JSON file...")
    save_data(OUTPUT_FILE, performers_by_page)
    log_message("=== Scraping process completed successfully ===")