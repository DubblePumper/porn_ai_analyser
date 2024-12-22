import os
import time
import json
import requests
import cloudscraper
from bs4 import BeautifulSoup
from PIL import Image
from dotenv import load_dotenv
import urllib.parse
import concurrent.futures

# Load environment variables
load_dotenv()
API_KEY = os.getenv("THEPORNDB_API_KEY")
if not API_KEY:
    raise ValueError("API key for ThePornDB is not set. Please check your .env file.")

# Global constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, 'datasets', 'pornstar_images')
JSON_PATH = os.path.join(BASE_DIR, 'datasets', 'performers_data.json')

# Utility functions
def log_message(message):
    """Log messages with a consistent format."""
    print(f"--- {message} ---")

def load_existing_performers(json_path):
    """Load existing performers from JSON file."""
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            return json.load(file)
    return []

def save_performers(json_path, performers):
    """Save performers to JSON file."""
    try:
        with open(json_path, 'w') as file:
            json.dump(performers, file, indent=4)
        log_message("Performers successfully saved.")
    except Exception as e:
        log_message(f"Error saving performers: {e}")
        raise

def performer_exists(performer_id, performers):
    """Check if a performer already exists by ID."""
    return any(p.get('id') == performer_id for p in performers if isinstance(p, dict))

def sanitize_name(name):
    """Sanitize the name to ensure it's valid for file systems."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name.strip()

def get_performer_folder_name(first_name, last_name):
    """Generate the folder name using the first and last name."""
    sanitized_first_name = sanitize_name(first_name)
    sanitized_last_name = sanitize_name(last_name)
    return f"{sanitized_first_name}_{sanitized_last_name}"

def optimize_image(image_path, quality=85):
    """Optimize image for reduced file size."""
    try:
        with Image.open(image_path) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(image_path, optimize=True, quality=quality)
        log_message(f"Image optimized: {image_path}")
    except Exception as e:
        log_message(f"Error optimizing image {image_path}: {e}")

def download_image(url, file_path, max_retries=10, retry_delay=10):
    """Download a single image with retries."""
    for attempt in range(max_retries):
        try:
            scraper = cloudscraper.create_scraper()
            response = scraper.get(url, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                optimize_image(file_path)
                log_message(f"Image downloaded: {file_path}")
                return file_path
            elif response.status_code == 404:
                log_message(f"Image not found (404): {url}. Skipping download.")
                return None
            else:
                log_message(f"Attempt {attempt + 1}: Failed to download image from {url}, status code: {response.status_code}, response text: {response.text}.")
        except Exception as e:
            log_message(f"Attempt {attempt + 1}: Error downloading image: {e}")
        if attempt < max_retries - 1:
            log_message(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            log_message(f"Max retries reached for image. Skipping: {file_path}")
    return None

def download_images(urls, first_name, last_name, max_retries=10, retry_delay=10):
    """Download images for a performer with retries using multithreading."""
    performer_folder_name = get_performer_folder_name(first_name, last_name)
    folder_path = os.path.join(OUTPUT_DIR, performer_folder_name)
    os.makedirs(folder_path, exist_ok=True)
    downloaded_paths = []
    image_count = len(urls)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for index, url in enumerate(urls):
            image_filename = f"{performer_folder_name}_{index + 1}.jpg"
            file_path = os.path.join(folder_path, image_filename)
            if os.path.exists(file_path):
                log_message(f"Image {index + 1} / {image_count} for {first_name} {last_name} already exists, skipping...")
                downloaded_paths.append(file_path)
                continue
            futures.append(executor.submit(download_image, url, file_path, max_retries, retry_delay))

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                downloaded_paths.append(result)

    return downloaded_paths

def get_theporndb_details(performer_name, max_retries=10, retry_delay=10):
    """Fetch performer details from ThePornDB with retries."""
    encoded_name = urllib.parse.quote(performer_name)
    search_url = f"https://api.theporndb.net/performers?q={encoded_name}"
    headers = {"accept": "application/json", "Authorization": f"Bearer {API_KEY}"}

    for attempt in range(max_retries):
        try:
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()
            data = response.json()
            if not data or 'data' not in data or not data['data']:
                log_message(f"No data found for performer {performer_name}.")
                return None

            performer_data = data['data'][0]
            name_parts = performer_data.get('name', '').split(' ')
            first_name = name_parts[0]
            last_name = name_parts[1] if len(name_parts) > 1 else ''
            return {
                    'id': performer_data.get('id', ''),
        'slug': performer_data.get('slug', ''),
        'name': performer_data.get('name', ''),
        'bio': performer_data.get('bio', ''),
        'rating': performer_data.get('rating', ''),
        'is_parent': performer_data.get('extras', {}).get('is_parent', False),
        'gender': performer_data.get('extras', {}).get('gender', ''),
        'birthday': performer_data.get('extras', {}).get('birthday', ''),
        'deathday': performer_data.get('extras', {}).get('deathday', ''),
        'birthplace': performer_data.get('extras', {}).get('birthplace', ''),
        'ethnicity': performer_data.get('extras', {}).get('ethnicity', ''),
        'nationality': performer_data.get('extras', {}).get('nationality', ''),
        'hair_color': performer_data.get('extras', {}).get('hair_colour', ''),
        'eye_color': performer_data.get('extras', {}).get('eye_colour', ''),
        'height': performer_data.get('extras', {}).get('height', ''),
        'weight': performer_data.get('extras', {}).get('weight', ''),
        'measurements': performer_data.get('extras', {}).get('measurements', ''),
        'waist_size': performer_data.get('extras', {}).get('waist', ''),
        'hip_size': performer_data.get('extras', {}).get('hips', ''),
        'cup_size': performer_data.get('extras', {}).get('cupsize', ''),
        'tattoos': performer_data.get('extras', {}).get('tattoos', ''),
        'piercings': performer_data.get('extras', {}).get('piercings', ''),
        'fake_boobs': performer_data.get('extras', {}).get('fake_boobs', False),
        'same_sex_only': performer_data.get('extras', {}).get('same_sex_only', False),
        'career_start_year': performer_data.get('extras', {}).get('career_start_year', ''),
        'career_end_year': performer_data.get('extras', {}).get('career_end_year', ''),
        'image_urls': download_images(
            urls=[poster['url'] for poster in performer_data.get('posters', [])],
            first_name=first_name,
            last_name=last_name,
        )
            }
        except requests.RequestException as e:
            log_message(f"Attempt {attempt + 1} failed for performer {performer_name}: {e}")
            if attempt < max_retries - 1:
                log_message(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                log_message(f"Max retries reached for {performer_name}. Skipping.")
                return None

def count_performer_images(performer):
    """Count the number of images a performer has."""
    return len(performer.get('image_urls', []))

def scrape_performers(max_performers=150, start_page=2):
    """Scrape performers from Pornhub."""
    performers = load_existing_performers(JSON_PATH)
    total_existing_performers = len(performers)
    log_message(f"Found {total_existing_performers} existing performers.")

    base_url = "https://nl.pornhub.com/pornstars?performerType=pornstar&page="
    scraper = cloudscraper.create_scraper()
    processed_count = 0

    for page in range(start_page, start_page + max_performers):
        log_message(f"Processing page {page}.")
        response = scraper.get(f"{base_url}{page}")
        soup = BeautifulSoup(response.content, 'html.parser')
        performer_cards = soup.find_all('li', class_='pornstarLi performerCard')

        if not performer_cards:
            log_message(f"No performers found on page {page}. Ending scrape.")
            break

        for card in performer_cards:
            name_tag = card.find('span', class_='pornStarName performerCardName')
            if name_tag:
                name = name_tag.get_text()
                if any(p.get('name') == name for p in performers if isinstance(p, dict)):
                    log_message(f"Page {page} | Performer {name} already exists in JSON, skipping.")
                    continue
                else:
                    log_message(f"Page {page} | Performer {name} does not exist in JSON, processing.")

                performer_details = get_theporndb_details(name)
                if performer_details and not performer_exists(performer_details['id'], performers):
                    name_parts = performer_details['name'].split(' ')
                    first_name = name_parts[0]
                    last_name = name_parts[1] if len(name_parts) > 1 else ''
                    performer_details['image_urls'] = download_images(performer_details.get('image_urls', []), first_name, last_name)
                    if performer_details['image_urls']:  # Check if performer has images
                        performers.append(performer_details)
                        save_performers(JSON_PATH, performers)
                        image_count = count_performer_images(performer_details)
                        log_message(f"Performer {performer_details['name']} has {image_count} images.")
                        processed_count += 1

        total_remaining = max_performers - processed_count
        log_message(f"Page {page} processed. Performers scraped: {processed_count}, Remaining: {total_remaining}.")
        time.sleep(2)

    log_message(f"Scraping completed. Total performers processed: {processed_count}. Total performers now: {len(performers)}.")
    return performers

# Main entry point
def main():
    log_message("Starting performer scrape.")
    scrape_performers()

if __name__ == '__main__':
    main()