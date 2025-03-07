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
import signal
import psutil

# Load environment variables
load_dotenv()
API_KEY = os.getenv("THEPORNDB_API_KEY")
if not API_KEY:
    raise ValueError("API key for ThePornDB is not set. Please check your .env file.")

# Global constants
# Define paths to use in the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, 'datasets', 'pornstar_images')
JSON_PATH = os.path.join(BASE_DIR, 'datasets', 'performers_details_data.json')
ALL_performers_list_PATH = r'E:\github repos\porn_ai_analyser\app\get_data\datasets\all_performers.json'

# Update paths to use raw strings
OUTPUT_DIR = r"{}".format(OUTPUT_DIR)
JSON_PATH = r"{}".format(JSON_PATH)

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
# check if the JSON file exists if not create the file
if not os.path.exists(JSON_PATH):
    with open(JSON_PATH, 'w') as file:
        json.dump([], file)

# OTHER CONSTANTS
# set the start page to 0 to start from the last page in the JSON file
START_PAGE = 179
MAX_performers_list = 150000
MAX_RETRIES = 100
RETRY_DELAY = 30
MAX_IMAGE_QUALITY = 85
# Update paths to use raw strings
OUTPUT_DIR = r"{}".format(OUTPUT_DIR)
JSON_PATH = r"{}".format(JSON_PATH)


def bereken_max_threads(ram_per_thread_mb=20):
    # Totale RAM in MB
    total_ram = psutil.virtual_memory().total / (1024 * 1024)

    # Beschikbare RAM in MB
    available_ram = psutil.virtual_memory().available / (1024 * 1024)

    # Aantal beschikbare CPU-cores
    cpu_count = os.cpu_count()

    # Maximale threads gebaseerd op RAM
    max_threads_ram = available_ram // ram_per_thread_mb

    # Echte limiet is het minimum van beide factoren
    max_threads = min(max_threads_ram, cpu_count * 2)  # Voor I/O-taken kun je 2x CPU-cores nemen

    return max_threads


# Define the maximum and minimum number of threads
MAX_THREADS = bereken_max_threads()
MIN_THREADS = 1
current_threads = MAX_THREADS
stop_requested = False


# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    global stop_requested
    log_message("Graceful shutdown requested.")
    stop_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# Utility functions
def log_message(message):
    """Log messages with a consistent format."""
    print(f"--- {message} ---")


def load_existing_performers_list(json_path):
    """Load existing performers_list from JSON file."""
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            data = json.load(file)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # If the data is a dictionary, convert it to a list of performers_list
                performers_list = []
                for page, performer_list in data.items():
                    performers_list.extend(performer_list)
                return performers_list
    return []


def save_performers_list(json_path, performers_list):
    """Save performers_list to JSON file."""
    try:
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(performers_list, file, indent=4, ensure_ascii=False)
        log_message(f"performers_list successfully saved to {json_path}.")
    except Exception as e:
        log_message(f"Error saving performers_list to {json_path}: {e}")
        raise


def performer_exists(performer_id, performers_list):
    """Controleer of een performer al bestaat op basis van ID."""
    return any(p.get('id') == performer_id for p in performers_list)


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


def optimize_image(image_path, quality=MAX_IMAGE_QUALITY):
    """Optimize image for reduced file size."""
    try:
        with Image.open(image_path) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(image_path, optimize=True, quality=quality)
        log_message(f"Image optimized: {image_path}")
    except Exception as e:
        log_message(f"Error optimizing image {image_path}: {e}")


session = requests.Session()  # Create a persistent session

# Create a global cloudscraper instance
scraper = cloudscraper.create_scraper()


def download_image(url, file_path, scraper, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    global current_threads, stop_requested
    if not url.startswith('http'):
        log_message(f"Invalid URL: {url}. Skipping download.")
        return None

    for attempt in range(max_retries):
        if stop_requested:
            log_message("Download interrupted by user.")
            return None
        try:
            with scraper.get(url, stream=True) as response:  # Use scraper for connection reuse
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
                elif response.status_code == 429:
                    log_message(f"Rate limit exceeded (429): {url}. Retrying after delay.")
                    time.sleep(retry_delay)
                else:
                    log_message(
                        f"Attempt {attempt + 1}: Failed to download image from {url}, status code: {response.status_code}, response text: {response.text}.")
        except requests.RequestException as e:
            log_message(f"Attempt {attempt + 1}: Error downloading image: {e}")
            if "WinError 10055" in str(e):
                current_threads = max(current_threads - 1, MIN_THREADS)
                log_message(f"Socket error encountered. Reducing thread count to {current_threads}.")
                wait_for_internet_connection()
        if attempt < max_retries - 1:
            log_message(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            log_message(f"Max retries reached for image. Skipping: {file_path}")
    return None


def download_images(urls, first_name, last_name, scraper, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    """Download images for a performer with retries using multithreading."""
    global current_threads, stop_requested
    performer_folder_name = get_performer_folder_name(first_name, last_name)
    folder_path = os.path.join(OUTPUT_DIR, performer_folder_name)
    os.makedirs(folder_path, exist_ok=True)
    downloaded_paths = []
    image_count = len(urls)

    with concurrent.futures.ThreadPoolExecutor(max_workers=current_threads) as executor:
        futures = []
        for index, url in enumerate(urls):
            if stop_requested:
                log_message("Image download interrupted by user.")
                break
            image_filename = f"{performer_folder_name}_{index + 1}.jpg"
            file_path = os.path.join(folder_path, image_filename)
            if os.path.exists(file_path):
                log_message(
                    f"Image {index + 1} / {image_count} for {first_name} {last_name} already exists, skipping...")
                downloaded_paths.append(file_path)
                continue
            futures.append(executor.submit(download_image, url, file_path, scraper, max_retries, retry_delay))

        for future in concurrent.futures.as_completed(futures):
            if stop_requested:
                log_message("Image download interrupted by user.")
                break
            result = future.result()
            if result:
                downloaded_paths.append(result)
                if current_threads < MAX_THREADS:
                    current_threads += 1
                    log_message(f"Increasing thread count to {current_threads}.")

    return downloaded_paths


def check_internet_connection():
    """Check if the internet connection is available."""
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except requests.RequestException:
        return False


def wait_for_internet_connection():
    """Wait until the internet connection is restored."""
    log_message("Waiting for internet connection to be restored...")
    while not check_internet_connection():
        time.sleep(5)
    log_message("Internet connection restored.")


def get_largest_performer_number(json_path):
    """
    Get the largest 'performer_number' from the performers_details_data.json file.
    Returns 0 if the file is empty or no valid performer_number data is found.
    """
    # Load performers details
    performers_details = load_existing_performers_list(json_path)

    # Check if the file contains valid performers' data and extract 'performer_number'
    if performers_details:
        largest_performer_number = max(
            (performer.get('performer_number', 0) for performer in performers_details),
            default=0
        )
        return largest_performer_number

    # If file is empty or contains no valid performer_number data, return 0
    return 0

performer_counter = get_largest_performer_number(JSON_PATH)

def get_theporndb_details(performer_name, page, scraper, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    """Fetch performer details from ThePornDB with retries."""
    global performer_counter  # Use the global performer counter
    encoded_name = urllib.parse.quote(performer_name.strip())
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
            image_urls = download_images(
                urls=[poster['url'] for poster in performer_data.get('posters', [])],
                first_name=first_name,
                last_name=last_name,
                scraper=scraper
            )
            performer_folder_name = get_performer_folder_name(first_name, last_name)

            # Assign and increment the global performer counter
            performer_number = performer_counter
            log_message(f"Performer {performer_name} assigned number: {performer_number}")

            performer_counter += 1

            return {
                'id': performer_data.get('id', ''),
                'slug': performer_data.get('slug', ''),
                'name': performer_data.get('name', ''),
                'bio': performer_data.get('bio', ''),
                'rating': performer_data.get('rating', ''),
                'is_parent': performer_data.get('extras', {}).get('is_parent', False),
                'gender': performer_data.get('extras', {}).get('gender', ''),
                'birthday': performer_data.get('extras', {}).get('birthday', ''),
                'deathday': performer_data.get('deathday', ''),
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
                'image_urls': image_urls,
                'image_amount': len(image_urls),
                'image_folder': performer_folder_name,  # Add folder name to performer details
                'page': page,  # Add page number to performer details
                'performer_number': performer_number  # Incremented by global counter
            }

        except requests.RequestException as e:
            log_message(f"Attempt {attempt + 1} failed for performer {performer_name}: {e}")
            if attempt < max_retries - 1:
                log_message(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        except Exception as e:
            log_message(f"Unexpected error: {e}")
            break

    log_message(f"Max retries reached for performer {performer_name}. Skipping.")
    return None


def count_performer_images(performer):
    """Count the number of images a performer has."""
    return len(performer.get('image_urls', []))


def performer_images_exist_by_checking_if_every_image_excist(performer_details):
    """Check if all images for a performer already exist in the folder."""
    name_parts = performer_details['name'].split(' ')
    first_name = name_parts[0]
    last_name = name_parts[1] if len(name_parts) > 1 else ''
    performer_folder_name = get_performer_folder_name(first_name, last_name)
    folder_path = os.path.join(OUTPUT_DIR, performer_folder_name)
    if not os.path.exists(folder_path):
        return False
    existing_images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return len(existing_images) >= performer_details.get('image_amount', 0)


def check_and_save_performer(performer_details, performers_list, json_path):
    """Controleer of een performer al bestaat en sla volledige details op."""
    # Controleer op basis van een unieke ID of de performer al in de JSON staat
    existing_performer = next((p for p in performers_list if p.get('id') == performer_details.get('id')), None)

    if not existing_performer:
        log_message(f"Performer {performer_details['name']} wordt toegevoegd aan JSON-bestand.")
        performers_list.append(performer_details)  # Voeg de performer toe aan de lijst
        save_performers_list(json_path, performers_list)  # Sla de bijgewerkte lijst op in het JSON-bestand
    else:
        log_message(f"Performer {performer_details['name']} bestaat al in JSON-bestand. Geen actie ondernomen.")


def count_all_images_in_json(json_path):
    """Count the total number of images for all performers_list in the JSON file."""
    performers_list = load_existing_performers_list(json_path)
    total_images = sum(performer.get('image_amount', 0) for performer in performers_list)
    return total_images


def get_latest_page_from_json(json_path, default_start_page):
    """
    Determine the latest page to start scraping from.
    It uses START_PAGE if explicitly set; otherwise, it uses the highest page in the JSON data.
    """
    performers_list = load_existing_performers_list(json_path)

    # Return default start page if no previous data exists
    if not performers_list:
        return START_PAGE if START_PAGE > 0 else default_start_page

    # If START_PAGE is explicitly set, prioritize it
    if START_PAGE > 0:
        return START_PAGE

    # Dynamically return the highest page in the performers data if START_PAGE is 0
    latest_page = max(performer.get('page', default_start_page) for performer in performers_list)
    return latest_page


def scrape_performers_list_from_json(json_path, max_performers_list=MAX_performers_list):
    """Get the latest page number from the JSON file."""
    if not os.path.exists(json_path):
        log_message(f"JSON file {json_path} does not exist. Using default start page.")
    else:
        log_message(f"Loading existing performers_list from {json_path}.")

    performers_list = load_existing_performers_list(ALL_performers_list_PATH)
    performers_details_list = load_existing_performers_list(JSON_PATH)
    total_existing_performers_list = len(performers_list)
    log_message(f"Found {total_existing_performers_list} existing performers_list.")

    # Determine the starting page
    start_page = get_latest_page_from_json(JSON_PATH, START_PAGE)
    log_message(f"Starting from page: {start_page}")

    # Load performers_list from JSON file
    with open(json_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    if isinstance(json_data, list):
        log_message("JSON data is a list, converting to dictionary format.")
        json_data = {f"page_{i + 1}": page for i, page in enumerate(json_data)}

    processed_count = 0
    total_images_downloaded = 0

    try:
        for page, performer_list in json_data.items():
            page_number = int(page.split('_')[1])
            if page_number < start_page:
                continue

            if stop_requested:
                log_message("Scraping interrupted by user.")
                break

            for performer_data in performer_list:
                if stop_requested:
                    log_message("Scraping interrupted by user.")
                    break

                name = performer_data.get('name', '')
                if not name:
                    continue

                log_message(f"Processing performer: {name}")
                performer_details = get_theporndb_details(name, page_number, scraper)
                if performer_details:
                    name_parts = performer_details['name'].split(' ')
                    first_name = name_parts[0]
                    last_name = name_parts[1] if len(name_parts) > 1 else ''

                    if performer_images_exist_by_checking_if_every_image_excist(performer_details) and performer_exists(
                            performer_details['name'], performers_list):
                        log_message(
                            f"Performer {performer_details['name']} already has the required number of images and exists in JSON. Skipping API call.")
                    else:
                        performer_details['image_urls'] = download_images(performer_details.get('image_urls', []),
                                                                          first_name, last_name, scraper)
                        if performer_details['image_urls']:  # Check if performer has images
                            check_and_save_performer(performer_details, performers_details_list, JSON_PATH)
                            image_count = count_performer_images(performer_details)
                            log_message(f"Performer {performer_details['name']} has {image_count} images.")
                            total_images_downloaded += image_count
                            processed_count += 1

                if processed_count >= max_performers_list:
                    break

        total_images = count_all_images_in_json(JSON_PATH)
        log_message(f"performers_list processed: {processed_count}. Total performers_list now: {len(performers_list)}.")
        log_message(f"Total images downloaded: {total_images}.")
    except KeyboardInterrupt:
        log_message("Scraping interrupted.")
    except Exception as e:
        log_message(f"Error occurred: {e}.")

    return performers_list


# Main entry point
def main():
    log_message("Starting performer scrape. max amount of threads: " + str(MAX_THREADS))
    scrape_performers_list_from_json(ALL_performers_list_PATH)


if __name__ == '__main__':
    main()