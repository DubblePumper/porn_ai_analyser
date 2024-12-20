import cloudscraper
import requests
from bs4 import BeautifulSoup
import json
import os
import time
from PIL import Image
from dotenv import load_dotenv
import urllib.parse

# Load environment variables
load_dotenv()

# Global variables for paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, 'datasets', 'pornstar_images')
JSON_PATH = os.path.join(BASE_DIR, 'datasets', 'performers_data.json')
API_KEY = os.getenv("THEPORNDB_API_KEY")
print(API_KEY)

# Check if API key is loaded
if not API_KEY:
    raise ValueError("API key for ThePornDB is not set. Please check your .env file.")

def log_message(message):
    print(f"--------------------------------------------------------\n{message}\n--------------------------------------------------------")

def get_theporndb_details(performer_name, page=1, per_page=10, retries=5, delay=4):
    encoded_name = urllib.parse.quote(performer_name).replace("_", " ")
    search_url = f"https://api.theporndb.net/performers?q={encoded_name}&page={page}&per_page={per_page}"
    log_message(f"Start zoeken in ThePornDB voor: {performer_name}")
    
    headers = {"accept": "application/json", "Authorization": f"Bearer {API_KEY}"}
    for attempt in range(retries):
        try:
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if not data or 'data' not in data or not data['data']:
                log_message(f"Geen data gevonden voor performer {performer_name}")
                return None

            performer_data = data['data'][0]
            image_urls = [poster['url'] for poster in performer_data.get('posters', [])] if 'posters' in performer_data else []

            performer_details = {
                'id': performer_data.get('id', ''),
                'name': performer_data.get('name', ''),
                'bio': performer_data.get('bio', ''),
                'rating': performer_data.get('rating', ''),
                'gender': performer_data['extras'].get('gender', ''),
                'image_urls': image_urls
            }
            log_message(f"Data succesvol gevonden voor {performer_name}")
            return performer_details

        except requests.exceptions.RequestException as e:
            log_message(f"Fout bij ophalen ThePornDB gegevens voor {performer_name}: {e}")
            if attempt + 1 < retries:
                log_message(f"Opnieuw proberen {attempt + 1} van {retries}")
                time.sleep(delay)
            else:
                return None

def download_images(urls, performer_name):
    if not urls:
        log_message(f"Geen afbeeldingen gevonden voor {performer_name}")
        return []

    folder_path = os.path.join(OUTPUT_DIR, performer_name)
    if not os.path.exists(folder_path):
        log_message(f"Geen folder gevonden voor {performer_name}, aanmaken...")
        os.makedirs(folder_path)

    downloaded_paths = []
    log_message(f"Downloaden afbeeldingen voor {performer_name} ({len(urls)} afbeeldingen gevonden)")
    for index, url in enumerate(urls):
        file_path = os.path.join(folder_path, f"{performer_name}_{index + 1}.jpg")
        if os.path.exists(file_path):
            log_message(f"Afbeelding {index + 1} bestaat al, overslaan...")
            downloaded_paths.append(file_path)
            continue

        try:
            scraper = cloudscraper.create_scraper()
            response = scraper.get(url, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                downloaded_paths.append(file_path)
                log_message(f"Afbeelding {index + 1} succesvol gedownload.")
            else:
                log_message(f"Fout bij downloaden afbeelding {index + 1}, statuscode: {response.status_code}")
        except Exception as e:
            log_message(f"Fout bij downloaden afbeelding {index + 1}: {e}")

    return downloaded_paths

def get_pornhub_performers(max_performers=15000000):
    performers = []
    base_url = "https://nl.pornhub.com/pornstars?performerType=pornstar&page="
    scraper = cloudscraper.create_scraper()

    page = 1
    while len(performers) < max_performers:
        log_message(f"Pagina {page} - Performer {len(performers) + 1} van {max_performers}")
        url = f"{base_url}{page}"
        response = scraper.get(url)

        soup = BeautifulSoup(response.content, 'html.parser')
        performer_cards = soup.find_all('li', class_='pornstarLi performerCard')

        if not performer_cards:
            log_message(f"Geen performer kaarten gevonden op pagina {page}")
            break

        for card in performer_cards:
            name_tag = card.find('span', class_='pornStarName performerCardName')
            if name_tag:
                name = name_tag.get_text()
                formatted_name = "_".join(name.split())
                performer_details = get_theporndb_details(formatted_name)
                if performer_details:
                    performers.append(performer_details)
                    downloaded_images = download_images(performer_details.get('image_urls', []), formatted_name)
                    performer_details['image_urls'] = downloaded_images
                    log_message(f"Alle afbeeldingen voor {formatted_name} succesvol gedownload.")

        page += 1
        time.sleep(2)

    return performers

def main():
    log_message("Start scraping performers")
    get_pornhub_performers()

if __name__ == '__main__':
    main()