import cloudscraper
from bs4 import BeautifulSoup
import csv
import os
from PIL import Image
import concurrent.futures
import re

def clean_performer_name(performer_name):
    # Remove unnecessary words like "Official"
    return re.sub(r'\b(Official|official)\b', '', performer_name).strip()

def get_performer_image(performer_name):
    formatted_name = performer_name.replace(' ', '+')

    # Create a cloudscraper session to bypass Cloudflare protection
    scraper = cloudscraper.create_scraper()

    # Set headers to simulate a request from a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept-Language': 'nl-NL,nl;q=0.9',
    }

    # Prepare the IAFD search URL for the performer
    search_url = f"https://www.iafd.com/results.asp?searchtype=comprehensive&searchstring={formatted_name}"

    # Send the request using the scraper
    response = scraper.get(search_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Find all links that may lead to performer pages
    performer_links = soup.find_all('a', href=True)

    for link in performer_links:
        if '/person.rme/id=' in link['href']:
            performer_url = 'https://www.iafd.com' + link['href']
            print(f"Found Performer URL: {performer_url}")

            # Now, get the details from the performer's page
            performer_response = scraper.get(performer_url, headers=headers)
            performer_soup = BeautifulSoup(performer_response.content, 'html.parser')

            headshot_div = performer_soup.find('div', id='headshot')
            if headshot_div:
                img_tag = headshot_div.find('img')
                if img_tag and 'src' in img_tag.attrs:
                    image_url = img_tag['src']
                    return image_url
                else:
                    return "No image found"
            else:
                return "No image section found"

    return "Performer not found"

def download_and_optimize_image(url, save_path):
    try:
        scraper = cloudscraper.create_scraper()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Referer': 'https://www.iafd.com/',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        response = scraper.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as out_file:
                for chunk in response.iter_content(1024):
                    out_file.write(chunk)
            # Optimize the image
            with Image.open(save_path) as img:
                img = img.convert("RGB")
                img.save(save_path, "JPEG", quality=85, optimize=True)
            print(f"Image downloaded and optimized: {save_path}")
        else:
            print(f"Failed to download image, status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading or optimizing image: {e}")

def process_performer(performer_name, images_dir, counter, not_found_list):
    print(f"Processing {performer_name}")
    image_path = os.path.join(images_dir, f"{performer_name.replace(' ', '_')}.jpg")
    if os.path.exists(image_path):
        print(f"Image already exists for {performer_name}, skipping download.")
        return
    image_url = get_performer_image(performer_name)
    if image_url == "Performer not found":
        cleaned_name = clean_performer_name(performer_name)
        if cleaned_name != performer_name:
            print(f"Retrying with cleaned name: {cleaned_name}")
            image_url = get_performer_image(cleaned_name)
    if image_url and image_url != "Performer not found":
        download_and_optimize_image(image_url, image_path)
        print(f"Downloaded and optimized image for {performer_name}")
        counter['count'] += 1
        print(f"Total images processed: {counter['count']}")
    else:
        print(f"No image found for {performer_name}")
        not_found_list.append(performer_name)

def process_csv_and_download_images(csv_path, images_dir):
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    counter = {'count': 0}
    not_found_list = []

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        performer_names = [row['Name'] for row in reader]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_performer, name, images_dir, counter, not_found_list) for name in performer_names]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    print(f"Total images processed: {counter['count']}")
    if not_found_list:
        print("Performers without images:")
        for name in not_found_list:
            print(name)

# Example usage
csv_path = 'app/dataset/final.csv'
images_dir = 'app/input_images'
process_csv_and_download_images(csv_path, images_dir)