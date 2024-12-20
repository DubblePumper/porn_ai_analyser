import random
import os
import yt_dlp  # Using yt-dlp to download videos
from pornhub_api import PornhubApi
import json  # Import json module
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

# Initialize PornhubAPI for searching
api = PornhubApi()

# Fetch 5 random tags (you can customize this)
tags = random.sample(api.video.tags("f").tags, 5)

# Pick a random category
category = random.choice(api.video.categories().categories)

# Search for videos with random tags and category
result = api.search.search(ordering="mostviewed", tags=tags, category=category)

# Path to the input_videos folder
output_folder = './app/input_videos/'

# Create the folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Path to the performers_data.json file
performers_data_path = './app/performers_data.json'

# Load existing performers data if the file exists
if os.path.exists(performers_data_path):
    with open(performers_data_path, 'r') as f:
        performers_data = json.load(f)
else:
    performers_data = {}

def get_most_similar_name(search_name, names):
    highest_ratio = 0
    most_similar_name = None
    for name in names:
        ratio = SequenceMatcher(None, search_name, name).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            most_similar_name = name
    return most_similar_name

def search_iafd_for_performer(performer_name):
    search_url = f"https://www.iafd.com/results.asp?searchtype=comprehensive&searchstring={performer_name.replace(' ', '+')}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    performers_table = soup.find('table', id='tblFem')
    if not performers_table:
        print(f"No performer table found: {performer_name}")
        return None

    performers = []
    for row in performers_table.find_all('tr')[1:]:  # Skip the header row
        cols = row.find_all('td')
        if len(cols) > 1:
            name = cols[1].text.strip()
            url = cols[1].find('a')['href']
            performers.append((name, url))
    return performers

def get_performer_details(performer_url):
    response = requests.get(performer_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    details = {}
    vitalbox = soup.find('div', id='vitalbox')
    if vitalbox:
        for bioheading in vitalbox.find_all('p', class_='bioheading'):
            biodata = bioheading.find_next_sibling('p', class_='biodata')
            if biodata:
                details[bioheading.text.strip()] = biodata.text.strip()
    return details

# Loop through the search results
for vid in result.videos:  # Accessing the videos attribute correctly
    # Print the video details to understand the structure

    # Assuming vid is a dictionary, access its attributes
    video_id = vid.video_id  # Corrected attribute access
    duration = vid.duration
    title = vid.title
    url = vid.url
    performers = vid.performers  # Assuming performers attribute exists

    # Print the video_data to understand its structure
    print("Video data:")
    print(f"ID: {video_id}, Duration: {duration}, Title: {title}, URL: {url}, Performers: {performers}")

    # Convert duration to seconds if necessary
    duration_seconds = int(duration.split(':')[0]) * 60 + int(duration.split(':')[1])

    # Check if the duration is between 10 minutes (600 seconds) and 30 minutes (1800 seconds)
    if 600 <= duration_seconds <= 1800:
        print(f"Found video: {title} ({duration_seconds // 60} minutes)")
        print(f"URL: {url}")

        # Define the path for saving the video
        video_path = os.path.join(output_folder, f"{title}.mp4")

        try:
            # Use yt-dlp to download the video
            ydl_opts = {
                'outtmpl': video_path,
                'format': 'bestvideo+bestaudio/best',
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            print(f"Video '{title}' downloaded successfully at {video_path}!")

            # Update performers data
            for performer in performers:
                performer_name = performer.name  # Assuming performer has a name attribute
                if performer_name not in performers_data:
                    performers_data[performer_name] = []
                performer_details = get_performer_details(performer.url)  # Assuming performer has a url attribute

                # Find the most similar name in the existing performers data
                similar_name = get_most_similar_name(performer_name, performers_data.keys())
                if similar_name:
                    performer_name = similar_name

                # Search IAFD for performer details
                iafd_performers = search_iafd_for_performer(performer_name)
                if iafd_performers:
                    best_match = get_most_similar_name(performer_name, [p[0] for p in iafd_performers])
                    if best_match:
                        performer_url = next(p[1] for p in iafd_performers if p[0] == best_match)
                        performer_details = get_performer_details(performer_url)

                performers_data[performer_name].append({
                    'video_id': video_id,
                    'title': title,
                    'duration': duration,
                    'url': url,
                    'details': performer_details
                })

            # Save the updated performers data to the JSON file
            with open(performers_data_path, 'w') as f:
                json.dump(performers_data, f, indent=4)

        except Exception as e:
            print(f"Failed to download video '{title}': {e}")