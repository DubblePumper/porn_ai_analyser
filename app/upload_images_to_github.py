import os
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to the directory containing images
image_directory = r"E:\github repos\porn_ai_analyser\app\datasets\pornstar_images"

# Ensure the directory path is correctly formatted
if not os.path.exists(image_directory):
    raise ValueError(f"The directory {image_directory} does not exist.")

# Change to your repository's directory
repo_directory = r"E:\github repos\porn_ai_analyser"
os.chdir(repo_directory)

# Get the list of all .jpg image files, including those in subdirectories
all_files = []
for root, dirs, files in os.walk(image_directory):
    for file in files:
        if file.lower().endswith('.jpg') and os.path.isfile(os.path.join(root, file)):
            all_files.append(os.path.join(root, file))

logging.info(f"Total {len(all_files)} .jpg files found.")

# Function to calculate the total size of files in a batch
def get_batch_size(files):
    return sum(os.path.getsize(file) for file in files)

# Initialize a lock for git operations
git_lock = Lock()

# Function to add, commit, and push files with logging
def process_batch(batch_files, batch_number):
    with git_lock:
        logging.info(f"Starting to process batch {batch_number}")
        
        # Add all files in the batch at once
        subprocess.run(['git', 'add'] + batch_files)
        logging.info(f"Added {len(batch_files)} files in batch {batch_number}")
        
        commit_message = f'Adding images batch {batch_number}'
        logging.info(f"Committing batch {batch_number}")
        subprocess.run(['git', 'commit', '-m', commit_message])
        logging.info(f"Committed batch {batch_number}")
        
        try:
            logging.info(f"Pushing batch {batch_number}")
            subprocess.run(['git', 'push'], check=True)
            logging.info(f"Pushed batch {batch_number}")
            logging.info(f"Pruning unreachable loose objects after pushing batch {batch_number}")
            subprocess.run(['git', 'prune'], check=True)
            logging.info(f"Pruned unreachable loose objects after pushing batch {batch_number}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to push batch {batch_number}: {e}")
            if "This repository is over its data quota" in e.output.decode():
                logging.error("LFS quota exceeded. Please purchase more data packs to restore access.")
                return False
        logging.info(f"Finished processing batch {batch_number}")
    return True

# Function to split a batch into smaller chunks if it exceeds the maximum size
def split_batch(batch_files, max_size_bytes):
    split_batches = []
    current_batch = []
    current_size = 0

    for file in batch_files:
        file_size = os.path.getsize(file)
        if current_size + file_size > max_size_bytes:
            split_batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append(file)
        current_size += file_size

    if current_batch:
        split_batches.append(current_batch)

    return split_batches

# Commit and push in batches of 1000 files or 100 MB, whichever is smaller
batch_size = 1000
max_batch_size_bytes = 100 * 1024 * 1024  # 100 MB

with ThreadPoolExecutor() as executor:
    batch_number = 1
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]
        if get_batch_size(batch_files) > max_batch_size_bytes:
            logging.warning(f"Batch {batch_number} exceeds 100 MB, splitting batch.")
            split_batches = split_batch(batch_files, max_batch_size_bytes)
            for split_batch_files in split_batches:
                if not process_batch(split_batch_files, batch_number):
                    break
                batch_number += 1
        else:
            if not process_batch(batch_files, batch_number):
                break
            batch_number += 1
