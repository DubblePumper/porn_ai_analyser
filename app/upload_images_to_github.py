import os
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to the directory containing images
image_directory = r"Y:\github repo's\video_ai_analyser\app\datasets\pornstar_images"

# Ensure the directory path is correctly formatted
if not os.path.exists(image_directory):
    raise ValueError(f"The directory {image_directory} does not exist.")

# Change to your repository's directory
repo_directory = r"Y:\github repo's\video_ai_analyser"
os.chdir(repo_directory)

# Get the list of all image files, including those in subdirectories
all_files = []
for root, dirs, files in os.walk(image_directory):
    for file in files:
        if os.path.isfile(os.path.join(root, file)):
            all_files.append(os.path.join(root, file))

logging.info(f"Total {len(all_files)} files found.")

# Function to calculate the total size of files in a batch
def get_batch_size(files):
    return sum(os.path.getsize(file) for file in files)

# Function to add a file to git
def add_file_to_git(file):
    subprocess.run(['git', 'add', file])
    logging.info(f"Added file: {file}")

# Function to commit and push a batch of files
def commit_and_push_batch(batch_files, batch_number):
    commit_message = f'Adding images batch {batch_number}'
    subprocess.run(['git', 'commit', '-m', commit_message])
    logging.info(f"Committed batch {batch_number}")
    
    try:
        subprocess.run(['git', 'push'], check=True)
        logging.info(f"Pushed batch {batch_number}")
        subprocess.run(['git', 'prune'], check=True)
        logging.info(f"Pruned unreachable loose objects after pushing batch {batch_number}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to push batch {batch_number}: {e}")
        if "This repository is over its data quota" in e.output.decode():
            logging.error("LFS quota exceeded. Please purchase more data packs to restore access.")
            return False
    return True

# Commit and push in batches of 1000 files or 100 MB, whichever is smaller
batch_size = 1000
max_batch_size_bytes = 100 * 1024 * 1024  # 100 MB

with ThreadPoolExecutor() as executor:
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]
        if get_batch_size(batch_files) > max_batch_size_bytes:
            logging.warning(f"Batch {i//batch_size + 1} exceeds 100 MB, reducing batch size.")
            while get_batch_size(batch_files) > max_batch_size_bytes:
                batch_files.pop()
