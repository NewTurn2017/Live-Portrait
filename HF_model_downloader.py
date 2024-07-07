from huggingface_hub import HfApi
import os
import requests
import time
from tqdm import tqdm

# Repository details
repo_id = "OwlMaster/LivePortrait"
repo_type = "model"

# Initialize Hugging Face API
api = HfApi()

# Get list of files in the repository
files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

# Function to ensure directory exists
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to format file size
def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

# Download each file
for file in files:
    # Construct the download URL
    url = f"https://huggingface.co/{repo_id}/resolve/main/{file}"
    
    # Prepare the local file path
    local_path = os.path.join(os.getcwd(), file)
    
    # Ensure the directory exists
    ensure_dir(local_path)
    
    # Download the file with progress bar
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        with open(local_path, 'wb') as f, tqdm(
            desc=file,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            start_time = time.time()
            downloaded = 0
            for data in response.iter_content(block_size):
                size = f.write(data)
                downloaded += size
                progress_bar.update(size)
            
            end_time = time.time()
            duration = end_time - start_time
            speed = downloaded / duration if duration > 0 else 0
            
        print(f"Downloaded: {local_path}")
        print(f"Size: {format_size(total_size)}")
        print(f"Speed: {format_size(speed)}/s")
        print()
    else:
        print(f"Failed to download: {file}")

print("Download complete.")