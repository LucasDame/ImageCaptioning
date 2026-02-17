import kagglehub
import os

if not os.path.exists("./data/Images"):
    # Download latest version
    path = kagglehub.dataset_download("adityajn105/flickr8k", output_dir="./data")

print("Path to dataset files:", path)
