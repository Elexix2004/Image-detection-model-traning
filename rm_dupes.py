import os
import shutil
from PIL import Image
import imagehash

# Define paths using raw strings to avoid Unicode errors
source_folder = r"path here"           # <-- Put your source images here
destination_folder = r"path here"  # <-- Unique images will be copied here

os.makedirs(destination_folder, exist_ok=True)

hashes = set()

for filename in os.listdir(source_folder):
    file_path = os.path.join(source_folder, filename)
    
    try:
        with Image.open(file_path) as img:
            img_hash = imagehash.average_hash(img)
            if img_hash not in hashes:
                hashes.add(img_hash)
                shutil.copy(file_path, os.path.join(destination_folder, filename))
    except Exception as e:
        print(f"Skipping {filename}: {e}")
