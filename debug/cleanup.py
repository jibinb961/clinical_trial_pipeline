import os
import shutil
from pathlib import Path

# Directories and files to clean
DATA_DIR = Path('data')
RELEASE_DIR = Path('release')
DEBUG_LOG = []

# Helper to remove a file
def remove_file(path):
    if path.exists() and path.is_file():
        path.unlink()
        print(f"Deleted file: {path}")
        DEBUG_LOG.append(f"Deleted file: {path}")

# Helper to remove all files in a directory (optionally skip some)
def remove_files_in_dir(directory, skip_files=None):
    skip_files = skip_files or set()
    for item in directory.iterdir():
        if item.is_file() and item.name not in skip_files:
            remove_file(item)
        elif item.is_dir():
            shutil.rmtree(item)
            print(f"Deleted directory: {item}")
            DEBUG_LOG.append(f"Deleted directory: {item}")

# 1. Remove cache files
drug_cache = DATA_DIR / 'drug_cache.sqlite'
remove_file(drug_cache)

cache_dir = DATA_DIR / 'cache'
if cache_dir.exists():
    remove_files_in_dir(cache_dir)

# 2. Remove processed, raw, and figures data
for subdir in ['processed', 'raw', 'figures']:
    dir_path = DATA_DIR / subdir
    if dir_path.exists():
        remove_files_in_dir(dir_path, skip_files={'.gitkeep'})

# 3. Remove release files except .gitkeep and README.md
if RELEASE_DIR.exists():
    remove_files_in_dir(RELEASE_DIR, skip_files={'.gitkeep', 'README.md'})

print("Cleanup complete.") 