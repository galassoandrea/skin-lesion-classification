"""
Downloads the HAM10000 dataset via the Kaggle API.
"""

import os
import zipfile
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")

def download():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.system(
        f"kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p {RAW_DATA_DIR}"
    )
    print("Extracting...")
    for zip_file in RAW_DATA_DIR.glob("*.zip"):
        with zipfile.ZipFile(zip_file, "r") as f:
            f.extractall(RAW_DATA_DIR)
        zip_file.unlink()
    print(f"Download and extraction complete.")

if __name__ == "__main__":
    download()