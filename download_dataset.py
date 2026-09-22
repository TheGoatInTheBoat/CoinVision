import subprocess
from pathlib import Path

# Kaggle dataset
DATASET = "sergiosaharovskiy/uscoins"

# Folder where datasets will be stored
DATASETS_DIR = Path("datasets")

# Create datasets folder if it doesn't exist
DATASETS_DIR.mkdir(exist_ok=True)

print("Downloading US Coins dataset...")

subprocess.run(
    [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(DATASETS_DIR),
        "--unzip",
    ],
    check=True,
)

print()
print("Dataset downloaded and unzipped successfully!")
print(f"Location: {DATASETS_DIR.resolve()}")