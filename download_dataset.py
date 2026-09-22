import subprocess
import sys
from pathlib import Path

DATASET = "sergiosaharovskiy/uscoins"
DATASETS_DIR = Path("datasets")
DATASETS_DIR.mkdir(exist_ok=True)

subprocess.run(
    [
        sys.executable,
        "-m",
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
    