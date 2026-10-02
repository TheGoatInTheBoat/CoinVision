from __future__ import annotations

import csv
import re
from pathlib import Path


# CoinVision paths

REPO_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = REPO_ROOT / "datasets" / "US Coins - Kaggle"
CSV_PATH = DATASET_ROOT / "dataset.csv"
OUTPUT_PATH = DATASET_ROOT / "metadata.csv"


# Dataset constants

SOURCE = "https://www.kaggle.com/datasets/sergiosaharovskiy/uscoins"
LICENSE = "CC0: Public Domain"

CATEGORY_INFO = {
    "Jefferson Nickels, 1938-Date": {
        "denomination": "Nickel",
        "series": "Jefferson Nickel",
        "code": "JN",
    },
    "Lincoln Cents, 1909-Date": {
        "denomination": "Penny",
        "series": "Lincoln Cent",
        "code": "LC",
    },
    "Washington Quarters, 1932-1998": {
        "denomination": "Quarter",
        "series": "Washington Quarter",
        "code": "WQ",
    },
}


# Side extraction

def extract_side(filename: str) -> str:
    """
    Extract Obverse/Reverse when explicitly present in the source filename.

    Returns an empty string when the side is not provided.
    """

    stem = Path(filename).stem

    match = re.search(
        r"(?:^|\s)(Obverse|Reverse)(?:\s*)$",
        stem,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).capitalize()

    return ""


# Metadata generation

def build_metadata() -> None:
    rows = []

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)

        for image_number, row in enumerate(reader, start=1):
            # The original Kaggle CSV contains:
            # category, filename
            category = row[0].strip()
            filename = row[1].strip()

            category_info = CATEGORY_INFO[category]

            relative_image_path = (
                Path("coins") / category / filename
            ).as_posix()


            side = extract_side(filename)

            rows.append(
                {
                    "image_id": f"img_{image_number:06d}",
                    "country": "USA",
                    "denomination": category_info["denomination"],
                    "series": category_info["series"],
                    "year": "",
                    "mint": "",
                    "side": side,
                    "image_path": relative_image_path,
                    "source": SOURCE,
                    "license": LICENSE,
                }
            )

    fieldnames = [
        "image_id",
        "country",
        "denomination",
        "series",
        "year",
        "mint",
        "side",
        "image_path",
        "source",
        "license",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Created metadata file: {OUTPUT_PATH}")
    print(f"Rows written: {len(rows):,}")


# Entry point

if __name__ == "__main__":
    build_metadata()
