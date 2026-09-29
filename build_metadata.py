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


# Specimen ID

def build_specimen_id(filename: str, series_code: str) -> str:
    """
    Convert the source filename's coin identifier into a consistent
    CoinVision specimen_id.

    Examples:
        Washington Quarter 148771_173 Obverse.jpg
            -> US-WQ-148771

        Washington Quarter 148771_212 Obverse.jpg
            -> US-WQ-148771

        Washington Quarter 12 1_1.jpg
            -> US-WQ-12-1

    The final underscore-separated component is treated as an image/
    photograph index rather than part of the physical specimen identifier.
    """

    stem = Path(filename).stem

    # Remove the side designation when it appears in the filename.
    stem = re.sub(r"\s+(?:Obverse|Reverse)\s*$", "", stem, flags=re.IGNORECASE)

    # Remove the known series name from the beginning.
    prefix_map = {
        "Lincoln Cent": "Lincoln Cent",
        "Washington Quarter": "Washington Quarter",
        "Jefferson Nickel": "Jefferson Nickel",
    }

    for prefix in prefix_map.values():
        if stem.startswith(prefix):
            identifier = stem[len(prefix):].strip()
            break
    else:
        identifier = stem.strip()

    # The source uses identifiers such as:
    #   148771_173
    #   12 1_1
    #
    # Treat the final "_N" portion as the image index and keep the
    # preceding portion as the specimen identifier.
    match = re.match(r"^(.*?)_(\d+)$", identifier)

    if match:
        specimen_number = match.group(1).strip()
    else:
        specimen_number = identifier.strip()

    # Normalize whitespace and punctuation for a consistent ID.
    specimen_number = re.sub(r"\s+", "-", specimen_number)
    specimen_number = re.sub(r"[^A-Za-z0-9-]", "-", specimen_number)
    specimen_number = re.sub(r"-+", "-", specimen_number).strip("-")

    return f"US-{series_code}-{specimen_number}"


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

            specimen_id = build_specimen_id(
                filename,
                category_info["code"],
            )

            side = extract_side(filename)

            rows.append(
                {
                    "image_id": f"img_{image_number:06d}",
                    "specimen_id": specimen_id,
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
        "specimen_id",
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
