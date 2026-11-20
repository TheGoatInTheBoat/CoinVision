
import csv
import shutil
from pathlib import Path
from collections import Counter

# CONFIGURATION

ROOT_DIR = Path(__file__).resolve().parent
COINS_DIR = ROOT_DIR / "coins"

DATASET_CSV = ROOT_DIR / "dataset.csv"
BACKUP_CSV = ROOT_DIR / "dataset_backup.csv"

# Map cleaned folder names to their original denomination names.
COIN_FOLDERS = {
    "Jefferson Nickels, 1938-Date SIZE=512 Cleaned":
        "Jefferson Nickels, 1938-Date",

    "Washington Quarters, 1932-1998 SIZE=512 Cleaned":
        "Washington Quarters, 1932-1998",

    "Lincoln Cents, 1909-Date SIZE=512 Cleaned":
        "Lincoln Cents, 1909-Date",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


# MAIN

def main():

    print("=" * 60)
    print("COINVISION DATASET CSV GENERATOR")
    print("=" * 60)

    dataset_entries = []
    denomination_counts = Counter()

    # BACK UP EXISTING CSV

    if DATASET_CSV.exists():

        shutil.copy2(DATASET_CSV, BACKUP_CSV)

        print(f"\nBackup created: {BACKUP_CSV.name}")

    # SCAN IMAGE FOLDERS

    for folder_name, denomination in COIN_FOLDERS.items():

        folder_path = COINS_DIR / folder_name

        if not folder_path.exists():
            print(f"\nWARNING: Folder not found: {folder_path}")
            continue

        image_paths = sorted(
            path for path in folder_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"\n{denomination}: {len(image_paths)} images")

        for image_path in image_paths:

            # Preserve the filename.
            dataset_entries.append([
                denomination,
                image_path.name
            ])

            denomination_counts[denomination] += 1

    # CHECK FOR DUPLICATE FILENAMES

    filename_counts = Counter(
        (entry[0], entry[1])
        for entry in dataset_entries
    )

    duplicates = [
        entry for entry, count in filename_counts.items()
        if count > 1
    ]

    if duplicates:
        print("\nWARNING: Duplicate filenames detected!")

        for denomination, filename in duplicates:
            print(f"  {denomination}: {filename}")

    # --------------------------------------------------------
    # WRITE DATASET CSV
    # --------------------------------------------------------

    dataset_entries.sort(
        key=lambda entry: (entry[0], entry[1].lower())
    )

    with DATASET_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerows(dataset_entries)

    # SUMMARY

    print("\n" + "=" * 60)
    print("DATASET CSV UPDATED")
    print("=" * 60)

    for denomination in COIN_FOLDERS.values():
        print(
            f"{denomination:35s}: "
            f"{denomination_counts[denomination]:,}"
        )

    print("-" * 60)
    print(f"Total images: {len(dataset_entries):,}")
    print(f"Output: {DATASET_CSV}")

    if duplicates:
        print(f"Duplicate entries: {len(duplicates)}")
    else:
        print("Duplicate entries: 0")

    print("\nDone!")


if __name__ == "__main__":
    main()
