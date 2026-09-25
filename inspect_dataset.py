from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError


# Configuration

REPO_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = REPO_ROOT / "datasets" / "US Coins - Kaggle"
CSV_PATH = DATASET_ROOT / "dataset.csv"
IMAGES_ROOT = DATASET_ROOT / "coins"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

EXPECTED_CATEGORIES = {
    "Jefferson Nickels, 1938-Date",
    "Lincoln Cents, 1909-Date",
    "Washington Quarters, 1932-1998",
}


# Utility functions

def normalize_path_component(value: str) -> str:
    """
    Normalize a filename/category for comparison.

    We intentionally do not perform aggressive normalization such as
    removing punctuation or changing spaces, because filenames should
    match the dataset exactly.
    """
    return value.strip().replace("\\", "/")


def format_number(value: int) -> str:
    return f"{value:,}"


# CSV inspection

def inspect_csv():
    print("=" * 70)
    print("CSV INSPECTION")
    print("=" * 70)

    if not CSV_PATH.exists():
        print("[ERROR] CSV does not exist:")
        print(f"        {CSV_PATH}")
        return None

    expected_images = set()
    csv_rows = []

    malformed_rows = []
    categories = Counter()

    with CSV_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.reader(f)

        for line_number, row in enumerate(reader, start=1):

            # Ignore completely empty rows.
            if not row or all(not cell.strip() for cell in row):
                continue

            if len(row) != 2:
                malformed_rows.append(
                    (line_number, row)
                )
                continue

            category = normalize_path_component(row[0])
            filename = normalize_path_component(row[1])

            key = (category, filename)

            expected_images.add(key)
            csv_rows.append(
                (line_number, category, filename)
            )

            categories[category] += 1

    print(f"CSV: {CSV_PATH}")
    print(f"Rows: {format_number(len(csv_rows))}")
    print(
        "Unique (category, filename) pairs: "
        f"{format_number(len(expected_images))}"
    )

    print("\nCategories:")
    for category, count in categories.most_common():
        print(f"  {count:>8,}  {category}")

    if malformed_rows:
        print(
            f"\n[ERROR] Malformed CSV rows: "
            f"{format_number(len(malformed_rows))}"
        )

        for line_number, row in malformed_rows[:20]:
            print(f"  Line {line_number}: {row}")

        if len(malformed_rows) > 20:
            print(
                f"  ... and "
                f"{len(malformed_rows) - 20:,} more"
            )

    unexpected_categories = set(categories) - EXPECTED_CATEGORIES
    missing_categories = EXPECTED_CATEGORIES - set(categories)

    if unexpected_categories:
        print("\n[ERROR] Unexpected categories in CSV:")
        for category in sorted(unexpected_categories):
            print(f"  {category}")

    if missing_categories:
        print(
            "\n[WARNING] Expected categories missing from CSV:"
        )
        for category in sorted(missing_categories):
            print(f"  {category}")

    return {
        "rows": csv_rows,
        "expected_images": expected_images,
        "categories": categories,
    }


# CSV duplicate entry inspection

def inspect_csv_duplicates(csv_rows):
    """
    Find exact duplicate CSV entries.

    A duplicate means the exact same (category, filename) pair
    occurs more than once in dataset.csv.

    Different images are NOT considered duplicates, even if they
    depict the same type, year, or physical coin.
    """
    print("\n" + "=" * 70)
    print("CSV DUPLICATE ENTRY CHECK")
    print("=" * 70)

    entries = defaultdict(list)

    for line_number, category, filename in csv_rows:
        key = (category, filename)
        entries[key].append(line_number)

    duplicates = {
        key: lines
        for key, lines in entries.items()
        if len(lines) > 1
    }

    if not duplicates:
        print("[OK] No duplicate CSV entries found.")
        return duplicates

    duplicate_occurrences = sum(
        len(lines)
        for lines in duplicates.values()
    )

    print(
        f"[ERROR] Found {len(duplicates):,} duplicated "
        "(category, filename) entries."
    )

    print(
        f"[ERROR] Total CSV rows involved: "
        f"{duplicate_occurrences:,}"
    )

    for (category, filename), lines in list(
        duplicates.items()
    )[:50]:

        print()
        print(f"  {category}\\{filename}")
        print(
            "  Appears on CSV lines: "
            + ", ".join(str(line) for line in lines)
        )

    if len(duplicates) > 50:
        print(
            f"\n  ... and "
            f"{len(duplicates) - 50:,} more duplicate entries."
        )

    return duplicates


# ============================================================
# Filesystem inspection
# ============================================================

def inspect_filesystem():
    print("\n" + "=" * 70)
    print("FILESYSTEM INSPECTION")
    print("=" * 70)

    if not IMAGES_ROOT.exists():
        print("[ERROR] Image directory does not exist:")
        print(f"        {IMAGES_ROOT}")
        return None

    image_files = []
    non_image_files = []

    for path in IMAGES_ROOT.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(path)
        else:
            non_image_files.append(path)

    print(
        f"Image files found: "
        f"{format_number(len(image_files))}"
    )

    print(
        f"Non-image files found: "
        f"{format_number(len(non_image_files))}"
    )

    if non_image_files:
        print("\n[WARNING] Non-image files inside coins/:")

        for path in non_image_files[:20]:
            print(
                f"  {path.relative_to(IMAGES_ROOT)}"
            )

        if len(non_image_files) > 20:
            print(
                f"  ... and "
                f"{len(non_image_files) - 20:,} more"
            )

    return {
        "images": image_files,
        "non_images": non_image_files,
    }


# CSV <-> filesystem consistency

def compare_csv_to_filesystem(
    csv_data,
    filesystem_data,
):
    print("\n" + "=" * 70)
    print("CSV <-> FILESYSTEM CONSISTENCY")
    print("=" * 70)

    expected_images = csv_data["expected_images"]
    actual_images = filesystem_data["images"]

    # Build actual set of (category, filename)
    actual_image_pairs = set()

    for path in actual_images:

        try:
            relative = path.relative_to(IMAGES_ROOT)

        except ValueError:
            continue

        parts = relative.parts

        if len(parts) < 2:
            # Image directly inside coins/
            category = ""
            filename = path.name

        else:
            category = parts[0]
            filename = parts[-1]

        actual_image_pairs.add(
            (
                normalize_path_component(category),
                normalize_path_component(filename),
            )
        )

    # CSV -> filesystem

    missing_files = expected_images - actual_image_pairs

    if missing_files:
        print(
            f"[ERROR] CSV entries with no corresponding image: "
            f"{format_number(len(missing_files))}"
        )

        for category, filename in sorted(
            missing_files
        )[:30]:
            print(
                f"  {category} -> {filename}"
            )

        if len(missing_files) > 30:
            print(
                f"  ... and "
                f"{len(missing_files) - 30:,} more"
            )

    else:
        print(
            "[OK] Every CSV entry has a corresponding image."
        )

    # filesystem -> CSV

    extra_files = actual_image_pairs - expected_images

    if extra_files:
        print(
            f"[ERROR] Images with no corresponding CSV entry: "
            f"{format_number(len(extra_files))}"
        )

        for category, filename in sorted(
            extra_files
        )[:30]:
            print(
                f"  {category} -> {filename}"
            )

        if len(extra_files) > 30:
            print(
                f"  ... and "
                f"{len(extra_files) - 30:,} more"
            )

    else:
        print(
            "[OK] Every image has a corresponding CSV entry."
        )

    # --------------------------------------------------------
    # Category-directory consistency
    # --------------------------------------------------------

    actual_categories = set(
        category
        for category, _ in actual_image_pairs
    )

    unexpected_categories = (
        actual_categories - EXPECTED_CATEGORIES
    )

    if unexpected_categories:
        print("\n[ERROR] Unexpected image directories:")

        for category in sorted(
            unexpected_categories
        ):
            print(f"  {category}")

    else:
        print(
            "[OK] No unexpected coin categories found."
        )

    # Images directly in coins/

    root_images = [
        p
        for p in actual_images
        if p.parent == IMAGES_ROOT
    ]

    if root_images:
        print(
            f"\n[ERROR] Images located directly inside coins/: "
            f"{format_number(len(root_images))}"
        )

        for path in root_images[:20]:
            print(f"  {path.name}")

        if len(root_images) > 20:
            print(
                f"  ... and "
                f"{len(root_images) - 20:,} more"
            )

    else:
        print(
            "[OK] No images are misplaced directly in coins/."
        )

    return {
        "actual_pairs": actual_image_pairs,
        "missing_files": missing_files,
        "extra_files": extra_files,
    }


# Image readability

def inspect_image_readability(image_files):
    print("\n" + "=" * 70)
    print("IMAGE READABILITY")
    print("=" * 70)

    unreadable = []
    metadata = []

    for index, path in enumerate(
        image_files,
        start=1,
    ):

        try:
            # verify() checks file integrity without fully
            # decoding the image.
            with Image.open(path) as image:
                image.verify()

            # Reopen because verify() invalidates the image.
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
                image_format = image.format

            if width <= 0 or height <= 0:
                unreadable.append(
                    (path, "invalid dimensions")
                )
                continue

            metadata.append(
                {
                    "path": path,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "format": image_format,
                }
            )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:

            unreadable.append(
                (path, str(exc))
            )

        if index % 5000 == 0:
            print(
                f"  Checked {format_number(index)} / "
                f"{format_number(len(image_files))}..."
            )

    if unreadable:
        print(
            f"\n[ERROR] Unreadable/corrupt images: "
            f"{format_number(len(unreadable))}"
        )

        for path, reason in unreadable[:30]:
            print(f"  {path}")
            print(f"    {reason}")

        if len(unreadable) > 30:
            print(
                f"  ... and "
                f"{len(unreadable) - 30:,} more"
            )

    else:
        print(
            f"[OK] All {format_number(len(image_files))} "
            "images are readable."
        )

    # Basic image statistics
    if metadata:

        dimensions = Counter(
            (item["width"], item["height"])
            for item in metadata
        )

        formats = Counter(
            item["format"]
            for item in metadata
        )

        modes = Counter(
            item["mode"]
            for item in metadata
        )

        print("\nImage formats:")

        for image_format, count in formats.most_common():
            print(
                f"  {image_format}: "
                f"{format_number(count)}"
            )

        print("\nColor modes:")

        for mode, count in modes.most_common():
            print(
                f"  {mode}: "
                f"{format_number(count)}"
            )

        print("\nMost common dimensions:")

        for (width, height), count in dimensions.most_common(10):
            print(
                f"  {width}x{height}: "
                f"{format_number(count)}"
            )

        widths = [
            item["width"]
            for item in metadata
        ]

        heights = [
            item["height"]
            for item in metadata
        ]

        print(
            f"\nWidth range:  "
            f"{min(widths)} - {max(widths)}"
        )

        print(
            f"Height range: "
            f"{min(heights)} - {max(heights)}"
        )

    return {
        "unreadable": unreadable,
        "metadata": metadata,
    }


# Filename / dataset sanity checks

def inspect_filename_sanity(
    image_files,
    csv_data,
):
    print("\n" + "=" * 70)
    print("FILENAME / LABEL SANITY")
    print("=" * 70)

    problems = []

    expected_categories = EXPECTED_CATEGORIES

    for path in image_files:

        category = path.parent.name
        filename = path.name

        # Check category directory.
        if category not in expected_categories:
            problems.append(
                (
                    path,
                    f"unexpected category directory: "
                    f"{category}",
                )
            )

        # Check extension.
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            problems.append(
                (
                    path,
                    f"unsupported image extension: "
                    f"{path.suffix}",
                )
            )

        # Check empty/whitespace filename.
        if not path.stem.strip():
            problems.append(
                (
                    path,
                    "empty/whitespace filename",
                )
            )

        # Detect suspicious filename whitespace.
        if filename != filename.strip():
            problems.append(
                (
                    path,
                    "leading/trailing whitespace in filename",
                )
            )

    if problems:
        print(
            f"[WARNING] Filename/category issues: "
            f"{format_number(len(problems))}"
        )

        for path, reason in problems[:30]:
            print(
                f"  {path.relative_to(IMAGES_ROOT)}"
            )
            print(f"    {reason}")

        if len(problems) > 30:
            print(
                f"  ... and "
                f"{len(problems) - 30:,} more"
            )

    else:
        print(
            "[OK] No obvious filename/category issues found."
        )

    # Check whether the same filename appears under multiple
    # categories. This is not necessarily an error.
    filename_to_categories = defaultdict(set)

    for category, filename in (
        csv_data["expected_images"]
    ):
        filename_to_categories[
            filename
        ].add(category)

    ambiguous_filenames = {
        filename: categories
        for filename, categories
        in filename_to_categories.items()
        if len(categories) > 1
    }

    if ambiguous_filenames:
        print(
            "\n[WARNING] Same filename appears in "
            "multiple categories:"
        )

        for filename, categories in list(
            ambiguous_filenames.items()
        )[:20]:

            print(
                f"  {filename}: "
                f"{', '.join(sorted(categories))}"
            )

        if len(ambiguous_filenames) > 20:
            print(
                f"  ... and "
                f"{len(ambiguous_filenames) - 20:,} more"
            )

    else:
        print(
            "[OK] No filename ambiguity across categories."
        )


# Final summary

def print_summary(
    csv_data,
    filesystem_data,
    consistency_data,
    readability_data,
    csv_duplicates,
):
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    errors = []

    if not csv_data:
        errors.append(
            "CSV could not be inspected."
        )

    if not filesystem_data:
        errors.append(
            "Image directory could not be inspected."
        )

    if consistency_data:

        if consistency_data["missing_files"]:
            errors.append(
                f"{len(consistency_data['missing_files']):,} "
                "CSV entries missing images"
            )

        if consistency_data["extra_files"]:
            errors.append(
                f"{len(consistency_data['extra_files']):,} "
                "images missing CSV entries"
            )

    if readability_data:
        if readability_data["unreadable"]:
            errors.append(
                f"{len(readability_data['unreadable']):,} "
                "unreadable images"
            )

    # IMPORTANT:
    # A duplicate means the exact same CSV entry appears
    # more than once. Different images are not duplicates.
    if csv_duplicates:
        errors.append(
            f"{len(csv_duplicates):,} "
            "duplicate CSV entries"
        )

    if errors:
        print("\nDATASET STATUS: FAIL\n")

        for error in errors:
            print(f"  [ERROR] {error}")

        print(
            "\nFix the errors above before treating this "
            "dataset as a clean CoinVision dataset."
        )

        return False

    print("\nDATASET STATUS: PASS")

    print(
        "No critical consistency, readability, or duplicate "
        "problems were detected."
    )

    return True


# Main

def main():

    print("=" * 70)
    print("CoinVision Dataset Inspector")
    print("=" * 70)

    print(f"Repository:     {REPO_ROOT}")
    print(f"Dataset:        {DATASET_ROOT}")
    print(f"CSV:            {CSV_PATH}")
    print(f"Images:         {IMAGES_ROOT}")

    # CSV

    csv_data = inspect_csv()

    if csv_data is None:
        sys.exit(1)

    # CSV duplicate entries

    csv_duplicates = inspect_csv_duplicates(
        csv_data["rows"]
    )

    # Filesystem

    filesystem_data = inspect_filesystem()

    if filesystem_data is None:
        sys.exit(1)

    # CSV <-> filesystem

    consistency_data = compare_csv_to_filesystem(
        csv_data,
        filesystem_data,
    )

    # Filename sanity

    inspect_filename_sanity(
        filesystem_data["images"],
        csv_data,
    )

    # Image readability

    readability_data = inspect_image_readability(
        filesystem_data["images"]
    )

    # Final result

    success = print_summary(
        csv_data,
        filesystem_data,
        consistency_data,
        readability_data,
        csv_duplicates,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()