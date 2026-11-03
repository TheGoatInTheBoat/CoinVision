
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError

# Crops this many pixels from each side of image
ZOOM_FACTOR = 100

# This script assumes it is saved in the CoinVision project directory.
PROJECT_DIR = Path(__file__).resolve().parent
COINS_DIR = PROJECT_DIR / "datasets" / "US Coins - Local" / "coins"
TARGET_SIZE = (512, 512)

FOLDERS = (
    "nickels_obverse",
    "nickels_reverse",
    "pennies_obverse",
    "pennies_reverse",
    "quarters_obverse",
    "quarters_reverse",
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def zoom_crop(image: Image.Image) -> Image.Image:
    """Crops ZOOM_FACTOR pixels from all four sides."""
    width, height = image.size

    if width <= 2 * ZOOM_FACTOR or height <= 2 * ZOOM_FACTOR:
        raise ValueError(
            f"ZOOM_FACTOR ({ZOOM_FACTOR}) is too large for image size "
            f"{width}x{height}"
        )

    return image.crop((
        ZOOM_FACTOR,
        ZOOM_FACTOR,
        width - ZOOM_FACTOR,
        height - ZOOM_FACTOR
    ))


def center_crop_square(image: Image.Image) -> Image.Image:
    """Crops equally from the longer axis so the result is square."""
    width, height = image.size

    if width > height:
        left = (width - height) // 2
        right = left + height
        return image.crop((left, 0, right, height))

    if height > width:
        top = (height - width) // 2
        bottom = top + width
        return image.crop((0, top, width, bottom))

    return image


def process_image(source_path: Path, output_path: Path) -> bool:
    """Processes an image and save it to a separate output folder."""
    try:
        with Image.open(source_path) as opened_image:
            # Respect camera orientation metadata, if present.
            image = ImageOps.exif_transpose(opened_image)
            image.load()

            # Convert unsupported color modes to RGB.
            if image.mode not in ("RGB", "RGBA", "L"):
                image = image.convert("RGB")

            # First: Remove ZOOM_FACTOR pixels from every side.
            image = zoom_crop(image)

            # Second: Center-crop to a square.
            image = center_crop_square(image)

            # Third: Resize to 512x512.
            image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

            # JPEG does not support transparency.
            suffix = source_path.suffix.lower()
            if suffix in {".jpg", ".jpeg"} and image.mode == "RGBA":
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.getchannel("A"))
                image = background

            save_options = {}
            if suffix in {".jpg", ".jpeg"}:
                save_options["quality"] = 95
                save_options["subsampling"] = 0

            image.save(output_path, **save_options)

        return True

    except (UnidentifiedImageError, OSError, ValueError) as error:
        print(f"  SKIPPED: {source_path.name} ({error})")
        return False


def main() -> None:
    if not COINS_DIR.is_dir():
        raise SystemExit(
            f"Could not find the coins directory:\n{COINS_DIR}\n"
            "Make sure this script is saved in the CoinVision project folder."
        )

    total_found = 0
    total_processed = 0
    total_skipped = 0

    print(f"Project: {PROJECT_DIR}")
    print(f"Images directory: {COINS_DIR}")
    print(f"Zoom factor: {ZOOM_FACTOR} pixels per side")
    print(f"Output size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print("Original images will NOT be modified.\n")

    for folder_name in FOLDERS:
        source_folder = COINS_DIR / folder_name
        output_folder = COINS_DIR / f"{folder_name}_zoom={ZOOM_FACTOR}"

        if not source_folder.is_dir():
            print(f"[MISSING FOLDER] {source_folder}")
            continue

        # Create the new output folder.
        output_folder.mkdir(parents=True, exist_ok=True)

        images = sorted(
            path for path in source_folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"[{folder_name}] Found {len(images)} image(s)")
        print(f"  Output: {output_folder.name}")

        total_found += len(images)

        for source_path in images:
            # Preserve subfolder structure
            relative_path = source_path.relative_to(source_folder)
            output_path = output_folder / relative_path

            # Create necessary subdirectories
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if process_image(source_path, output_path):
                total_processed += 1
            else:
                total_skipped += 1

    print("\nFinished.")
    print(f"Images found:     {total_found}")
    print(f"Images processed: {total_processed}")
    print(f"Images skipped:   {total_skipped}")
    print(f"Zoom factor:      {ZOOM_FACTOR}")


if __name__ == "__main__":
    main()
