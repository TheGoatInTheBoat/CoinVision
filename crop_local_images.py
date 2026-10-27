from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError

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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def center_crop_square(image: Image.Image) -> Image.Image:
    """Crop equally from the longer axis so the result is square."""
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


def process_image(path: Path) -> bool:
    """Center-crop and resize one image, replacing it in place."""
    try:
        with Image.open(path) as opened_image:
            # Respect camera orientation metadata, if present.
            image = ImageOps.exif_transpose(opened_image)
            image.load()

            # Keep the original color mode where practical, but convert
            # palette/CMYK/etc. images to RGB for consistent output.
            if image.mode not in ("RGB", "RGBA", "L"):
                image = image.convert("RGB")

            image = center_crop_square(image)
            image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

            # Preserve alpha for PNG/WebP; JPEG does not support alpha.
            suffix = path.suffix.lower()
            if suffix in {".jpg", ".jpeg"} and image.mode == "RGBA":
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.getchannel("A"))
                image = background

            save_options = {}
            if suffix in {".jpg", ".jpeg"}:
                save_options["quality"] = 95
                save_options["subsampling"] = 0

            image.save(path, **save_options)

        return True

    except (UnidentifiedImageError, OSError, ValueError) as error:
        print(f"  SKIPPED: {path.name} ({error})")
        return False


def main() -> None:
    if not COINS_DIR.is_dir():
        raise SystemExit(
            f"Could not find the coins directory:\n{COINS_DIR}\n"
            "Make sure this script is saved in your CoinVision project folder."
        )

    total_found = 0
    total_processed = 0
    total_skipped = 0

    print(f"Project: {PROJECT_DIR}")
    print(f"Images directory: {COINS_DIR}")
    print(f"Output size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print("Images will be overwritten in place.\n")

    for folder_name in FOLDERS:
        folder = COINS_DIR / folder_name

        if not folder.is_dir():
            print(f"[MISSING FOLDER] {folder}")
            continue

        images = sorted(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"[{folder_name}] Found {len(images)} image(s)")
        total_found += len(images)

        for path in images:
            if process_image(path):
                total_processed += 1
            else:
                total_skipped += 1

    print("\nFinished.")
    print(f"Images found:     {total_found}")
    print(f"Images processed: {total_processed}")
    print(f"Images skipped:   {total_skipped}")


if __name__ == "__main__":
    main()
