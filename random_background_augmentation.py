
import cv2
import numpy as np
import random
from pathlib import Path

# CONFIGURATION
# Repository root (directory containing this script)
ROOT_DIR = Path(__file__).resolve().parent

# Input directories
COINS_DIR = (
    ROOT_DIR
    / "datasets"
    / "US Coins - Kaggle"
    / "coins"
)

BACKGROUNDS_DIR = (
    ROOT_DIR
    / "datasets"
    / "US Coins - Kaggle"
    / "backgrounds"
)

# Coin denomination folders
COIN_FOLDERS = [
    "Lincoln Cents, 1909-Date SIZE=512",
    "Washington Quarters, 1932-1998 SIZE=512",
    "Jefferson Nickels, 1938-Date SIZE=512",
]

# Output folder suffix
OUTPUT_SUFFIX = ", random_bg"

# Reproducibility: same seed produces the same background selections
RANDOM_SEED = 1

# GrabCut settings
GRABCUT_ITERATIONS = 5

# How much of the image border is initially considered
# definite background. 0.03 means 3% on each side.
GRABCUT_MARGIN = 0.03

# Feather the segmentation boundary to reduce harsh edges.
# Larger values produce softer edges.
MASK_BLUR_SIZE = 5

# Optional: save grayscale foreground masks for inspection.
SAVE_DEBUG_MASKS = False

DEBUG_DIR = ROOT_DIR / "models" / "random_bg_debug"

# Supported image extensions
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp"
}


# BACKGROUND HANDLING
def load_background_paths():
    """Find all usable background images."""

    backgrounds = [
        path
        for path in BACKGROUNDS_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not backgrounds:
        raise RuntimeError(
            f"No background images found in: {BACKGROUNDS_DIR}"
        )

    return backgrounds


def resize_background(background, target_width, target_height):
    """
    Resize and center-crop a background to match the coin image.
    Preserves the background's aspect ratio.
    """

    bg_height, bg_width = background.shape[:2]

    scale = max(
        target_width / bg_width,
        target_height / bg_height
    )

    new_width = int(bg_width * scale)
    new_height = int(bg_height * scale)

    background = cv2.resize(
        background,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_CUBIC
    )

    # Center crop
    x_start = (new_width - target_width) // 2
    y_start = (new_height - target_height) // 2

    background = background[
        y_start:y_start + target_height,
        x_start:x_start + target_width
    ]

    return background


# COIN SEGMENTATION
def isolate_coin(image):
    """
    Use GrabCut to estimate the coin foreground.

    Assumes the coin is approximately centered in the image.
    Returns a softened foreground mask.
    """

    height, width = image.shape[:2]

    margin_x = int(width * GRABCUT_MARGIN)
    margin_y = int(height * GRABCUT_MARGIN)

    # Initialize GrabCut mask.
    mask = np.zeros((height, width), np.uint8)

    rect = (
        margin_x,
        margin_y,
        width - 2 * margin_x,
        height - 2 * margin_y
    )

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    # Run GrabCut
    cv2.grabCut(
        image,
        mask,
        rect,
        background_model,
        foreground_model,
        GRABCUT_ITERATIONS,
        cv2.GC_INIT_WITH_RECT
    )

    # Definite and probable foreground pixels
    foreground_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # Keep the largest connected foreground component.
    # This helps remove isolated background noise.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground_mask,
        connectivity=8
    )

    if count > 1:
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        foreground_mask = np.where(
            labels == largest_component,
            255,
            0
        ).astype(np.uint8)

    # Close small gaps in the foreground mask.
    kernel = np.ones((5, 5), np.uint8)

    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Slightly soften the boundary for natural compositing.
    if MASK_BLUR_SIZE > 1:
        blur_size = MASK_BLUR_SIZE

        if blur_size % 2 == 0:
            blur_size += 1

        foreground_mask = cv2.GaussianBlur(
            foreground_mask,
            (blur_size, blur_size),
            0
        )

    return foreground_mask


# IMAGE COMPOSITING
def composite_coin(image, background):
    """
    Isolate the coin and place it over a new background.
    """

    height, width = image.shape[:2]

    background = resize_background(
        background,
        width,
        height
    )

    foreground_mask = isolate_coin(image)

    # Convert mask to alpha values between 0 and 1.
    alpha = foreground_mask.astype(np.float32) / 255.0

    # Expand alpha to three channels.
    alpha = np.expand_dims(alpha, axis=2)

    # Blend original coin and background.
    result = (
        image.astype(np.float32) * alpha
        + background.astype(np.float32) * (1 - alpha)
    )

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result, foreground_mask


# MAIN PROCESSING
def main():

    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("COIN BACKGROUND AUGMENTATION")
    print("=" * 60)

    if not COINS_DIR.exists():
        raise FileNotFoundError(
            f"Coin directory not found: {COINS_DIR}"
        )

    if not BACKGROUNDS_DIR.exists():
        raise FileNotFoundError(
            f"Background directory not found: {BACKGROUNDS_DIR}"
        )

    background_paths = load_background_paths()

    print(f"\nBackground images found: {len(background_paths)}")

    if SAVE_DEBUG_MASKS:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_failed = 0

    for folder_name in COIN_FOLDERS:

        input_folder = COINS_DIR / folder_name

        output_folder = COINS_DIR / (
            folder_name + OUTPUT_SUFFIX
        )

        if not input_folder.exists():
            print(f"\nWARNING: Folder not found: {input_folder}")
            continue

        output_folder.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing: {folder_name}")
        print(f"Output: {output_folder.name}")

        image_paths = [
            path
            for path in input_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"Images found: {len(image_paths)}")

        for image_path in image_paths:

            try:
                image = cv2.imread(str(image_path))

                if image is None:
                    print(f"FAILED to read: {image_path.name}")
                    total_failed += 1
                    continue

                # Select a random background.
                background_path = random.choice(background_paths)

                background = cv2.imread(str(background_path))

                if background is None:
                    print(
                        f"FAILED to read background: "
                        f"{background_path.name}"
                    )
                    total_failed += 1
                    continue

                # Segment and composite.
                result, mask = composite_coin(
                    image,
                    background
                )

                # Preserve relative subdirectory structure.
                relative_path = image_path.relative_to(input_folder)

                output_path = output_folder / relative_path
                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                # Preserve original filename.
                success = cv2.imwrite(
                    str(output_path),
                    result
                )

                if not success:
                    print(f"FAILED to save: {output_path}")
                    total_failed += 1
                    continue

                if SAVE_DEBUG_MASKS:
                    debug_path = (
                        DEBUG_DIR
                        / folder_name
                        / relative_path.with_suffix(".png")
                    )

                    debug_path.parent.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    cv2.imwrite(str(debug_path), mask)

                total_processed += 1

                print(
                    f"Processed: {relative_path} "
                    f"| Background: {background_path.name}"
                )

            except Exception as error:
                print(
                    f"ERROR processing {image_path.name}: {error}"
                )
                total_failed += 1

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Successfully processed: {total_processed}")
    print(f"Failed: {total_failed}")
    print(f"Background images available: {len(background_paths)}")
    print(f"Output suffix: {OUTPUT_SUFFIX}")

    if SAVE_DEBUG_MASKS:
        print(f"Debug masks saved to: {DEBUG_DIR}")


if __name__ == "__main__":
    main()
