from pathlib import Path
from PIL import Image

# size/settings

INPUT_FOLDER = Path("coins/Washington Quarters, 1932-1998")
N = 100  # N x N pixels

OUTPUT_FOLDER = Path(f"coins/{INPUT_FOLDER.name} SIZE={N}")

# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png"
}

# Image processing

def process_image(input_path, output_path, size):
    with Image.open(input_path) as img:

        # standardize the color mode
        img.seek(0)
        img = img.convert("RGB")

        width, height = img.size

        # Crop to a square:
        # - Too wide: remove pixels from the LEFT side
        #       - I noticed photos tend to contain excess pixels on left
        # - Too tall: remove pixels from the BOTTOM
        #       - I noticed photos tend to contain excess pixels on bottom
        #       - However, "too tall" should apply to few to zero photos in the dataset
        if width > height:
            img = img.crop((
                width - height,
                0,
                width,
                height
            ))

        elif height > width:
            img = img.crop((
                0,
                0,
                width,
                width
            ))

        # Resize to N x N
        # LANCZOS resizing
        img = img.resize(
            (size, size),
            Image.Resampling.LANCZOS
        )

        # Save the processed image
        # Keep the original filename
        img.save(output_path, quality=95)


def main():
    if not INPUT_FOLDER.is_dir():
        print(f"ERROR: Input folder not found: {INPUT_FOLDER}")
        print("Place this script in the directory containing the folder.")
        return

    if N < 1:
        print("ERROR: N must be a positive integer.")
        return

    # Create the output folder if it doesn't exist
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    count = 0
    errors = 0

    # Process images directly inside the input folder
    for file_path in INPUT_FOLDER.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        output_path = OUTPUT_FOLDER / file_path.name

        try:
            process_image(file_path, output_path, N)
            count += 1
            print(f"Processed: {file_path.name}")

        except Exception as e:
            errors += 1
            print(f"FAILED: {file_path.name} — {e}")

    print("\n========== COMPLETE ==========")
    print(f"Images processed: {count}")
    print(f"Errors:           {errors}")
    print(f"Output folder:    {OUTPUT_FOLDER.resolve()}")
    print(f"Final dimensions: {N} x {N}")


if __name__ == "__main__":
    main()