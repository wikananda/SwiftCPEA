"""CLI entrypoint for checking accepted image counts."""

import argparse
from pathlib import Path

from ..dataset.config import DEFAULT_IMAGES_DIR, DEFAULT_REDOWNLOAD_FILE
from ..dataset.checks import count_images, species_name_from_folder_slug

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse image-count checker CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Check the number of accepted images in each species folder."
    )
    parser.add_argument("target", type=int, help="Required number of images per folder")
    parser.add_argument(
        "--images-dir",
        default=DEFAULT_IMAGES_DIR,
        help=f"Directory containing one folder per species. Default: {DEFAULT_IMAGES_DIR}",
    )
    parser.add_argument(
        "--redownload-file",
        default=DEFAULT_REDOWNLOAD_FILE,
        help=(
            "Where to write species below target. The file is written as taxon query "
            f"names, not folder slugs. Default: {DEFAULT_REDOWNLOAD_FILE}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run accepted image count checks."""
    args = parse_args()
    images_dir = (PROJECT_ROOT / args.images_dir).resolve()
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"Error: {images_dir} directory not found.")
        return

    not_meeting_target = []
    species_folders = sorted(path for path in images_dir.iterdir() if path.is_dir())

    for folder in species_folders:
        image_count = count_images(folder)
        print(f"{folder.name}: {image_count}/{args.target}")

        if image_count < args.target:
            not_meeting_target.append(species_name_from_folder_slug(folder.name))

    print("\nFolders not meeting the target count:")
    if not_meeting_target:
        for species in not_meeting_target:
            print(species)

        redownload_path = (PROJECT_ROOT / args.redownload_file).resolve()
        redownload_path.parent.mkdir(parents=True, exist_ok=True)
        redownload_path.write_text(
            "\n".join(not_meeting_target) + "\n", encoding="utf-8"
        )
        print(f"\nSaved {len(not_meeting_target)} species to {redownload_path}")
    else:
        print("All folders meet the target count!")


if __name__ == "__main__":
    main()
