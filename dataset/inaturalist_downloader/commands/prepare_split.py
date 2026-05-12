"""CLI entrypoint for preparing train/val/test dataset splits."""

import argparse
from pathlib import Path

from ..dataset.config import DEFAULT_IMAGES_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SPLIT_DIR, SPLITS
from ..dataset.splitter import build_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse split preparation CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Split iNaturalist class folders into train/val/test using text files."
    )
    parser.add_argument(
        "--images-dir",
        default=DEFAULT_IMAGES_DIR,
        help=f"Source directory containing one folder per species. Default: {DEFAULT_IMAGES_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination directory for train/val/test folders. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--split-dir",
        default=DEFAULT_SPLIT_DIR,
        help=(
            "Directory containing train.txt, val.txt, and test.txt. "
            f"Default: {DEFAULT_SPLIT_DIR}"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "move", "symlink"),
        default="copy",
        help="How to place class folders into the split output. Default: copy",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Copy images directly into each split folder instead of split/class_name/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing destination class folder if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Run split preparation."""
    args = parse_args()

    images_dir = (PROJECT_ROOT / args.images_dir).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    split_dir = (PROJECT_ROOT / args.split_dir).resolve()

    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")

    for split_name in SPLITS:
        split_file = split_dir / f"{split_name}.txt"
        build_split(
            split_name=split_name,
            split_file=split_file,
            images_dir=images_dir,
            output_dir=output_dir,
            mode=args.mode,
            flat=args.flat,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
