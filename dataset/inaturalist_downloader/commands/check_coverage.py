"""CLI entrypoint for checking split coverage."""

import argparse
from pathlib import Path

from ..dataset.config import DEFAULT_SPLIT_DIR
from ..dataset.checks import load_species_set

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPECIES_FILE = "fish_ds/species.txt"


def parse_args() -> argparse.Namespace:
    """Parse split coverage checker CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Check that species.txt entries are covered by train/val/test split files."
    )
    parser.add_argument(
        "--species-file",
        default=DEFAULT_SPECIES_FILE,
        help=f"Species list to check. Default: {DEFAULT_SPECIES_FILE}",
    )
    parser.add_argument(
        "--split-dir",
        default=DEFAULT_SPLIT_DIR,
        help=(
            "Directory containing train.txt, val.txt, and test.txt. "
            f"Default: {DEFAULT_SPLIT_DIR}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run split coverage check."""
    args = parse_args()
    split_dir = (PROJECT_ROOT / args.split_dir).resolve()
    species_file = (PROJECT_ROOT / args.species_file).resolve()

    species_all = load_species_set(species_file)
    train = load_species_set(split_dir / "train.txt")
    val = load_species_set(split_dir / "val.txt")
    test = load_species_set(split_dir / "test.txt")

    covered = train | val | test
    missing = species_all - covered

    print(f"Total species in {species_file}: {len(species_all)}")
    print(f"Total covered in splits: {len(covered)}")
    print(f"Missing: {len(missing)}")

    if missing:
        print("\nMissing species:")
        for species in sorted(missing):
            print(species)
    else:
        print("\nAll species are covered!")


if __name__ == "__main__":
    main()
