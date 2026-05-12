# Using Ravi & Larochelle 2017 split for mini-imagenet

import argparse
import os
import shutil
from pathlib import Path
import csv

def _prepare_mini_imagenet_split(
    images_dir: str,
    csv_dir: str,
    output_dir: str,
    copy: bool = False,
):
    images_dir = Path(images_dir)
    csv_dir = Path(csv_dir)
    output_dir = Path(output_dir)

    split_classes: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        csv_path = csv_dir / f"{split}.csv"
        classes = set()
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                classes.add(row["class_name"])
        split_classes[split] = classes
        print(f"[INFO] {split}: {len(classes)} classes in CSV")

    for split, classes in split_classes.items():
        split_dest = output_dir / split
        os.makedirs(split_dest, exist_ok=True)
        for cls in classes:
            src = images_dir / cls
            dst = split_dest / cls
            if src.exists():
                if copy:
                    shutil.copytree(src, dst)
                else:
                    shutil.move(str(src), dst)
            else:
                print(f"[WARN] Directory not found: {src}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="data/mini_imagenet_raw")
    parser.add_argument("--csv_dir", type=str, default="dataset/mini_imagenet")
    parser.add_argument("--output_dir", type=str, default="data/mini_imagenet")
    parser.add_argument("--copy", type=bool, default=True)
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _prepare_mini_imagenet_split(
        images_dir=PROJECT_ROOT / args.images_dir,
        csv_dir=PROJECT_ROOT / args.csv_dir,
        output_dir=PROJECT_ROOT / args.output_dir,
        copy=args.copy,
    )

