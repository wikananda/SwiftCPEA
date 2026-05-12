import os
from pathlib import Path
from typing import List, Tuple
import numpy as np

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

class DatasetLoader(Dataset):
    def __init__(self, root: str, split: str = 'train', transform=None,
                 max_imgs_per_class: int = None, image_size: int = 224):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"

        self.root = Path(root) / split
        self.transform = transform or (
            self._train_transform(image_size) if split == "train"
            else self._val_test_transform(image_size)
        )

        self.class_names: List[str] = sorted(
            # Filter out .DS_Store
            p.name for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
        self.class_to_label = {c: i for i, c in enumerate(self.class_names)}

        self.samples: List[Tuple[Path, int]] = []
        for cls in self.class_names:
            cls_imgs = [
                img_path
                for img_path in sorted((self.root / cls).iterdir())
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]
            if max_imgs_per_class is not None:
                cls_imgs = cls_imgs[:max_imgs_per_class]
            for img_path in cls_imgs:
                self.samples.append((img_path, self.class_to_label[cls]))
    
    def get_labels(self) -> List[int]:
        return [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

    @staticmethod
    def _train_transform(image_size: int = 224):
        resize_to = int(image_size * 256 / 224)  # maintain standard crop ratio
        return transforms.Compose([
            transforms.Resize(resize_to),
            transforms.CenterCrop(image_size),
            # transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),

            # ── Underwater: colour shift ───────────────────────────────────
            # Water absorbs red light → blue/green cast.
            # hue shift [-0.15, +0.15] simulates varying water depth & colour.
            # transforms.ColorJitter(
            #     brightness=(0.6, 1.0),  # depth-dependent brightness drop
            #     contrast=0.3,
            #     saturation=0.3,
            #     hue=0.08,               # colour cast from water column
            # ),

            # ── Underwater: turbidity / motion blur ────────────────────────
            # GaussianBlur simulates suspended particles and camera motion.
            # Applied randomly (p=0.5) so clean images are also seen.
            # transforms.RandomApply(
            #    [transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 1.0))],
            #    p=0.15,
            # ),

            # ── Underwater: extreme colour loss (murky water) ──────────────
            # Occasionally desaturate to simulate very turbid or deep water.
            # transforms.RandomGrayscale(p=0.05),

            transforms.ToTensor(),
            transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
        ])


    @staticmethod
    def _val_test_transform(image_size: int = 224):
        resize_to = int(image_size * 256 / 224)
        return transforms.Compose([
            transforms.Resize(resize_to),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
        ])