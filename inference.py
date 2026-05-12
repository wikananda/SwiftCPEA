from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import argparse

from models.backbones.swiftformer import SwiftFormerBackbone
from models.cpea import CPEA
from models.model import SeaLensClassifier

def _transform(image_size: int = 224):
    resize_to = int(image_size * 256 / 224)
    return transforms.Compose([
        transforms.Resize(resize_to),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
    ])

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _load_model(device: str, ckpt_path: str = "cp/sealens_fs_cpea.pth",
                backbone_name: str = "swiftformer_l3.dist_in1k",
                image_size: int = 224,
                dropout: float = 0.3,
                class_aware_factor: float = 2.5,
                feat_dropout: float = 0.0):
    # Define models
    backbone = SwiftFormerBackbone(
        name=backbone_name,
        pretrained=True,
    )

    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size)
        feat_map = backbone.backbone.forward_features(dummy)
        seq_len = feat_map.shape[-2] * feat_map.shape[-1]

    # CPEA and model wrapper
    cpea = CPEA(
        in_dim=backbone.embed_dim,
        seq_len=seq_len,
        dropout=dropout,
        class_aware_factor=class_aware_factor,
    )
    model = SeaLensClassifier(
        backbone=backbone,
        head=cpea,
        feat_dropout=feat_dropout,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model

def _build_support_set(gallery_dir: str, n_shot: int, device: torch.device, image_size: int = 224):
    gallery = Path(gallery_dir)
    
    class_names = sorted(
        p.name for p in gallery.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    class_image_paths = []
    too_small_classes = []

    for cls_name in class_names:
        cls_dir = gallery / cls_name
        imgs = [
            p for p in sorted(cls_dir.iterdir())
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(imgs) < n_shot:
            too_small_classes.append((cls_name, len(imgs)))
        else:
            class_image_paths.append(imgs[:n_shot])

    if too_small_classes:
        details = ", ".join(
            f"{cls_name} ({count})" for cls_name, count in too_small_classes
        )
        raise ValueError(
            f"Every support class must have at least n_shot={n_shot} images. "
            f"Classes with too few images: {details}"
        )

    if not class_image_paths:
        raise ValueError(f"No valid support images found in gallery: {gallery}")

    support_images = []
    support_labels = []

    # CPEA expects the same shot-interleaved support order used by TaskSampler:
    # [c0s0, c1s0, ..., cNs0, c0s1, c1s1, ..., cNs1, ...]
    for shot_idx in range(n_shot):
        for label_idx, imgs in enumerate(class_image_paths):
            img_path = imgs[shot_idx]
            img = Image.open(img_path).convert("RGB")
            support_images.append(_transform(image_size)(img))
            support_labels.append(label_idx)

    support_images = torch.stack(support_images).to(device)
    support_labels = torch.tensor(support_labels).to(device)

    return support_images, support_labels, class_names

def inference(query_path, model, support_images, support_labels, class_names, device, threshold, image_size: int = 224):
    query_image = _transform(image_size)(Image.open(query_path).convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(support_images, query_image, support_labels)
        pred_idx = logits.argmax(dim=1).item()
        pred_class = class_names[pred_idx]
        probs = torch.softmax(logits, dim=1)
        confidence = probs[0, pred_idx].item()
    if confidence < threshold:
        return "Unknown", confidence
    print(f"Predicted class: {pred_class} | Confidence: {confidence*100:.2f}%")
    return pred_class, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Path to query image", type=str)
    parser.add_argument("--gallery", help="Path to support gallery folder", default="data/freshwater_fish_ds_raw")
    parser.add_argument("--backbone", help="Backbone model name", type=str, default="swiftformer_l3.dist_in1k")
    parser.add_argument("--checkpoint", help="Path to checkpoint", type=str, default="cp/sealens_fs_swiftcpea.pth")
    parser.add_argument("--n_shot", help="Number of shots", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--image_size", help="Input image size (must match training)", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.3, help="CPEA dropout used during training")
    parser.add_argument("--class_aware_factor", type=float, default=2.5, help="CPEA class-aware factor used during training")
    parser.add_argument("--feat_dropout", type=float, default=0.0, help="Feature dropout used during training")
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = _load_model(
        device,
        args.checkpoint,
        args.backbone,
        args.image_size,
        args.dropout,
        args.class_aware_factor,
        args.feat_dropout,
    )
    model.eval()
    support_images, support_labels, class_names = _build_support_set(args.gallery, args.n_shot, device, args.image_size)
    pred_class, confidence = inference(args.query, model, support_images, support_labels, class_names, device, args.threshold, args.image_size)
    print(f"Predicted: {pred_class} ({confidence*100:.1f}%)")


if __name__ == "__main__":
    main()
