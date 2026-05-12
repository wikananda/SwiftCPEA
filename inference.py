from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import argparse

from models.backbones.swiftformer import SwiftFormerBackbone
from models.cpea import CPEA
from models.model import SwiftCPEA

def _transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def _load_model(device: str, ckpt_path: str = "cp/swiftcpea.pth",
                backbone_name: str = "swiftformer_l3.dist_in1k",
                image_size: int = 224):
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
    cpea = CPEA(in_dim=backbone.embed_dim, seq_len=seq_len)
    model = SwiftCPEA(backbone=backbone, head=cpea).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model

def _build_support_set(gallery_dir: str, n_shot: int, device: torch.device, image_size: int = 224):
    gallery = Path(gallery_dir)
    
    class_names = sorted(
        p.name for p in gallery.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    support_images = []
    support_labels = []

    for label_idx, cls_name in enumerate(class_names):
        cls_dir = gallery / cls_name
        imgs = sorted(cls_dir.iterdir())[:n_shot]

        for img_path in imgs:
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
    parser.add_argument("--gallery", help="Path to support gallery folder", default="data/mini_imagenet_raw")
    parser.add_argument("--backbone", help="Backbone model name", type=str, default="swiftformer_l3.dist_in1k")
    parser.add_argument("--checkpoint", help="Path to checkpoint", type=str, default="cp/swiftcpea.pth")
    parser.add_argument("--n_shot", help="Number of shots", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image_size", help="Input image size (must match training)", type=int, default=224)
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = _load_model(device, args.checkpoint, args.backbone, args.image_size)
    model.eval()
    support_images, support_labels, class_names = _build_support_set(args.gallery, args.n_shot, device, args.image_size)
    pred_class, confidence = inference(args.query, model, support_images, support_labels, class_names, device, args.threshold, args.image_size)
    print(f"Predicted: {pred_class} ({confidence*100:.1f}%)")


if __name__ == "__main__":
    main()