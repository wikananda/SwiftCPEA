import torch
import timm

class SwiftFormerBackbone(torch.nn.Module):
    """SwiftFormer backbone via timm.

    With ``num_classes=0`` timm removes the classifier head and returns
    pooled feature embeddings of shape ``(B, D)`` — exactly what the
    few-shot head needs.
    """
    def __init__(
        self,
        name: str = "swiftformer_l3.dist_in1k",
        pretrained: bool = True,
        cache_dir: str = "cp",
        **create_kwargs,
    ):
        super().__init__()
        # num_classes=0 -> timm returns (B, D) features, no classifier head
        self.backbone = timm.create_model(
            model_name=name,
            pretrained=pretrained,
            num_classes=0, # feature extractor mode
            cache_dir=cache_dir,
            **create_kwargs,
        )

    @property
    def num_features(self) -> int:
        return self.backbone.num_features

    @property
    def embed_dim(self) -> int:
        """Alias for num_features — used by heads and model assembler."""
        return self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, C, H, W)``
        Returns:
            ``(B, D)`` pooled feature embeddings
        """
        features = self.backbone.forward_features(x)
        B, C, H, W = features.shape
        features = features.view(B, C, -1).transpose(1, 2) # Shape: (B, H*2, C)
        cls_token = features.mean(dim=1, keepdim=True)
        # Concatenate to match CPEA expectations: [class_Token, Patches]
        features = torch.cat([cls_token, features], dim=1) # Shape: (B, H*W + 1, C)
        return features