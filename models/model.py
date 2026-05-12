import torch
import torch.nn as nn
from torch import Tensor

class SwiftCPEA(nn.Module):
    def __init__(
        self,
        backbone=None,
        head=None,
        feat_dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.feat_drop = nn.Dropout(p=feat_dropout)

    def _extract(self, image: Tensor) -> Tensor:
        features = self.backbone(image)  # (B, 1+H*W, C) — [cls_token, patch_tokens...]
        return self.feat_drop(features)  # regularise features before the head

    def forward(
        self,
        support_images: Tensor,
        query_images: Tensor,
        support_labels: Tensor,
    ) -> Tensor:
        support_features = self._extract(support_images)
        query_features = self._extract(query_images)

        n_shot = support_images.size(0) // len(torch.unique(support_labels))
        results, _ = self.head(query_features, support_features, n_shot=n_shot) # CPEA: (feat_query, feat_shot)
        logits = torch.cat(results, dim=0)  # list of (1, S) → (Q, S)
        return logits