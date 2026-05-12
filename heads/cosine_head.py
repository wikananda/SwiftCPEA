# Source: https://docsaid.org/en/papers/face-recognition/cosface/
# Further Cosine head implementation: https://github.com/mk-minchul/AdaFace
# AdaCos (achieving best accuracy in omniglot): https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineHead(nn.Module):
    def __init__(
        self, 
        num_features: int, 
        num_classes: int,
        head_name: str = 'cosface', 
        s: float = 30.0,
        m: float = 0.35
    ):
        super().__init__()
        self.head_name = head_name
        self.num_features = num_features
        self.num_classes = num_classes
        self.s = s
        self.m = m

        if head_name == "cosface":
            self.head = CosFace(num_features, num_classes, s, m)
        else:
            raise ValueError(f"Unknown head name: {head_name}")

    def forward(self, x, label=None):
        return self.head(x, label)

class CosFace(nn.Module):
    def __init__(self, num_features, num_classes, s=30.0, m=0.35):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # Normalize input and weights
        x = F.normalize(input)
        W = F.normalize(self.W)

        # Compute cosine similarity
        logits = F.linear(x, W) # dot product

        # For inference mode
        if label is None:
            return logits

        # pick the correct class
        row_idx = torch.arange(0, logits.shape[0])
        target_logits = logits[row_idx, label]
        # subtract
        final_target_logits = target_logits - self.m
        logits[row_idx, label] = final_target_logits

        # scaling
        logits = logits * self.s
        return logits

# APPLY ADA COS later
        
