import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.head(x)