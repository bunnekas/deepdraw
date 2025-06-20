import torch.nn as nn
from torch import amp

class ClipClassifier(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        for name, param in self.backbone.named_parameters():
            if any(f'blocks.{i}' in name for i in range(12)):
                param.requires_grad = False

    def forward(self, x):
        if x.is_cuda:
            with amp.autocast(device_type='cuda'):
                feats = self.backbone(x)
                return self.classifier(feats)
        else:
            feats = self.backbone(x)
            return self.classifier(feats)