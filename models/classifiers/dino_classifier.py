import torch.nn as nn
import torch

class DinoClassifier(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 768),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(384, num_classes)
        )

    def forward(self, x):
        # Extract features with appropriate precision
        with torch.amp.autocast(device_type='cuda'):
            feats = self.backbone(x)
        
        # Apply classifier
        return self.classifier(feats)