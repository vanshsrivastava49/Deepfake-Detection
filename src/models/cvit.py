import torch.nn as nn
from torchvision.models import resnet18
from einops.layers.torch import Rearrange

class CViT(nn.Module):
    def __init__(self, num_classes=2, img_size=224, patch=7, dim=512, depth=6, heads=8, mlp_dim=2048):
        super().__init__()
        backbone = resnet18(weights=None)
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])  # feature extractor

        self.to_patch = Rearrange('b c h w -> b (h w) c')
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=heads, dim_feedforward=mlp_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)          # [B, 512, 7, 7]
        x = self.to_patch(x)     # [B, 49, 512]
        x = self.transformer(x)  # transformer over patches
        x = x.mean(dim=1)        # average pooling
        return self.classifier(x)
