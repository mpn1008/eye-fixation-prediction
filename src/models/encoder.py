from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models import ResNet101_Weights, resnet101


@dataclass
class EncoderFeatures:
    """Container for the five feature levels produced by the encoder."""

    s1: torch.Tensor  # stem  →  64 ch, H/2  × W/2
    s2: torch.Tensor  # layer1 → 256 ch, H/4  × W/4
    s3: torch.Tensor  # layer2 → 512 ch, H/8  × W/8
    s4: torch.Tensor  # layer3 → 1024 ch, H/16 × W/16
    s5: torch.Tensor  # layer4 → 2048 ch, H/32 × W/32


class ResNetEncoder(nn.Module):
    """
    Pretrained ResNet-50 split into five progressive stages.

    Input : (B, 3, H, W)
    Output: EncoderFeatures with tensors at 5 scales.
    """

    # channels produced at each stage
    out_channels: tuple[int, ...] = (64, 256, 512, 1024, 2048)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # weights = ResNet50_Weights.DEFAULT if pretrained else None
        # backbone = resnet50(weights=weights)
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        # Stage 0 – stem (conv1 + bn1 + relu + maxpool)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # → 256 ch
        self.layer2 = backbone.layer2  # → 512 ch
        self.layer3 = backbone.layer3  # → 1024 ch
        self.layer4 = backbone.layer4  # → 2048 ch

    def forward(self, x: torch.Tensor) -> EncoderFeatures:
        s1 = self.stem(x)  # H/4  (maxpool halves after the /2 conv)
        s2 = self.layer1(s1)  # H/4
        s3 = self.layer2(s2)  # H/8
        s4 = self.layer3(s3)  # H/16
        s5 = self.layer4(s4)  # H/32
        return EncoderFeatures(s1=s1, s2=s2, s3=s3, s4=s4, s5=s5)
