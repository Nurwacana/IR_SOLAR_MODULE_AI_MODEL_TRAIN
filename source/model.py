"""
Definisi Arsitektur Model Deep Learning untuk Klasifikasi 12 Kelas Citra Modul Surya
Menyediakan pilihan model dari yang sangat ringan (Compact CNN, MobileNetV3)
hingga model menengah (ResNet-18, EfficientNet-B0) dan mendalam (ResNet-50).
"""

import torch
import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES


class SqueezeExcitationBlock(nn.Module):
    """Squeeze-and-Excitation (SE) Block untuk memberikan atensi adaptif pada fitur anomali."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        weights = self.fc(x).view(b, c, 1, 1)
        return x * weights


# ==============================================================================
# 1. Custom Compact CNN (~400K Parameters) - PALING RINGAN (Ultra-Lightweight)
# ==============================================================================
class SolarCompactCNN(nn.Module):
    """
    Arsitektur konvolusi ringkas khusus citra termal modul surya (~450 ribu parameter).
    Dapat dilatih sangat cepat dari scratch tanpa perlu download bobot ImageNet.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # Blok 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 -> 56

            # Blok 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 -> 28

            # Blok 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28 -> 14

            # Blok 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SqueezeExcitationBlock(256, reduction=16),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.classifier(feats)


# ==============================================================================
# 2. MobileNetV3-Small (~2.5M Parameters) - SANGAT RINGAN & CEPAT
# ==============================================================================
class SolarMobileNetV3(nn.Module):
    """
    MobileNetV3-Small: Dirancang untuk efisiensi tinggi dengan Depthwise Separable Conv.
    Sangat cocok untuk inferensi real-time di drone / edge device.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
        except Exception:
            self.backbone = models.mobilenet_v3_small(weights=None)

        in_features = self.backbone.classifier[0].in_features  # 576
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ==============================================================================
# 3. EfficientNet-B0 (~4.0M Parameters) - EFISIENSI TINGGI DENGAN SE-BLOCK
# ==============================================================================
class SolarEfficientNet(nn.Module):
    """EfficientNet-B0 dengan compound scaling dan SE-block bawaan."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
        except Exception:
            self.backbone = models.efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ==============================================================================
# 4. ResNet-18 (~11.2M Parameters) - RINGAN, STABIL & AKURAT (Acuan Literatur)
# ==============================================================================
class SolarResNet18(nn.Module):
    """
    ResNet-18: Model residual yang jauh lebih ringan dibanding ResNet-50 (~11M params).
    Sangat stabil dalam konvergensi dan terbukti optimal pada dataset modul surya.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except Exception:
            self.backbone = models.resnet18(weights=None)

        in_features = self.backbone.fc.in_features  # 512 kanal fitur
        self.se_block = SqueezeExcitationBlock(channels=in_features, reduction=16)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(in_features * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)

        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)

        features = self.se_block(features)
        avg_out = self.avg_pool(features).flatten(1)
        max_out = self.max_pool(features).flatten(1)
        pooled = torch.cat([avg_out, max_out], dim=1)

        return self.classifier(pooled)


# ==============================================================================
# 5. ResNet-50 (~25.5M Parameters) - DEEP & HEAVY
# ==============================================================================
class SolarResNet50(nn.Module):
    """ResNet-50 Deep Residual Network dengan SE Attention Head."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        except Exception:
            self.backbone = models.resnet50(weights=None)

        in_features = self.backbone.fc.in_features  # 2048 kanal fitur
        self.se_block = SqueezeExcitationBlock(channels=in_features, reduction=16)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(in_features * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)

        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)

        features = self.se_block(features)
        avg_out = self.avg_pool(features).flatten(1)
        max_out = self.max_pool(features).flatten(1)
        pooled = torch.cat([avg_out, max_out], dim=1)

        return self.classifier(pooled)


# ==============================================================================
# Factory Builder
# ==============================================================================
SUPPORTED_MODELS = {
    "compact_cnn": SolarCompactCNN,
    "mobilenetv3": SolarMobileNetV3,
    "efficientnet": SolarEfficientNet,
    "resnet18": SolarResNet18,
    "resnet50": SolarResNet50,
}


def build_model(
    model_type: str = "resnet18",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """
    Factory function untuk inisialisasi model.

    Pilihan model_type:
      - 'compact_cnn' : ~450K params  (Paling ringan, dilatih dari scratch)
      - 'mobilenetv3' : ~2.5M params  (Sangat ringan, mobile-optimized)
      - 'efficientnet': ~4.0M params  (Efisien & akurat, SE-block)
      - 'resnet18'    : ~11.2M params (Paling seimbang & stabil, acuan utama)
      - 'resnet50'    : ~25.5M params (Model dalam/berat)
    """
    model_key = model_type.lower().replace("-", "_")

    if model_key == "compact_cnn":
        return SolarCompactCNN(num_classes=num_classes)
    elif model_key in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_key](num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(
            f"Tipe model '{model_type}' tidak didukung. "
            f"Pilih salah satu dari: {list(SUPPORTED_MODELS.keys())}"
        )
