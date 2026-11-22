from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class FormMaterialClassifier(nn.Module):
    """Task-specific form and material classifier; it has no volume/take outputs."""

    def __init__(self, geometry_features: int, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:  # pragma: no cover - surfaced as an environment error
            raise RuntimeError("torchvision is required for the RGB classifier") from exc
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.rgb_stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.rgb_layer1 = backbone.layer1
        self.rgb_layer2 = backbone.layer2
        self.rgb_layer3 = backbone.layer3
        self.rgb_layer4 = backbone.layer4
        self.geometry_features = int(geometry_features)

        # The three aligned channels are metric depth, depth validity, and
        # floor-relative height. Keep this branch small: the real material
        # labels are grouped by object and scarce, while RGB carries most of
        # the material signal.
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
        )

        # Material sees both appearance and spatial RGB-D cues. In particular,
        # the validity channel can represent missing depth without imposing a
        # brittle hand-written "white spot means metal/glass" rule.
        self.material_head = nn.Sequential(
            nn.Linear(512 * 2 + 64 * 2 + 12, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 6),
        )

        # Form can use calibrated shape and depth-validity measurements as well as RGB.
        self.form_geometry = nn.Sequential(
            nn.Linear(geometry_features, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
        )
        self.form_head = nn.Sequential(
            nn.Linear(512 + 64 * 2 + 64, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 2),
        )

    def forward(self, rgb: torch.Tensor, geometry: torch.Tensor,
                depth_map: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.rgb_stem(rgb)
        x = self.rgb_layer1(x)
        x = self.rgb_layer2(x)
        x = self.rgb_layer3(x)
        feature_map = self.rgb_layer4(x)

        pooled_mean = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        pooled_std = feature_map.var(dim=(2, 3), unbiased=False).clamp_min(1.0e-6).sqrt()
        rgb_mean = rgb.mean(dim=(2, 3))
        rgb_std = rgb.var(dim=(2, 3), unbiased=False).clamp_min(1.0e-6).sqrt()
        depth_features_map = self.depth_encoder(depth_map)
        depth_mean = F.adaptive_avg_pool2d(depth_features_map, 1).flatten(1)
        depth_std = depth_features_map.var(dim=(2, 3), unbiased=False).clamp_min(1.0e-6).sqrt()
        depth_stats = torch.cat((depth_map.mean(dim=(2, 3)),
                                 depth_map.var(dim=(2, 3), unbiased=False).clamp_min(1.0e-6).sqrt()), dim=1)
        material_features = torch.cat((pooled_mean, pooled_std, depth_mean, depth_std,
                                       rgb_mean, rgb_std, depth_stats), dim=1)
        form_features = torch.cat((pooled_mean, depth_mean, depth_std,
                                   self.form_geometry(geometry)), dim=1)
        return {
            "form": self.form_head(form_features),
            "material": self.material_head(material_features),
        }

    def config(self) -> dict[str, Any]:
        return {"geometry_features": self.geometry_features, "depth_channels": 3,
                "architecture": "resnet18_rgbd_form_material_v2"}
