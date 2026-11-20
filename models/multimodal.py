from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


FUSION_MODES = {"rgb_only", "late", "intermediate", "intermediate_geom"}


class MultiModalAttributeModel(nn.Module):
    """Pretrained RGB encoder with explicit depth and geometry fusion options."""

    def __init__(self, geometry_features: int, fusion: str = "intermediate_geom",
                 pretrained: bool = False) -> None:
        super().__init__()
        if fusion not in FUSION_MODES:
            raise ValueError(f"fusion must be one of {sorted(FUSION_MODES)}")
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:  # pragma: no cover - surfaced as an environment error
            raise RuntimeError("torchvision is required for the pretrained RGB encoder") from exc
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.rgb_stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.rgb_layer1 = backbone.layer1
        self.rgb_layer2 = backbone.layer2
        self.rgb_layer3 = backbone.layer3
        self.rgb_layer4 = backbone.layer4
        self.fusion = fusion
        self.geometry_features = int(geometry_features)

        self.geom_mlp: nn.Module | None = None
        if fusion in {"late", "intermediate_geom"}:
            self.geom_mlp = nn.Sequential(
                nn.Linear(geometry_features, 64), nn.LayerNorm(64), nn.GELU(),
                nn.Linear(64, 64), nn.GELU(),
            )

        self.depth_encoder: nn.Module | None = None
        self.spatial_fusion: nn.Module | None = None
        if fusion in {"intermediate", "intermediate_geom"}:
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(24), nn.SiLU(inplace=True),
                nn.Conv2d(24, 48, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(48), nn.SiLU(inplace=True),
                nn.Conv2d(48, 96, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(96), nn.SiLU(inplace=True),
                nn.Conv2d(96, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.SiLU(inplace=True),
            )
            self.spatial_fusion = nn.Sequential(
                nn.Conv2d(512 + 128, 256, 1, bias=False),
                nn.BatchNorm2d(256), nn.GELU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )

        if fusion == "rgb_only":
            fused_dim = 512
        elif fusion == "late":
            fused_dim = 512 + 64
        elif fusion == "intermediate":
            fused_dim = 256
        else:
            fused_dim = 256 + 64
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.form_head = nn.Linear(128, 2)
        self.material_head = nn.Linear(128, 6)
        self.volume_head = nn.Linear(128, 1)
        self.mass_head = nn.Linear(128, 1)
        self.take_head = nn.Linear(128, 1)

        # The pretrained texture backbone adapts more slowly than new task heads.
        self.pretrained = bool(pretrained)

    def forward(self, rgb: torch.Tensor, geometry: torch.Tensor,
                depth_map: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.rgb_stem(rgb)
        x = self.rgb_layer1(x)
        x = self.rgb_layer2(x)
        x = self.rgb_layer3(x)
        x = self.rgb_layer4(x)
        rgb_vec = F.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.fusion == "rgb_only":
            fused = rgb_vec
        elif self.fusion == "late":
            assert self.geom_mlp is not None
            fused = torch.cat((rgb_vec, self.geom_mlp(geometry)), dim=1)
        else:
            if depth_map is None:
                raise ValueError(f"depth_map is required for fusion={self.fusion}")
            assert self.depth_encoder is not None and self.spatial_fusion is not None
            depth_features = self.depth_encoder(depth_map)
            depth_features = F.interpolate(depth_features, size=x.shape[-2:], mode="bilinear", align_corners=False)
            spatial = self.spatial_fusion(torch.cat((x, depth_features), dim=1))
            fused = spatial
            if self.fusion == "intermediate_geom":
                assert self.geom_mlp is not None
                fused = torch.cat((spatial, self.geom_mlp(geometry)), dim=1)
        h = self.trunk(fused)
        return {
            "form": self.form_head(h),
            "material": self.material_head(h),
            "volume_log": self.volume_head(h).squeeze(1),
            "mass_log": self.mass_head(h).squeeze(1),
            "take_logit": self.take_head(h).squeeze(1),
        }

    def config(self) -> dict[str, Any]:
        return {"geometry_features": self.geometry_features, "fusion": self.fusion}

