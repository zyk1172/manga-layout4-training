from __future__ import annotations
from typing import Any
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

LEVELS = ("p2", "p3", "p4", "p5")
STRIDES = (4, 8, 16, 32)
RETURN_INDICES = (1, 3, 8, 12)


class ConvBNAct(nn.Sequential):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, g: int = 1):
        super().__init__(
            nn.Conv2d(c1, c2, k, stride=s, padding=k//2, groups=g, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=False),
        )


class DSBlock(nn.Module):
    def __init__(self, c: int, stride: int = 1):
        super().__init__()
        self.dw = ConvBNAct(c, c, 3, stride, c)
        self.pw = ConvBNAct(c, c, 1)
    def forward(self, x: Tensor) -> Tensor:
        return self.pw(self.dw(x))


class MobileBackbone(nn.Module):
    out_channels = {"p2":16,"p3":24,"p4":48,"p5":576}
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = mobilenet_v3_small(weights=weights).features
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        got: dict[int, Tensor] = {}
        wanted = set(RETURN_INDICES)
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in wanted:
                got[i] = x
        return tuple(got[i] for i in RETURN_INDICES)  # type: ignore[return-value]


class DetailBranch(nn.Module):
    def __init__(self, c: int = 32):
        super().__init__()
        self.stem = ConvBNAct(3, c, 3, 2)     # stride 2
        self.refine = DSBlock(c)
        self.down = DSBlock(c, 2)             # stride 4
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        s2 = self.refine(self.stem(x))
        return s2, self.down(s2)


class P2Fusion(nn.Module):
    def __init__(self, semantic: int = 16, detail: int = 32, out: int = 96):
        super().__init__()
        half = out // 2
        self.a = ConvBNAct(semantic, half, 1)
        self.b = ConvBNAct(detail, half, 1)
        self.fuse = DSBlock(out)
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return self.fuse(torch.cat([self.a(a), self.b(b)], dim=1))


class LightFPN(nn.Module):
    def __init__(self, out: int = 96):
        super().__init__()
        inc = {"p2":out,"p3":24,"p4":48,"p5":576}
        self.lat = nn.ModuleDict({k: nn.Conv2d(v,out,1) for k,v in inc.items()})
        self.out = nn.ModuleDict({k: ConvBNAct(out,out) for k in LEVELS})
    def forward(self, feats: dict[str, Tensor]) -> dict[str, Tensor]:
        td: dict[str, Tensor] = {}
        prev = None
        for k in reversed(LEVELS):
            y = self.lat[k](feats[k])
            if prev is not None:
                y = y + F.interpolate(prev, size=y.shape[-2:], mode="nearest")
            td[k] = y
            prev = y
        return {k:self.out[k](td[k]) for k in LEVELS}


class DetectionHead(nn.Module):
    def __init__(self, c: int = 96, classes: int = 4, blocks: int = 2):
        super().__init__()
        self.cls_tower = nn.Sequential(*(DSBlock(c) for _ in range(blocks)))
        self.box_tower = nn.Sequential(*(DSBlock(c) for _ in range(blocks)))
        self.cls = nn.Conv2d(c, classes, 1)
        self.box = nn.Conv2d(c, 4, 1)
        nn.init.constant_(self.cls.bias, -4.59512)
    def forward(self, feats: dict[str, Tensor]) -> dict[str, list[Tensor]]:
        cls, box = [], []
        for k in LEVELS:
            cls.append(self.cls(self.cls_tower(feats[k])))
            box.append(self.box(self.box_tower(feats[k])))
        return {"cls_logits":cls,"bbox_reg":box}


class BalloonMaskHead(nn.Module):
    """Stride-2 binary balloon foreground head for accurate contour recovery."""
    def __init__(self, detail_c: int = 32, fpn_c: int = 96):
        super().__init__()
        self.p2 = ConvBNAct(fpn_c, 32, 1)
        self.fuse = nn.Sequential(ConvBNAct(detail_c+32,64), DSBlock(64), DSBlock(64))
        self.out = nn.Conv2d(64, 1, 1)
        nn.init.constant_(self.out.bias, -2.0)
    def forward(self, detail_s2: Tensor, p2: Tensor) -> Tensor:
        p2 = F.interpolate(self.p2(p2), size=detail_s2.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.fuse(torch.cat([detail_s2,p2], dim=1)))


class MangaLayout4(nn.Module):
    def __init__(self, num_classes: int = 4, fpn_channels: int = 96, detail_channels: int = 32,
                 head_blocks: int = 2, pretrained_backbone: bool = True,
                 mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        super().__init__()
        self.backbone = MobileBackbone(pretrained_backbone)
        self.detail = DetailBranch(detail_channels)
        self.p2_fusion = P2Fusion(16, detail_channels, fpn_channels)
        self.fpn = LightFPN(fpn_channels)
        self.det = DetectionHead(fpn_channels, num_classes, head_blocks)
        self.mask = BalloonMaskHead(detail_channels, fpn_channels)
        self.register_buffer("pixel_mean", torch.tensor(mean).view(1,3,1,1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(std).view(1,3,1,1), persistent=False)
    def forward_features(self, x: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        x = (x - self.pixel_mean) / self.pixel_std
        b2,b3,b4,b5 = self.backbone(x)
        d2,d4 = self.detail(x)
        feats = self.fpn({"p2":self.p2_fusion(b2,d4),"p3":b3,"p4":b4,"p5":b5})
        return feats, d2
    def forward(self, x: Tensor) -> dict[str, Any]:
        feats,d2 = self.forward_features(x)
        out = self.det(feats)
        out["balloon_mask_logits"] = self.mask(d2, feats["p2"])
        return out


class RawExportWrapper(nn.Module):
    def __init__(self, model: MangaLayout4):
        super().__init__(); self.model=model
    def forward(self, x: Tensor):
        o=self.model(x)
        return (
            o["cls_logits"][0],o["bbox_reg"][0],
            o["cls_logits"][1],o["bbox_reg"][1],
            o["cls_logits"][2],o["bbox_reg"][2],
            o["cls_logits"][3],o["bbox_reg"][3],
            o["balloon_mask_logits"],
        )


def build_model(cfg: dict[str, Any], pretrained_backbone: bool | None = None) -> MangaLayout4:
    m=cfg["model"]; n=cfg["input"]["normalization"]
    if pretrained_backbone is None:
        pretrained_backbone = str(m.get("backbone_init", "imagenet1k_v1")).lower() != "random"
    return MangaLayout4(
        num_classes=int(m["num_classes"]), fpn_channels=int(m["fpn_channels"]),
        detail_channels=int(m["detail_channels"]), head_blocks=int(m["head_blocks"]),
        pretrained_backbone=pretrained_backbone,
        mean=tuple(n["mean"]), std=tuple(n["std"]),
    )
