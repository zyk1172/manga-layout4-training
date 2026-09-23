from __future__ import annotations
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, WeightedRandomSampler

from . import CLASS_TO_ID
from .manifest import iter_manifest
from .rle import decode_mask


def _intersection_fraction(box: list[float], crop: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    area = max(0.0, (x2 - x1) * (y2 - y1))
    ix1 = max(x1, crop[0])
    iy1 = max(y1, crop[1])
    ix2 = min(x2, crop[2])
    iy2 = min(y2, crop[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / max(area, 1e-6)


def _letterbox_with_jitter(
    image: Image.Image,
    balloon: Image.Image,
    boxes: list[list[float]],
    size: int,
    scale_jitter: float,
    translate_fraction: float,
    train: bool,
) -> tuple[Image.Image, Image.Image, list[list[float]], list[int]]:
    """Resize to a square canvas with bounded translation and optional mild zoom.

    `scale_jitter=1` is ordinary letterbox.  For training, jitter may zoom in/out,
    but the center can move by at most `translate_fraction * size` from the
    centered placement.  This deliberately avoids arbitrary object crops.
    """
    w, h = image.size
    base = min(size / w, size / h)
    scale = base * scale_jitter
    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))
    image = image.resize((nw, nh), Image.Resampling.BILINEAR)
    balloon = balloon.resize((nw, nh), Image.Resampling.NEAREST)

    center_x = (size - nw) // 2
    center_y = (size - nh) // 2
    if train and translate_fraction > 0:
        max_shift = round(size * translate_fraction)
        ox = center_x + random.randint(-max_shift, max_shift)
        oy = center_y + random.randint(-max_shift, max_shift)
    else:
        ox, oy = center_x, center_y

    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    mask = Image.new("L", (size, size), 0)
    canvas.paste(image, (ox, oy))
    mask.paste(balloon, (ox, oy))

    out: list[list[float]] = []
    kept: list[int] = []
    for i, b in enumerate(boxes):
        t = [
            b[0] * scale + ox,
            b[1] * scale + oy,
            b[2] * scale + ox,
            b[3] * scale + oy,
        ]
        old = max(1e-6, (t[2] - t[0]) * (t[3] - t[1]))
        clipped = [
            max(0.0, t[0]),
            max(0.0, t[1]),
            min(float(size), t[2]),
            min(float(size), t[3]),
        ]
        new = max(0.0, (clipped[2] - clipped[0]) * (clipped[3] - clipped[1]))
        if clipped[2] > clipped[0] and clipped[3] > clipped[1] and new / old >= 0.60:
            out.append(clipped)
            kept.append(i)
    return canvas, mask, out, kept


class MangaLayoutDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        cfg: dict[str, Any],
        split: str,
        training: bool,
        rows: list[dict[str, Any]] | None = None,
    ):
        self.cfg = cfg
        self.split = split
        self.training = training
        self.rows = rows if rows is not None else list(iter_manifest(Path(manifest), split))
        self.size = int(cfg["input"]["image_size"])
        self.stage1 = True

    def set_stage1(self, enabled: bool):
        self.stage1 = bool(enabled)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if image.size != (int(row["width"]), int(row["height"])):
            raise RuntimeError(
                f"image dimension mismatch {row['image_id']}: {image.size} vs {(row['width'], row['height'])}"
            )
        anns = row["annotations"]
        boxes = [list(map(float, a["bbox_xyxy"])) for a in anns]
        labels = [CLASS_TO_ID[a["class_name"]] for a in anns]

        # Only balloon RLE is retained in the compact manifest.  Other classes
        # are box targets, so they never need to materialize a full mask here.
        balloon_np = np.zeros((image.height, image.width), dtype=np.uint8)
        for a in anns:
            if a["class_name"] == "balloon":
                segmentation = a.get("segmentation")
                if segmentation is None:
                    raise RuntimeError(f"balloon mask missing for annotation {a.get('annotation_id')}")
                balloon_np |= decode_mask(segmentation)
        balloon = Image.fromarray(balloon_np * 255, mode="L")

        # A half-spread is the only content crop allowed in V1.  It is used only
        # on sufficiently wide pages and requires >=60% of each retained box.
        s = self.cfg["sampling"]
        if (
            self.training
            and image.width / image.height >= float(s["half_spread_min_aspect"])
            and random.random() < float(s["half_spread_probability"])
        ):
            mid = image.width // 2
            side = random.randint(0, 1)
            crop = (0, 0, mid, image.height) if side == 0 else (mid, 0, image.width, image.height)
            keep = [i for i, b in enumerate(boxes) if _intersection_fraction(b, crop) >= 0.60]
            new_boxes: list[list[float]] = []
            new_labels: list[int] = []
            for i in keep:
                b = boxes[i]
                clipped = [
                    max(b[0], crop[0]) - crop[0],
                    max(b[1], crop[1]) - crop[1],
                    min(b[2], crop[2]) - crop[0],
                    min(b[3], crop[3]) - crop[1],
                ]
                if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                    new_boxes.append(clipped)
                    new_labels.append(labels[i])
            boxes, labels = new_boxes, new_labels
            image = image.crop(crop)
            balloon = balloon.crop(crop)

        scale_jitter = 1.0
        translate_fraction = 0.0
        if self.training and self.stage1:
            aug = self.cfg["augmentation"]
            brightness = float(aug["brightness"])
            contrast = float(aug["contrast"])
            image = ImageEnhance.Brightness(image).enhance(1.0 + random.uniform(-brightness, brightness))
            image = ImageEnhance.Contrast(image).enhance(1.0 + random.uniform(-contrast, contrast))
            scale_jitter = random.uniform(float(aug["scale_min"]), float(aug["scale_max"]))
            translate_fraction = float(aug["translate_fraction"])

        image, balloon, boxes, kept = _letterbox_with_jitter(
            image,
            balloon,
            boxes,
            self.size,
            scale_jitter,
            translate_fraction,
            self.training and self.stage1,
        )
        labels = [labels[i] for i in kept]

        arr = np.asarray(image, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr.copy()).permute(2, 0, 1)
        boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.tensor(labels, dtype=torch.long)
        mask = balloon.resize((self.size // 2, self.size // 2), Image.Resampling.NEAREST)
        mask_t = torch.from_numpy((np.asarray(mask) > 127).astype(np.float32)).unsqueeze(0)
        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "balloon_mask": mask_t,
            "image_id": row["image_id"],
            "book_id": row["book_id"],
            "split": row["split"],
        }
        return x, target


def collate(batch):
    return torch.stack([x for x, _ in batch]), [t for _, t in batch]


def weighted_sampler(dataset: MangaLayoutDataset, cfg: dict[str, Any], seed: int):
    s = cfg["sampling"]
    weights = []
    for row in dataset.rows:
        weight = 1.0
        if row.get("has_onomatopoeia"):
            weight *= float(s["onomatopoeia_page_weight"])
        if row.get("has_balloon"):
            weight *= float(s["balloon_page_weight"])
        weights.append(weight)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        len(weights),
        replacement=True,
        generator=generator,
    )
