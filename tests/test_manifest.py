import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from manga_layout4.manifest import build_manifest, iter_manifest

ROOT = Path(__file__).resolve().parents[1]


def _rle(mask: np.ndarray) -> dict:
    h, w = mask.shape
    flat = mask.reshape(-1, order="F")
    counts = []
    current = 0
    run = 0
    for bit in flat.tolist():
        bit = int(bit)
        if bit == current:
            run += 1
        else:
            counts.append(run)
            current = bit
            run = 1
    counts.append(run)
    return {"size": [h, w], "counts": counts}


def test_build_manifest_keeps_only_balloon_rle(tmp_path, monkeypatch):
    source = tmp_path / "old"
    images = source / "data/raw/manga109s/images/BookA"
    segroot = source / "data/external/MangaSegmentation/jsons"
    splits = source / "data/splits"
    images.mkdir(parents=True)
    segroot.mkdir(parents=True)
    splits.mkdir(parents=True)
    Image.new("RGB", (12, 10), "white").save(images / "000.jpg")
    (splits / "train_books.txt").write_text("BookA\n", encoding="utf-8")
    (splits / "val_books.txt").write_text("", encoding="utf-8")
    (splits / "test_books.txt").write_text("", encoding="utf-8")

    categories = [
        {"id": 1, "name": "frame"},
        {"id": 2, "name": "text"},
        {"id": 3, "name": "face"},
        {"id": 4, "name": "body"},
        {"id": 5, "name": "balloon"},
        {"id": 6, "name": "onomatopoeia"},
    ]
    annotations = []
    for aid, cid, x1 in [(1, 1, 0), (2, 2, 2), (3, 5, 4), (4, 6, 6)]:
        mask = np.zeros((10, 12), dtype=np.uint8)
        mask[2:6, x1 : min(x1 + 4, 12)] = 1
        annotations.append(
            {"id": aid, "image_id": 10, "category_id": cid, "segmentation": _rle(mask)}
        )
    payload = {
        "images": [{"id": 10, "width": 12, "height": 10, "file_name": "BookA/000.jpg"}],
        "categories": categories,
        "annotations": annotations,
    }
    (segroot / "YumeiroCooking.json").write_text(json.dumps(payload), encoding="utf-8")

    cfg = yaml.safe_load((ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
    monkeypatch.setenv("MANGA_VISION_SOURCE_ROOT", str(source))
    output = tmp_path / "pages.jsonl"
    report = build_manifest(cfg, output)
    assert report["pages"]["train"] == 1
    row = next(iter(iter_manifest(output, "train")))
    assert {a["class_name"] for a in row["annotations"]} == {
        "frame",
        "text",
        "balloon",
        "onomatopoeia",
    }
    for ann in row["annotations"]:
        assert ("segmentation" in ann) == (ann["class_name"] == "balloon")
