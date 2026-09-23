#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manga_layout4 import CLASSES
from manga_layout4.config import load_config
from manga_layout4.manifest import iter_manifest
from manga_layout4.rle import parse_rle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/base.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/pages.jsonl")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not args.manifest.is_file():
        raise SystemExit("manifest missing; run build_manifest.py first")

    pages = Counter()
    anns = Counter()
    anns_by_split = Counter()
    errors: list[str] = []
    seen = set()
    excluded = set(cfg["split_policy"]["exclude_dimension_mismatch_books"])

    for row in iter_manifest(args.manifest):
        split = row["split"]
        pages[split] += 1
        key = (row["book_id"], row["file_name"])
        if key in seen:
            errors.append(f"duplicate page {key}")
        seen.add(key)
        if row["book_id"] in excluded:
            errors.append(f"excluded book leaked: {row['book_id']}")
        if not Path(row["image_path"]).is_file():
            errors.append(f"missing image {row['image_path']}")
        width, height = int(row["width"]), int(row["height"])
        if width <= 0 or height <= 0:
            errors.append(f"invalid page size {row['image_id']}: {width}x{height}")

        for ann in row["annotations"]:
            name = ann["class_name"]
            anns[name] += 1
            anns_by_split[(split, name)] += 1
            if name not in CLASSES:
                errors.append(f"unknown class {name}")
                continue
            box = ann.get("bbox_xyxy")
            if not isinstance(box, list) or len(box) != 4:
                errors.append(f"malformed bbox {ann.get('annotation_id')}")
                continue
            x1, y1, x2, y2 = map(float, box)
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                errors.append(f"invalid bbox {row['image_id']}/{ann.get('annotation_id')}: {box}")
            if name == "balloon":
                if "segmentation" not in ann:
                    errors.append(f"balloon RLE missing {ann.get('annotation_id')}")
                else:
                    try:
                        h, w, _ = parse_rle(ann["segmentation"])
                        if (w, h) != (width, height):
                            errors.append(
                                f"RLE/page mismatch {row['image_id']}/{ann['annotation_id']}"
                            )
                    except Exception as exc:
                        errors.append(f"RLE error {ann.get('annotation_id')}: {exc}")
            elif "segmentation" in ann:
                errors.append(f"unexpected non-balloon RLE {ann.get('annotation_id')}/{name}")

    expected = cfg["split_policy"].get("expected_pages", {})
    for split in ("train", "val", "test"):
        if split in expected and pages[split] != int(expected[split]):
            errors.append(
                f"{split} page coverage mismatch: {pages[split]} != expected {int(expected[split])}"
            )
    for split in ("train", "val"):
        for class_name in CLASSES:
            if anns_by_split[(split, class_name)] <= 0:
                errors.append(f"no {split} annotations for {class_name}")

    report = {
        "schema_version": "manga-layout4-audit-v1",
        "status": "PASS" if not errors else "FAIL",
        "pages": dict(pages),
        "annotations": dict(anns),
        "annotations_by_split": {
            split: {class_name: anns_by_split[(split, class_name)] for class_name in CLASSES}
            for split in ("train", "val", "test")
        },
        "errors": errors[:200],
        "legacy_test_observed": True,
        "formal_training_allowed_splits": ["train", "val"],
    }
    out = ROOT / "data/manifests/audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
