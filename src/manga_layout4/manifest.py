from __future__ import annotations
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from . import CLASSES
from .config import source_paths
from .rle import RLEError, mask_extent


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(c for c in value if not unicodedata.category(c).startswith(("P", "Z")))


def _load_array(path: Path, key: str) -> list[dict[str, Any]]:
    # MangaSeg JSON snapshots are cumulative. Parse only the requested top-level
    # array so memory use stays bounded.
    text = path.read_text(encoding="utf-8")
    marker = f'"{key}"'
    i = text.find(marker)
    if i < 0:
        raise ValueError(f"missing {key} in {path}")
    start = text.find("[", i + len(marker))
    if start < 0:
        raise ValueError(f"{key} is not an array in {path}")
    value, _ = json.JSONDecoder().raw_decode(text, start)
    return value


def _read_books(split_root: Path, split: str) -> set[str]:
    path = split_root / f"{split}_books.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return {x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}


def _image_path(image_root: Path, file_name: str) -> Path | None:
    # Try exact relative path, then common zero-padding variants.
    direct = image_root / file_name
    if direct.is_file():
        return direct
    p = Path(file_name)
    try:
        n = int(p.stem)
    except ValueError:
        return None
    for stem in (f"{n:03d}", f"{n:04d}", str(n)):
        candidate = image_root / p.parent / f"{stem}{p.suffix}"
        if candidate.is_file():
            return candidate
    # Some historical trees place JPGs directly below raw/manga109s/<book>.
    if image_root.name == "manga109s":
        for stem in (f"{n:03d}", f"{n:04d}", str(n)):
            candidate = image_root / p.parent / f"{stem}{p.suffix}"
            if candidate.is_file():
                return candidate
    return None


def build_manifest(cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    paths = source_paths(cfg)
    json_root = paths["mangaseg"] / "jsons"
    files = sorted(json_root.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no MangaSeg JSON under {json_root}")
    authoritative = next((p for p in files if p.stem == "YumeiroCooking"), files[-1])
    meta = json.loads(authoritative.read_text(encoding="utf-8"))
    categories = {int(c["id"]): str(c["name"]) for c in meta.get("categories", [])}
    # Normalize spelling variants in source metadata.
    source_to_target = {}
    for cid, name in categories.items():
        n = name.casefold().strip()
        if n in {"text/dialog", "text", "dialog"}:
            source_to_target[cid] = "text"
        elif n in {"frame", "panel", "frame/panel"}:
            source_to_target[cid] = "frame"
        elif n in {"balloon", "speech balloon"}:
            source_to_target[cid] = "balloon"
        elif n in {"onomatopoeia", "sfx", "sound effect"}:
            source_to_target[cid] = "onomatopoeia"
    missing = set(CLASSES) - set(source_to_target.values())
    if missing:
        raise RuntimeError(f"MangaSeg category map missing: {sorted(missing)}; got {categories}")

    image_by_id = {int(i["id"]): i for i in meta.get("images", [])}
    del meta
    split_books = {s: _read_books(paths["splits"], s) for s in ("train", "val", "test")}
    book_to_split = {}
    for split, books in split_books.items():
        for book in books:
            if book in book_to_split:
                raise RuntimeError(f"book in multiple splits: {book}")
            book_to_split[book] = split
    normalized_book = {_normalize(k): k for k in book_to_split}

    exclude_books = set(cfg["split_policy"].get("exclude_dimension_mismatch_books", []))
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[int] = set()
    stats = Counter()
    bad: list[dict[str, Any]] = []

    for json_path in files:
        for ann in _load_array(json_path, "annotations"):
            try:
                aid = int(ann["id"]); cid = int(ann["category_id"]); iid = int(ann["image_id"])
            except (KeyError, TypeError, ValueError):
                stats["malformed_annotation"] += 1
                continue
            target_name = source_to_target.get(cid)
            if target_name is None:
                continue
            if aid in seen:
                stats["duplicate_annotation_id"] += 1
                continue
            seen.add(aid)
            image = image_by_id.get(iid)
            if image is None:
                stats["missing_image_metadata"] += 1
                continue
            try:
                bbox, area, (mw, mh) = mask_extent(ann.get("segmentation"))
            except (RLEError, ValueError) as exc:
                bad.append({"annotation_id": aid, "error": str(exc), "json": json_path.name})
                stats["invalid_rle"] += 1
                continue
            iw, ih = int(image["width"]), int(image["height"])
            if (mw, mh) != (iw, ih):
                stats["rle_dimension_mismatch"] += 1
                bad.append({"annotation_id": aid, "rle": [mw,mh], "image": [iw,ih]})
                continue
            record = {
                "annotation_id": aid,
                "class_name": target_name,
                "bbox_xyxy": bbox,
                "area": area,
            }
            # V1 needs pixel supervision only for balloon.  Keeping RLE for
            # frame/text/SFX would make every epoch carry large unused strings.
            if target_name == "balloon":
                record["segmentation"] = ann["segmentation"]
            anns_by_image[iid].append(record)
            stats[f"annotation_{target_name}"] += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    by_split = Counter()
    page_class_counts = Counter()
    missing_local = []
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for iid, image in sorted(image_by_id.items()):
            file_name = str(image.get("file_name", "")).replace("\\", "/")
            if "/" not in file_name:
                continue
            source_book = file_name.split("/", 1)[0]
            book = source_book if source_book in book_to_split else normalized_book.get(_normalize(source_book))
            if book is None or book in exclude_books:
                continue
            split = book_to_split[book]
            local = _image_path(paths["images"], file_name.replace(source_book, book, 1))
            if local is None:
                if len(missing_local) < 100:
                    missing_local.append(file_name)
                by_split[f"{split}_missing_local_image"] += 1
                continue
            annotations = anns_by_image.get(iid, [])
            # Keep empty pages too: they are valid negative-background supervision.
            names = Counter(a["class_name"] for a in annotations)
            for name, n in names.items():
                page_class_counts[f"{split}_{name}"] += n
            row = {
                "schema_version": "manga-layout4-page-v1",
                "image_id": iid,
                "book_id": book,
                "split": split,
                "file_name": file_name.replace(source_book, book, 1),
                "image_path": str(local.resolve()),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "annotations": annotations,
                "has_balloon": names["balloon"] > 0,
                "has_onomatopoeia": names["onomatopoeia"] > 0,
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            records += 1
            by_split[split] += 1

    report = {
        "schema_version": "manga-layout4-manifest-report-v1",
        "source_root": str(paths["root"]),
        "mangaseg_json_count": len(files),
        "authoritative_snapshot": authoritative.name,
        "source_categories": categories,
        "target_classes": list(CLASSES),
        "pages": dict(by_split),
        "annotations": {name: stats[f"annotation_{name}"] for name in CLASSES},
        "page_class_counts": dict(page_class_counts),
        "invalid_rle": stats["invalid_rle"],
        "rle_dimension_mismatch": stats["rle_dimension_mismatch"],
        "duplicate_annotation_ids": stats["duplicate_annotation_id"],
        "missing_local_examples": missing_local,
        "excluded_books": sorted(exclude_books),
        "legacy_test_observed": True,
        "formal_training_uses_test": False,
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return report


def iter_manifest(path: Path, split: str | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if split is None or row.get("split") == split:
                yield row
