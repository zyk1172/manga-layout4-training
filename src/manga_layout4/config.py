from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("schema_version") != "manga-layout4-v1":
        raise ValueError(f"unsupported config schema: {cfg.get('schema_version')}")
    return cfg


def resolve_source_root(cfg: dict[str, Any]) -> Path:
    source = cfg["source"]
    env = os.environ.get(source.get("env_var", "MANGA_VISION_SOURCE_ROOT"))
    candidates = []
    if env:
        candidates.append(Path(env))
    for key in ("windows_candidates", "mac_candidates"):
        candidates.extend(Path(p) for p in source.get(key, []))
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    rendered = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not locate the old read-only data repository. Set MANGA_VISION_SOURCE_ROOT.\n"
        + rendered
    )


def source_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    root = resolve_source_root(cfg)
    out = {
        "root": root,
        "images": root / cfg["source"]["manga109s_images_rel"],
        "mangaseg": root / cfg["source"]["mangaseg_rel"],
        "splits": root / cfg["source"]["split_rel"],
    }
    # Historical checkouts sometimes exposed images directly below manga109s.
    if not out["images"].is_dir():
        alt = root / "data/raw/manga109s"
        if alt.is_dir():
            out["images"] = alt
    return out
