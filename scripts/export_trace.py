#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manga_layout4.config import load_config
from manga_layout4.model import RawExportWrapper, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/base.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if bool(args.checkpoint) == bool(args.random_init):
        raise SystemExit("choose exactly one of --checkpoint or --random-init")

    cfg = load_config(args.config)
    import torch

    # Never download backbone weights for an export-only graph build.  A formal
    # checkpoint already contains all learned parameters; random-init smoke only
    # tests graph/operator convertibility.
    model = build_model(cfg, pretrained_backbone=False)
    source = "random_init_architecture_smoke"
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
        source = str(args.checkpoint)
    model.eval().to(args.device)
    wrapper = RawExportWrapper(model).eval()
    size = int(cfg["input"]["image_size"])
    example = torch.rand(1, 3, size, size, device=args.device)
    with torch.no_grad():
        reference = wrapper(example)
    traced = torch.jit.trace(wrapper, example, strict=True)
    traced = torch.jit.freeze(traced)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.output))
    with torch.no_grad():
        got = traced(example)
    diffs = [float((left - right).abs().max().cpu()) for left, right in zip(reference, got)]
    report = {
        "schema_version": "manga-layout4-trace-v1",
        "source": source,
        "output": str(args.output),
        "shapes": [list(value.shape) for value in got],
        "max_abs_diff": diffs,
        "status": "PASS" if max(diffs, default=0.0) <= 1e-5 else "FAIL",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
