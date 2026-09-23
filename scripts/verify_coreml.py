#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--coreml", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()

    import torch
    import coremltools as ct

    traced = torch.jit.load(str(args.trace), map_location="cpu").eval()
    coreml = ct.models.MLModel(str(args.coreml), compute_units=ct.ComputeUnit.ALL)
    names = [
        "p2_cls", "p2_bbox", "p3_cls", "p3_bbox", "p4_cls", "p4_bbox",
        "p5_cls", "p5_bbox", "balloon_mask",
    ]
    diffs = []
    for seed in range(args.samples):
        generator = torch.Generator().manual_seed(9000 + seed)
        x = torch.rand((1, 3, 640, 640), generator=generator)
        with torch.no_grad():
            reference = traced(x)
        got = coreml.predict({"image": x.numpy().astype(np.float32)})
        diffs.append(
            {name: float(np.max(np.abs(got[name] - ref.numpy()))) for name, ref in zip(names, reference)}
        )
    worst = max(value for row in diffs for value in row.values())
    report = {
        "schema_version": "manga-layout4-coreml-equivalence-v1",
        "trace": str(args.trace),
        "coreml": str(args.coreml),
        "max_abs_diff": worst,
        "per_sample": diffs,
        "status": "PASS" if worst <= 5e-3 else "FAIL",
    }
    output = args.output_report or (args.coreml.parent / "coreml-equivalence.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
