# On-device checklist

## Windows — one-time preparation

```powershell
cd F:\codex\manga-layout4-training
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e .[dev]
$env:MANGA_VISION_SOURCE_ROOT='F:\codex\manga-model'
python scripts/cache_backbone.py
```

`cache_backbone.py` should finish before a disconnected/offline formal run. It only retrieves the generic torchvision ImageNet backbone; it never downloads an old manga checkpoint.

## Required gates

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_gates.ps1
```

Expected artifacts after PASS:

- `data/manifests/pages.jsonl`
- `data/manifests/pages.report.json`
- `data/manifests/audit.json`
- `outputs/preflight/report.json`
- `outputs/overfit/report.json`

Do not start formal training if any gate fails. Fix the cause and rerun that gate.

## Mac architecture export gate before formal training

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[mac,dev]'
python scripts/export_trace.py --random-init --output outputs/export-smoke/model_raw.pt
python scripts/export_coreml.py --trace outputs/export-smoke/model_raw.pt --output outputs/export-smoke/MangaLayout4V1-Smoke.mlpackage
python scripts/verify_coreml.py --trace outputs/export-smoke/model_raw.pt --coreml outputs/export-smoke/MangaLayout4V1-Smoke.mlpackage --output-report outputs/export-smoke/coreml-equivalence.json
```

Copy/sync the PASS report back to the Windows project at the same relative path.

## Formal run

```powershell
python scripts/train.py --config configs/base.yaml --device cuda --output outputs/formal-v1
```

The trainer automatically uses the micro-batch/gradient-accumulation recommendation saved by preflight and refuses to start if the Core ML architecture gate is absent.

## Mac final export after a candidate exists

```bash
python scripts/export_trace.py --checkpoint outputs/formal-v1/best.pt --output outputs/export/model_raw.pt
python scripts/export_coreml.py --trace outputs/export/model_raw.pt --output outputs/export/MangaLayout4V1.mlpackage
python scripts/verify_coreml.py --trace outputs/export/model_raw.pt --coreml outputs/export/MangaLayout4V1.mlpackage --output-report outputs/export/coreml-equivalence.json
```

Only after FP32 equivalence passes should FP16 or other compression be tested as a separate deployment experiment.

## Expected first device-only adjustments

Normal small adjustments are: preflight-selected micro-batch, `num_workers` if Windows I/O behaves differently, and potentially learning-rate scaling if effective batch cannot remain 8. Any change to class definitions, data split, assignment rule, loss family, or augmentation semantics is not a "small adjustment" and should be a new recorded experiment.
