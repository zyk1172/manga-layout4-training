# MangaLayout4 Training

Independent training project for four manga-structure classes:

`frame / text / balloon / onomatopoeia`

`face` and `body` are intentionally not targets. Balloon additionally gets a high-resolution foreground mask so the app can recover real balloon contours.

## Important separation from the old project

This repository is intended to be created as a **new repository**, not a branch of `manga-vision-training`. It reads the existing local datasets read-only through `MANGA_VISION_SOURCE_ROOT`. It does not modify V2B5, V3, mReader, or the old training data.

## Quick start on the existing Windows host

```powershell
# Put this new repo somewhere independent, for example F:\codex\manga-layout4-training
cd F:\codex\manga-layout4-training
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the same CUDA PyTorch build already proven on this host, then project deps.
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e .[dev]

$env:MANGA_VISION_SOURCE_ROOT='F:\codex\manga-model'
python scripts/cache_backbone.py
python scripts/build_manifest.py --config configs/base.yaml
python scripts/audit_dataset.py --config configs/base.yaml
python scripts/preflight.py --config configs/base.yaml --device cuda
python scripts/overfit_gate.py --config configs/base.yaml --device cuda
python scripts/train.py --config configs/base.yaml --device cuda --output outputs/formal-v1
```

Do not start `train.py` until the audit, real-model batch preflight, and overfit gate pass.

## Mac Core ML architecture gate — before the long run

After the Windows data/CUDA/overfit gates pass, copy/sync this independent project to the Mac and validate the **untrained architecture graph**. This tests operator convertibility before spending hours on formal training.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[mac,dev]'
python scripts/export_trace.py --config configs/base.yaml --random-init --output outputs/export-smoke/model_raw.pt
python scripts/export_coreml.py --config configs/base.yaml --trace outputs/export-smoke/model_raw.pt --output outputs/export-smoke/MangaLayout4V1-Smoke.mlpackage
python scripts/verify_coreml.py --trace outputs/export-smoke/model_raw.pt --coreml outputs/export-smoke/MangaLayout4V1-Smoke.mlpackage --output-report outputs/export-smoke/coreml-equivalence.json
```

Sync/copy `outputs/export-smoke/coreml-equivalence.json` back into the Windows project. `train.py` refuses the formal run until this report says `PASS`.

After training, repeat the same trace/Core ML/equivalence sequence with `--checkpoint outputs/formal-v1/best.pt`. The first Core ML conversion should remain full FP32. FP16/quantization is a later measured optimization, not a training prerequisite.

## Data policy

The manifest builder uses MangaSeg's pixel-level segmentation annotations for the selected classes. All boxes are derived from segmentation-mask extents, but to keep the training manifest compact it retains full RLE only for `balloon`, the only V1 pixel-mask task. Box/mask geometry therefore stays consistent without carrying unused frame/text/SFX masks through every epoch.

The old Manga109-s test split has already been observed in the previous project. It is therefore blocked from training/tuning and is only a legacy benchmark after a model is frozen. A new independent holdout should be annotated/reserved before any final-release claim.

`PrayerHaNemurenai` is excluded by default because the previous audit found a 98-page MangaSeg/Manga109-s dimension mismatch. The script refuses to reinterpret those pages as negative examples.

## Why not simply use Ultralytics YOLO

The current Ultralytics licensing model places proprietary/commercial use under an Enterprise license unless AGPL obligations are accepted. This project avoids making that a hidden product dependency.

## Why not blindly transplant the full RTMDet COCO recipe

RTMDet is the principal reference for the optimization recipe, but generic COCO Mosaic/MixUp/horizontal-flip augmentation is not copied blindly. Manga panels and Japanese text have page-structure constraints. V1 keeps the well-supported optimization components (quality-aware classification, GIoU, EMA, warmup/cosine, two-stage augmentation) while retaining a small static, Core-ML-friendly model graph and a stride-4 P2 feature level.


## Design / operating notes

- `docs/RESEARCH_DECISIONS.md`: what was adopted from mature detector training and what was deliberately rejected.
- `docs/DEVICE_DATA_PLAN.md`: recovered device, dataset, split and deployment constraints.
- `docs/ON_DEVICE_CHECKLIST.md`: exact Windows/Mac run sequence.
- `docs/LICENSE_AND_RELEASE.md`: dataset/framework release caveats.
