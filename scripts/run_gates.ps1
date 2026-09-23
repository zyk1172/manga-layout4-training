param(
  [string]$Config = "configs/base.yaml",
  [string]$SourceRoot = "F:\codex\manga-model",
  [string]$Device = "cuda"
)
$ErrorActionPreference = "Stop"
$env:MANGA_VISION_SOURCE_ROOT = $SourceRoot
python scripts/cache_backbone.py
python scripts/build_manifest.py --config $Config
python scripts/audit_dataset.py --config $Config
python scripts/preflight.py --config $Config --device $Device
python scripts/overfit_gate.py --config $Config --device $Device
Write-Host "All Windows training gates passed. Formal training is now allowed."
