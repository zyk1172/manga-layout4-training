# Known device/data conditions

## Windows formal training host

Known previous environment:

- old data/training root: `F:\codex\manga-model`
- Python 3.11.14
- PyTorch 2.7.0 + cu126
- torchvision 0.22.0 + cu126
- CUDA runtime 12.6
- NVIDIA GeForce GTX 1660 SUPER
- 640x640 FP32 V2A benchmark: batch 4 ~10.17 images/s, ~2.89 GiB reserved; batch 8 ~4.63 GiB reserved
- V2B5 observed training: ~8.81 images/s, ~3.64 GiB reserved

V1 does not assume batch 4 is still safe. `preflight.py` tries 4 -> 2 -> 1 on the real new graph and accepts the first choice under the configured VRAM ceiling, then raises gradient accumulation to preserve effective batch size 8.

## Mac export host

Known prior path: `/Users/zhengyunkai/Documents/开发项目/正式项目/工具/manga-vision-training`.
Known previous conversion stack: Apple Silicon, Python 3.11.14, PyTorch 2.7.0, torchvision 0.22.0, coremltools 9.0.

Use a fresh Mac venv. Do not copy the Windows virtual environment.

## Physical deployment target

Previously validated target: iPhone 16 Pro (`iPhone17,1`), arm64e, iOS 27.0 build 24A437.
The old V2B5 provider measured about 25.5 ms mean Core ML model time and ~77.6 ms mean provider total over 40 frozen validation pages. Those values are a deployment reference, not a performance promise for MangaLayout4V1 because the new balloon mask output adds compute.

## Local data reused read-only

The project expects the old data tree to contain:

- `data/raw/manga109s/images`
- `data/external/MangaSegmentation/jsons`
- `data/splits/train_books.txt`, `val_books.txt`, `test_books.txt`

Set `MANGA_VISION_SOURCE_ROOT` instead of copying data.

MangaSeg defines six categories. V1 reads only `frame`, `text`, `balloon`, and `onomatopoeia`; `face` and `body` never become targets or background labels at the annotation-object level.

The manifest stores bbox metadata for frame/text/SFX and retains pixel RLE only for balloon, which materially reduces training-memory overhead while preserving the required contour target.

## Known exclusions and evaluation status

`PrayerHaNemurenai` is excluded by default because the old audit found 98 pages whose MangaSeg dimensions do not match the local canonical Manga109-s images. V1 never treats these unresolved pages as clean negatives.

The old Manga109-s test books were already observed in the previous project. They remain blocked from training/tuning and can only be used as a legacy benchmark after a candidate is frozen. A new independent reference set is still required for a final release claim.
