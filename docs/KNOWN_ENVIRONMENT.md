# Known environment carried from the previous project

## Windows training host

- Repository location used previously: `F:\codex\manga-model`
- Python: 3.11.14
- PyTorch: 2.7.0 + cu126
- torchvision: 0.22.0 + cu126
- CUDA runtime: 12.6
- GPU: NVIDIA GeForce GTX 1660 SUPER
- Proven V2B5 batch: 4 at 640×640 FP32
- Previous V2B5 observed throughput: ~8.81 images/s
- Previous V2B5 reserved VRAM: ~3.64 GiB
- Previous V2A batch-8 reserved VRAM: ~4.63 GiB; batch 4 was the conservative recommendation.

V1 therefore defaults to micro-batch 4 / FP32, but `scripts/preflight.py` must confirm this using the **actual new model**. If reserved memory exceeds 80% of VRAM, use batch 2 and raise gradient accumulation to preserve effective batch.

## Mac handoff host

Previous project path: `/Users/zhengyunkai/Documents/开发项目/正式项目/工具/manga-vision-training`.
Known stack: Python 3.11.14, PyTorch 2.7.0, torchvision 0.22.0, coremltools 9.0, Apple Silicon.
Use Mac for Core ML conversion and equivalence checks; do not copy a Windows virtual environment to Mac.

## Physical target

Previously verified target: iPhone 16 Pro (`iPhone17,1`), arm64e, iOS 27.0 build 24A437.
V1 keeps a static 640×640 graph and raw outputs so decoding/NMS/contour post-processing remains outside the neural network graph.
