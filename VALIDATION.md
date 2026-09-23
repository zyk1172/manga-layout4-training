# Validation status of the reviewed V1 scaffold

Re-reviewed after the training-method audit.

Validated in the current build environment:

- unit tests: 7 passed;
- randomized RLE extent vs full-mask decoding: passed (200 randomized masks inside the test suite);
- compact-manifest synthetic fixture: passed; full RLE is retained only for balloon instances;
- model forward/backward: 640x640, batch 2, four detection classes + 8 x 320x320 mask prototypes + per-location mask coefficients; finite multi-task loss and successful backward;
- instance-mask decoding smoke: passed;
- TorchScript raw graph: 13 outputs; trace smoke passed;
- Python bytecode compile: passed;
- random-weight evaluation pipeline: box AP50/AP75 + balloon instance Mask AP50 + boundary AP50 path executes without error.

The review changed the method in four material ways before formal training:

1. replaced the page-level balloon union mask with YOLACT-style instance prototypes + per-detection coefficients so instance identity is preserved;
2. replaced fixed SFX double weighting with frequency-derived LVIS/Detectron2-style repeat-factor sampling;
3. disabled crop/zoom-in augmentation in the formal V1 baseline so complete page geometry is preserved;
4. strengthened the overfit/selection metrics so a falling total loss cannot hide a dead detection or instance-mask head.

Not executable in the current environment because the user's local devices/datasets are not mounted here:

- full local MangaSeg/Manga109-s manifest build and exact page/class frequency audit;
- GTX 1660 SUPER real-model VRAM/throughput preflight;
- real 16-page memorization/overfit gate;
- Mac coremltools conversion and Core ML numerical equivalence for the 13-output graph;
- physical iPhone latency/memory validation;
- formal long training.

Formal training remains blocked until the real-data audit, CUDA preflight, memorization gate and pre-training Core ML gate all report PASS.
