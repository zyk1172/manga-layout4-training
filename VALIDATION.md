# Validation status of the delivered V1 scaffold

Validated in the current build environment:

- unit tests: 6 passed;
- randomized RLE extent vs full-mask decoding: passed (200 randomized masks inside the test suite);
- compact-manifest synthetic fixture: passed; only balloon retains RLE;
- model forward/backward: 640x640, batch 2, four detection classes + 320x320 balloon mask; finite loss and successful backward;
- TorchScript raw graph: 9 outputs, trace/frozen-trace equivalence passed with maximum observed absolute difference below 1e-6 in the last smoke run;
- Python bytecode compile: passed.

Not executable in the current environment because the user's local devices/datasets are not mounted here:

- full local MangaSeg/Manga109-s manifest build and exact page/class audit;
- GTX 1660 SUPER real-model VRAM/throughput preflight;
- 16-page on-device overfit gate using the real dataset;
- Mac coremltools conversion and Core ML numerical equivalence;
- physical iPhone latency/memory validation;
- formal long training.

The project intentionally blocks formal training until the first four relevant pre-training gates are recorded as PASS. The post-training iPhone gate remains a promotion/release gate.
