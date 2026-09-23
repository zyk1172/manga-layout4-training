# Research decisions for V1

## Design goal

The first formal version should be boring to operate: high-quality ground truth, a small static graph, explicit gates, and a short list of mature optimization choices. It should not require a new teacher loop, pseudo-label adjudication, custom CUDA operators, or a second deployment architecture.

## External references reviewed

- RTMDet, *An Empirical Study of Designing Real-Time Object Detectors* (arXiv:2212.07784). Reference ideas: decoupled real-time head design, soft/quality-aware supervision, strong detector training, extensibility to instance segmentation.
- GFL, *Generalized Focal Loss* (arXiv:2006.04388). Reference idea: make classification confidence reflect localization quality.
- MangaSeg, CVPR 2025, *Advancing Manga Analysis: Comprehensive Segmentation Annotations for the Manga109 Dataset*. Reference facts: six manga categories with pixel-level masks and instance information; onomatopoeia is a distinct thin/irregular category.
- Apple Core ML Tools PyTorch conversion guide. Reference constraint: keep a traceable, static raw-output PyTorch graph and convert directly to an ML Program.

These are references for methodology, not copied frameworks.

## What V1 adopts

1. ImageNet initialization for the MobileNetV3-Small backbone. This is not continuation from V2B5/V3; detector heads, FPN/detail fusion and mask head train as a new model.
2. P2/P3/P4/P5 multi-scale features, because the previous project demonstrated that P2 materially helps small manga elements.
3. Decoupled classification and box towers.
4. Quality Focal-style targets: a positive class target is the detached IoU of the current predicted box and its assigned GT. This makes the score useful for ranking/NMS.
5. GIoU box loss.
6. YOLACT-style shared mask prototypes + per-instance coefficients, supervised with per-instance BCE + Dice at stride 2.
7. AdamW, linear warmup, cosine decay, EMA, gradient clipping, deterministic seed.
8. Strong-to-weak schedule: mild scale/translation/photometric augmentation first, then plain letterbox late.
9. LVIS/Detectron2-style repeat-factor sampling computed from actual image frequency; no simultaneous hand-tuned SFX loss multiplier.
10. Full-page views only in the formal V1 baseline; half-spread remains an explicit later ablation.

## What V1 deliberately does not adopt

### Full MMDetection/RTMDet-Ins runtime

It is a strong research/training stack, but the deployment target is a small Core ML graph on iPhone and the training host is a GTX 1660 SUPER Windows machine. V1 avoids mmcv/custom CUDA/runtime dependencies and keeps the deployment model in ordinary torch/torchvision operators.


### RT-DETRv2 / DINO / Mask DINO as the V1 deployment graph

They are strong modern detector/segmentation families and were reviewed. RT-DETRv2 explicitly improves deployment practicality, and DINO/Mask DINO are strong end-to-end detection/segmentation references. They were not chosen for this first mobile graph because transformer/deformable-attention/query-mask machinery would materially increase implementation, VRAM and Core ML conversion risk on the known GTX 1660 SUPER -> iPhone path. Their ideas remain candidates for a future accuracy ceiling study on stronger training hardware, not the first production-oriented baseline.

### Re-implementing DynamicSoftLabelAssigner in V1

RTMDet's dynamic soft-label assignment is valuable, but a fresh custom implementation would become a new correctness risk. The previous detector's FCOS center/size assignment is already exercised by the old project. V1 changes the data truth, class set, initialization, score loss and mask task while holding assignment geometry stable. A dynamic assigner is reserved as a controlled V2 ablation only if the clean four-class baseline exposes an assignment bottleneck.

### Mosaic/MixUp by default

Generic real-time detector recipes often use them early and switch them off late. Manga pages are structured documents: Mosaic creates impossible panel adjacency, MixUp overlays incompatible text/panel geometry, and the previous project already found crop-distribution questions difficult to isolate. V1 therefore uses conservative augmentations. Mosaic/MixUp may be tested later as an A/B experiment, never silently enabled.

### Horizontal flip

Disabled because it mirrors text and reading-direction cues.

### Pseudo labels / teacher consensus

None in V1. The prior project showed that noisy text/balloon pseudo labels and crop supervision could improve training loss while blind false positives worsened. V1 starts from MangaSeg ground truth only.

### AMP on the GTX 1660 SUPER

The previous host benchmark found FP32 faster for the established mobile detector path. V1 defaults to FP32 and re-runs the actual new-model memory/throughput preflight before formal training.

## Assignment policy

The assignment remains FCOS-style center sampling plus FPN size ranges. Tiny objects fall back from center sampling to valid in-range interior locations if they would otherwise receive no positive location. This is intentionally conservative and testable.

## Balloon contour policy

MangaSeg balloon instance identity is preserved. The network predicts 8 shared 320×320 prototypes plus 8 coefficients at every detector location. The coefficient vector of each surviving balloon detection composes an instance mask, which is cropped to that detection box. This is a YOLACT-style fully convolutional decomposition and keeps instance separation outside dynamic ROI/control-flow operators in Core ML.

## Promotion rule

Do not select a checkpoint by validation loss alone. Candidate selection is an equal-weight mean across four box AP50 values plus balloon instance Mask AP50 and boundary AP50; AP75, recall, FP/page and duplicate FP remain diagnostics. Final product claims require a fresh independent holdout, because the old Manga109-s test split has already been observed by the previous project.
