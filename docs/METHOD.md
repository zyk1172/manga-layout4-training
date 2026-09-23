# Training method — V1

## Target

Exactly four object classes: `frame`, `text`, `balloon`, `onomatopoeia`.
The detector predicts boxes for all four. A lightweight YOLACT-style balloon instance-mask path predicts shared high-resolution mask prototypes plus per-detection coefficients. Each detected balloon therefore receives its own mask/contour rather than sharing one page-level foreground map.

## Why this architecture

The project keeps a deployment-proven mobile shape: MobileNetV3-Small, P2/P3/P4/P5 features, a lightweight FPN, decoupled classification/regression towers, and raw tensor outputs. It deliberately retains P2 (stride 4), because small text and SFX are important. The new model does **not** load V2B5 detector weights; only ImageNet backbone initialization is used.

This is a clean new training lifecycle, not a V2B5/V3 continuation.

## Mature training ideas adopted

The recipe takes the parts of modern real-time detector training that are portable and well supported on the target hardware:

- ImageNet backbone initialization instead of random backbone initialization.
- Quality-aware focal classification: positive class quality is the detached box IoU, so classification scores are aligned with localization quality. The box head is initialized to stride-proportional object extents so this quality signal is non-trivial from the first updates.
- GIoU box regression.
- Decoupled classification/regression towers.
- AdamW + linear warmup + cosine decay.
- Exponential moving average of model weights.
- Strong-to-weak training schedule: full-page scale-down/translation/photometric augmentation in the first 80%, then plain letterbox late. V1 does not crop/zoom into the page.
- Gradient clipping and deterministic seeds.
- Book-level train/validation separation.

The recipe is inspired by RTMDet/GFL, but the graph remains ordinary PyTorch/torchvision operators to keep Core ML conversion low-risk.

## Deliberate deviations from generic COCO recipes

Mosaic, MixUp, random arbitrary crops, and horizontal flips are **off by default**. Generic RTMDet uses Mosaic/MixUp and then switches them off late, but manga page structure is not a generic natural-image task: Mosaic fabricates impossible panel boundaries, arbitrary crops were already implicated in prior project distribution-shift experiments, and horizontal flips mirror Japanese text. The safe augmentation is full page + valid half-spread views + moderate scale/translation.

If later evidence supports Mosaic/MixUp, add them as an explicit A/B experiment, not as a silent default.

## SFX treatment

MangaSeg reports onomatopoeia as a relatively scarce, thin and irregular category. V1 therefore keeps it separate from `text` and uses LVIS/Detectron2-style repeat-factor sampling derived from the **actual fraction of training pages** containing each category. Fixed SFX loss multipliers are not stacked on top of resampling. SFX AP is tracked independently so overall metrics cannot hide regressions.

## Balloon contour

MangaSeg provides **instance** masks, so V1 preserves that supervision. A stride-2 prototype head produces 8 shared 320×320 mask bases and every detector location predicts 8 coefficients. For a balloon detection, the coefficients linearly combine the prototypes and the result is cropped by that instance box. This follows the mature YOLACT decomposition (shared prototypes + per-instance coefficients), keeps the neural graph fully convolutional/static for Core ML, and avoids the instance-merging failure of a page-level semantic mask.

Validation uses balloon instance Mask AP50 and boundary-sensitive AP50 rather than page-level foreground IoU. Box AP75 and FP/page are also recorded diagnostically.

## Fail-fast gates

Formal training is not the first command. Required sequence:

1. dataset build/audit;
2. actual-model CUDA batch benchmark;
3. 16-page 300-step overfit test;
4. TorchScript raw-output trace smoke;
5. on Mac, Core ML conversion/equivalence smoke;
6. formal training.

A failure at any gate blocks the long run.
