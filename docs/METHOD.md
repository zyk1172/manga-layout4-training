# Training method — V1

## Target

Exactly four object classes: `frame`, `text`, `balloon`, `onomatopoeia`.
The detector predicts boxes for all four. A dedicated high-resolution binary balloon-mask head predicts balloon foreground; per-balloon contours are recovered by intersecting the dense mask with detected balloon boxes.

## Why this architecture

The project keeps a deployment-proven mobile shape: MobileNetV3-Small, P2/P3/P4/P5 features, a lightweight FPN, decoupled classification/regression towers, and raw tensor outputs. It deliberately retains P2 (stride 4), because small text and SFX are important. The new model does **not** load V2B5 detector weights; only ImageNet backbone initialization is used.

This is a clean new training lifecycle, not a V2B5/V3 continuation.

## Mature training ideas adopted

The recipe takes the parts of modern real-time detector training that are portable and well supported on the target hardware:

- ImageNet backbone initialization instead of random backbone initialization.
- Quality-aware focal classification: positive class quality is the detached box IoU, so classification scores are aligned with localization quality.
- GIoU box regression.
- Decoupled classification/regression towers.
- AdamW + linear warmup + cosine decay.
- Exponential moving average of model weights.
- Strong-to-weak training schedule: mild scale/translation/photometric augmentation in the first 80%, then weak augmentation in the last 20%.
- Gradient clipping and deterministic seeds.
- Book-level train/validation separation.

The recipe is inspired by RTMDet/GFL, but the graph remains ordinary PyTorch/torchvision operators to keep Core ML conversion low-risk.

## Deliberate deviations from generic COCO recipes

Mosaic, MixUp, random arbitrary crops, and horizontal flips are **off by default**. Generic RTMDet uses Mosaic/MixUp and then switches them off late, but manga page structure is not a generic natural-image task: Mosaic fabricates impossible panel boundaries, arbitrary crops were already implicated in prior project distribution-shift experiments, and horizontal flips mirror Japanese text. The safe augmentation is full page + valid half-spread views + moderate scale/translation.

If later evidence supports Mosaic/MixUp, add them as an explicit A/B experiment, not as a silent default.

## SFX treatment

MangaSeg reports onomatopoeia as a relatively scarce, thin and irregular category. V1 therefore:

- keeps it separate from `text`;
- weights pages containing SFX higher in the sampler;
- gives SFX positives a modestly higher classification weight;
- tracks SFX AP independently so overall metrics cannot hide regressions.

## Balloon contour

MangaSeg provides pixel-level instance masks. V1 trains a dense balloon foreground head at stride 2 (320×320 for a 640 input). At inference, each balloon detection box gates the balloon foreground mask; connected components and box overlap select the contour belonging to that detection. This avoids a dynamic instance-mask kernel in the Core ML graph while retaining a precise shape output.

## Fail-fast gates

Formal training is not the first command. Required sequence:

1. dataset build/audit;
2. actual-model CUDA batch benchmark;
3. 16-page 300-step overfit test;
4. TorchScript raw-output trace smoke;
5. on Mac, Core ML conversion/equivalence smoke;
6. formal training.

A failure at any gate blocks the long run.
