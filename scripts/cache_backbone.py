#!/usr/bin/env python3
"""Cache the generic ImageNet backbone before formal/offline training."""
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def main() -> None:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    params = sum(p.numel() for p in model.parameters())
    print(f"PASS: cached {weights} ({params:,} parameters)")


if __name__ == "__main__":
    main()
