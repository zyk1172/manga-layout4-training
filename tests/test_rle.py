import numpy as np
from manga_layout4.rle import decode_mask, mask_extent


def _encode_uncompressed(mask: np.ndarray) -> dict:
    h, w = mask.shape
    flat = mask.reshape(-1, order="F")
    counts = []
    current = 0
    run = 0
    for bit in flat.tolist():
        bit = int(bit)
        if bit == current:
            run += 1
        else:
            counts.append(run)
            run = 1
            current = bit
    counts.append(run)
    return {"size": [h, w], "counts": counts}


def test_uncompressed_rle():
    r = {"size": [3, 4], "counts": [4, 2, 6]}
    m = decode_mask(r)
    assert m.sum() == 2
    box, area, size = mask_extent(r)
    assert area == 2 and size == (4, 3)
    assert box == [1, 1, 2, 3]


def test_extent_matches_full_decode_randomized():
    rng = np.random.default_rng(20260923)
    for _ in range(200):
        h = int(rng.integers(2, 25))
        w = int(rng.integers(2, 25))
        mask = (rng.random((h, w)) < float(rng.uniform(0.03, 0.45))).astype(np.uint8)
        if not mask.any():
            mask[int(rng.integers(0, h)), int(rng.integers(0, w))] = 1
        rle = _encode_uncompressed(mask)
        decoded = decode_mask(rle)
        assert np.array_equal(decoded, mask)
        ys, xs = np.nonzero(mask)
        expected = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        box, area, size = mask_extent(rle)
        assert box == expected
        assert area == int(mask.sum())
        assert size == (w, h)
