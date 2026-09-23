from __future__ import annotations
from typing import Any, Sequence
import numpy as np


class RLEError(ValueError):
    pass


def decode_compressed_counts(value: str) -> list[int]:
    counts: list[int] = []
    index = 0
    while index < len(value):
        number = 0
        shift = 0
        more = True
        last = 0
        while more:
            if index >= len(value):
                raise RLEError("truncated compressed RLE")
            last = ord(value[index]) - 48
            index += 1
            if not 0 <= last <= 63:
                raise RLEError("invalid compressed RLE character")
            number |= (last & 0x1F) << shift
            more = bool(last & 0x20)
            shift += 5
            if shift > 60:
                raise RLEError("RLE integer too large")
        if last & 0x10:
            number |= -1 << shift
        counts.append(number)
    for i in range(3, len(counts)):
        counts[i] += counts[i - 2]
    return counts


def parse_rle(segmentation: Any) -> tuple[int, int, list[int]]:
    if not isinstance(segmentation, dict):
        raise RLEError("segmentation is not an RLE object")
    size = segmentation.get("size")
    if not isinstance(size, list) or len(size) != 2:
        raise RLEError("malformed RLE size")
    h, w = int(size[0]), int(size[1])
    raw = segmentation.get("counts")
    if isinstance(raw, str):
        counts = decode_compressed_counts(raw)
    elif isinstance(raw, list):
        counts = [int(x) for x in raw]
    else:
        raise RLEError("malformed RLE counts")
    if h <= 0 or w <= 0 or any(x < 0 for x in counts):
        raise RLEError("invalid RLE values")
    if sum(counts) != h * w:
        raise RLEError(f"RLE length mismatch: {sum(counts)} != {h*w}")
    return h, w, counts


def mask_extent(segmentation: Any) -> tuple[list[int], int, tuple[int, int]]:
    h, w, counts = parse_rle(segmentation)
    position = 0
    area = 0
    xmin, ymin, xmax, ymax = w, h, -1, -1
    for run_index, run_length in enumerate(counts):
        if run_index % 2 == 1 and run_length:
            start = position
            end = position + run_length - 1
            # Column-major COCO RLE. Iterate columns touched by the run; this
            # avoids allocating a full mask while preserving an exact extent.
            start_x, start_y = divmod(start, h)
            end_x, end_y = divmod(end, h)
            xmin = min(xmin, start_x)
            xmax = max(xmax, end_x)
            if start_x == end_x:
                ymin = min(ymin, start_y)
                ymax = max(ymax, end_y)
            else:
                # Any run that crosses a column boundary necessarily includes
                # the bottom of the first column and the top of the next one.
                ymin = 0
                ymax = h - 1
            area += run_length
        position += run_length
    if area <= 0:
        raise RLEError("empty mask")
    return [xmin, ymin, xmax + 1, ymax + 1], area, (w, h)


def decode_mask(segmentation: Any) -> np.ndarray:
    h, w, counts = parse_rle(segmentation)
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, n in enumerate(counts):
        if i % 2 == 1 and n:
            flat[pos:pos+n] = 1
        pos += n
    return flat.reshape((h, w), order="F")
