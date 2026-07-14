from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image


def hdf5_value_to_bytes(value: Any) -> bytes:
    """Convert common HDF5 JPEG storage values into raw bytes."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, np.bytes_):
        return bytes(value)
    if isinstance(value, np.void):
        return bytes(value)
    if isinstance(value, np.ndarray):
        if value.dtype == np.uint8:
            return value.tobytes()
        if value.shape == ():
            return hdf5_value_to_bytes(value.item())
    if isinstance(value, str):
        return value.encode("latin1")
    raise TypeError(f"Unsupported JPEG HDF5 value type: {type(value)!r}")


def decode_jpeg_grayscale(jpeg_bytes: bytes) -> np.ndarray:
    with Image.open(BytesIO(jpeg_bytes)) as image:
        return np.asarray(image.convert("L"))


def decode_jpeg_rgb(jpeg_bytes: bytes) -> np.ndarray:
    with Image.open(BytesIO(jpeg_bytes)) as image:
        return np.asarray(image.convert("RGB"))


def encode_jpeg_grayscale(frame: np.ndarray, quality: int) -> bytes:
    if frame.ndim != 2:
        raise ValueError(f"Expected a grayscale frame, got shape {frame.shape}")
    clipped = np.clip(frame, 0, 255).astype(np.uint8, copy=False)
    buffer = BytesIO()
    Image.fromarray(clipped, mode="L").save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=False,
    )
    return buffer.getvalue()
