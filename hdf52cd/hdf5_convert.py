from __future__ import annotations

import glob as globlib
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import h5py
import numpy as np

from .cd_backend import CDBackendConfig, CDFrameBackend, NightriderGPUCDBackend
from .jpeg import (
    decode_jpeg_grayscale,
    decode_jpeg_rgb,
    encode_jpeg_grayscale,
    hdf5_value_to_bytes,
)


@dataclass(frozen=True)
class ConversionConfig:
    frames_key: str | None = None
    device: str = "cpu"
    jpeg_quality: int = 95
    nightrider_path: str | None = None
    nightrider_ref: str = "origin/gpu_pipeline"
    th_pos: float = 0.4
    th_neg: float = 0.4
    ref_us: int = 100
    dt_us: int = 16667
    overwrite: bool = False


@dataclass(frozen=True)
class ConversionResult:
    input_path: str
    output_path: str
    frames_key: str
    frame_count: int
    width: int
    height: int
    backend: str
    device: str
    status: str
    error: str = ""


BackendFactory = Callable[[int, int, ConversionConfig], CDFrameBackend]
ProgressCallback = Callable[[int, int], None]


def discover_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            for glob in ("*.h5", "*.hdf5"):
                paths.extend(path.rglob(glob))
        else:
            if any(c in pattern for c in "*?["):
                paths.extend(Path(match) for match in sorted(globlib.glob(pattern)))
            else:
                paths.append(path)
    deduped = sorted({p.resolve() for p in paths})
    return [p for p in deduped if p.exists() and p.is_file()]


def convert_file(
    input_path: Path,
    output_dir: Path,
    config: ConversionConfig,
    backend_factory: BackendFactory | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ConversionResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_cd{input_path.suffix}"
    tmp_output_path = output_dir / f".{output_path.name}.tmp"
    if output_path.exists():
        if not config.overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")
    if tmp_output_path.exists():
        if not config.overwrite:
            raise FileExistsError(f"Temporary output already exists: {tmp_output_path}")
        tmp_output_path.unlink()

    try:
        with h5py.File(input_path, "r") as src:
            frames_key = resolve_frames_key(src, config.frames_key)
            frame_count = int(src[frames_key].shape[0])
            if frame_count == 0:
                raise ValueError("Frame dataset is empty")
            first_frame = decode_jpeg_rgb(hdf5_value_to_bytes(src[frames_key][0]))
            height, width = first_frame.shape[:2]
        if progress_callback is not None:
            progress_callback(0, frame_count)

        backend = create_cd_backend(width, height, config, backend_factory)

        shutil.copy2(input_path, tmp_output_path)
        with h5py.File(tmp_output_path, "r+") as dst:
            frames_key = resolve_frames_key(dst, config.frames_key)
            replace_dataset_with_cd_jpegs(
                dst,
                frames_key,
                backend,
                config.jpeg_quality,
                progress_callback=progress_callback,
            )
            write_conversion_metadata(dst, frames_key, config, backend, input_path)
            validate_output(dst, frames_key, frame_count)

        tmp_output_path.replace(output_path)
        return ConversionResult(
            input_path=str(input_path),
            output_path=str(output_path),
            frames_key=frames_key,
            frame_count=frame_count,
            width=width,
            height=height,
            backend=backend.name,
            device=backend.actual_device,
            status="success",
        )
    except BaseException:
        if tmp_output_path.exists():
            tmp_output_path.unlink()
        raise


def create_cd_backend(
    width: int,
    height: int,
    config: ConversionConfig,
    backend_factory: BackendFactory | None = None,
) -> CDFrameBackend:
    if backend_factory is not None:
        return backend_factory(width, height, config)
    backend_config = CDBackendConfig(
        device=config.device,
        nightrider_path=config.nightrider_path,
        nightrider_ref=config.nightrider_ref,
        th_pos=config.th_pos,
        th_neg=config.th_neg,
        ref_us=config.ref_us,
        dt_us=config.dt_us,
    )
    return NightriderGPUCDBackend(width, height, backend_config)


def resolve_frames_key(h5: h5py.File, explicit_key: str | None) -> str:
    if explicit_key:
        key = explicit_key.strip("/")
        if key not in h5:
            raise KeyError(f"Frame dataset not found: {explicit_key}")
        if not isinstance(h5[key], h5py.Dataset):
            raise TypeError(f"Frame key is not a dataset: {explicit_key}")
        return key

    candidates: list[str] = []

    def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if not isinstance(obj, h5py.Dataset) or not obj.shape or obj.shape[0] < 1:
            return
        lowered = name.lower()
        name_hint = any(token in lowered for token in ("jpeg", "jpg", "image", "frame", "video"))
        if not name_hint:
            return
        try:
            decode_jpeg_rgb(hdf5_value_to_bytes(obj[0]))
        except Exception:
            return
        candidates.append(name)

    h5.visititems(visit)
    if not candidates:
        raise KeyError("Could not autodetect a JPEG frame dataset; pass --frames-key")
    if len(candidates) > 1:
        joined = ", ".join(candidates)
        raise KeyError(f"Multiple candidate frame datasets found; pass --frames-key: {joined}")
    return candidates[0]


def list_frame_candidates(input_path: Path) -> list[str]:
    with h5py.File(input_path, "r") as h5:
        candidates: list[str] = []

        def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if not isinstance(obj, h5py.Dataset) or not obj.shape or obj.shape[0] < 1:
                return
            try:
                decode_jpeg_rgb(hdf5_value_to_bytes(obj[0]))
            except Exception:
                return
            candidates.append(name)

        h5.visititems(visit)
        return candidates


def replace_dataset_with_cd_jpegs(
    h5: h5py.File,
    dataset_key: str,
    backend: CDFrameBackend,
    jpeg_quality: int,
    progress_callback: ProgressCallback | None = None,
) -> None:
    old = h5[dataset_key]
    parent_path, name = split_hdf5_path(dataset_key)
    parent = h5[parent_path] if parent_path else h5
    tmp_name = unique_replacement_dataset_name(parent, name)
    new = create_replacement_dataset(parent, tmp_name, old)

    try:
        write_cd_jpegs(
            old,
            new,
            backend,
            jpeg_quality,
            progress_callback=progress_callback,
        )
        del parent[name]
        parent.move(tmp_name, name)
    except BaseException:
        if tmp_name in parent:
            del parent[tmp_name]
        raise


def create_replacement_dataset(parent: h5py.Group, name: str, old: h5py.Dataset) -> h5py.Dataset:
    frame_count = int(old.shape[0])
    attrs = dict(old.attrs.items())
    original_dtype = str(old.dtype)

    kwargs: dict[str, Any] = {"dtype": h5py.vlen_dtype(np.dtype("uint8"))}
    if old.compression:
        kwargs["compression"] = old.compression
        kwargs["compression_opts"] = old.compression_opts
    if old.chunks:
        kwargs["chunks"] = old.chunks
    if old.maxshape and len(old.maxshape) == 1:
        kwargs["maxshape"] = old.maxshape

    new = parent.create_dataset(name, shape=(frame_count,), **kwargs)
    for key, value in attrs.items():
        new.attrs[key] = value
    new.attrs["cd_replaced_original_storage"] = original_dtype
    return new


def unique_replacement_dataset_name(parent: h5py.Group, original_name: str) -> str:
    base = f"__hrdf2cd_replacement_{original_name}"
    if base not in parent:
        return base
    for index in range(1, 1000):
        candidate = f"{base}_{index}"
        if candidate not in parent:
            return candidate
    raise RuntimeError(f"Could not choose a temporary replacement dataset name for {original_name!r}")


def write_cd_jpegs(
    frame_dataset: h5py.Dataset,
    output_dataset: h5py.Dataset,
    backend: CDFrameBackend,
    jpeg_quality: int,
    progress_callback: ProgressCallback | None = None,
) -> None:
    frame_count = int(frame_dataset.shape[0])
    if frame_count == 0:
        raise ValueError("Frame dataset is empty")
    if int(output_dataset.shape[0]) != frame_count:
        raise ValueError(f"Output frame count mismatch: expected {frame_count}, got {output_dataset.shape[0]}")

    first_rgb = decode_jpeg_rgb(hdf5_value_to_bytes(frame_dataset[0]))
    if frame_count == 1:
        blank = np.zeros(first_rgb.shape[:2], dtype=np.uint8)
        write_jpeg_frame(output_dataset, 0, encode_jpeg_grayscale(blank, jpeg_quality))
        if progress_callback is not None:
            progress_callback(1, frame_count)
        return

    # First call initializes NIGHTRIDER's internal DVS state and returns empty events.
    backend.process_frame(first_rgb)
    if progress_callback is not None:
        progress_callback(1, frame_count)

    first_cd: bytes | None = None
    for index in range(1, frame_count):
        current_rgb = decode_jpeg_rgb(hdf5_value_to_bytes(frame_dataset[index]))
        cd = backend.process_frame(current_rgb)
        cd_bytes = encode_jpeg_grayscale(cd, jpeg_quality)
        write_jpeg_frame(output_dataset, index, cd_bytes)
        if first_cd is None:
            first_cd = cd_bytes
            write_jpeg_frame(output_dataset, 0, cd_bytes)
        if progress_callback is not None:
            progress_callback(index + 1, frame_count)

    assert first_cd is not None


def write_jpeg_frame(dataset: h5py.Dataset, index: int, jpeg_bytes: bytes) -> None:
    dataset[index] = np.frombuffer(jpeg_bytes, dtype=np.uint8)


def split_hdf5_path(dataset_key: str) -> tuple[str, str]:
    stripped = dataset_key.strip("/")
    if "/" not in stripped:
        return "", stripped
    parent, name = stripped.rsplit("/", 1)
    return parent, name


def write_conversion_metadata(
    h5: h5py.File,
    frames_key: str,
    config: ConversionConfig,
    backend: CDFrameBackend,
    input_path: Path,
) -> None:
    config_json = json.dumps(asdict(config), sort_keys=True)
    h5.attrs["cd_conversion_version"] = "0.2.0"
    h5.attrs["cd_source_file"] = str(input_path)
    h5.attrs["cd_source_frames_key"] = frames_key
    h5.attrs["cd_index_policy"] = "same_length_repeat_first"
    h5.attrs["cd_encoding"] = "jpeg_grayscale"
    h5.attrs["cd_event_rendering"] = "nightrider_binary_gray_background_off0_on255"
    h5.attrs["cd_backend"] = backend.name
    h5.attrs["cd_device"] = backend.actual_device
    h5.attrs["cd_simulator_source"] = "NIGHTRIDER gpu_pipeline IECBSLiveGPU.process_batch"
    h5.attrs["cd_nightrider_ref"] = config.nightrider_ref
    h5.attrs["cd_sim_config_json"] = config_json
    h5.attrs["cd_sim_config_hash"] = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    h5.attrs["cd_created_at"] = datetime.now(timezone.utc).isoformat()


def validate_output(h5: h5py.File, frames_key: str, expected_count: int) -> None:
    dataset = h5[frames_key]
    if int(dataset.shape[0]) != expected_count:
        raise ValueError(f"Frame count changed: expected {expected_count}, got {dataset.shape[0]}")
    for index in range(expected_count):
        decode_jpeg_grayscale(hdf5_value_to_bytes(dataset[index]))


def write_manifest(path: Path, results: Iterable[ConversionResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
