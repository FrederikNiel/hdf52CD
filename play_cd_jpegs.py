#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


DEFAULT_HDF5 = Path("/home/frederik/Documents/rover_rgbd_hd720_expert_400k_10pct_cd.hdf5")
DEFAULT_FRAMES_KEY = "/observations/rgb_jpeg"


def hdf5_value_to_bytes(value: Any) -> bytes:
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
    raise TypeError(f"Unsupported HDF5 frame value type: {type(value)!r}")


def decode_jpeg(value: Any) -> np.ndarray:
    encoded = np.frombuffer(hdf5_value_to_bytes(value), dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise ValueError("OpenCV could not decode JPEG frame")
    return frame


def as_bgr24(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    raise ValueError(f"Unsupported decoded frame shape: {frame.shape}")


def opencv_has_gui() -> bool:
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return "NONE" not in line
    return False


def print_file_status(h5: h5py.File, path: Path, frames_key: str, frame_count: int) -> None:
    print(f"file: {path}")
    print(f"frames: {frames_key} ({frame_count})")
    if "cd_conversion_version" not in h5.attrs:
        print("warning: no cd_conversion_version attribute found; this file may still contain raw RGB JPEGs")
        return

    version = h5.attrs.get("cd_conversion_version", "?")
    encoding = h5.attrs.get("cd_encoding", "?")
    backend = h5.attrs.get("cd_backend", "?")
    device = h5.attrs.get("cd_device", "?")
    print(f"cd: version={version} encoding={encoding} backend={backend} device={device}")


def iter_frames(dataset: h5py.Dataset, start: int, stop: int):
    for index in range(start, stop):
        yield index, decode_jpeg(dataset[index])


def play_with_opencv(dataset: h5py.Dataset, start: int, stop: int, fps: float, window_name: str) -> None:
    delay_ms = max(1, round(1000 / fps))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        for index, frame in iter_frames(dataset, start, stop):
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord("q")):
                break
            if index == stop - 1:
                cv2.waitKey(1)
    finally:
        cv2.destroyAllWindows()


def play_with_ffplay(dataset: h5py.Dataset, start: int, stop: int, fps: float) -> None:
    first = as_bgr24(decode_jpeg(dataset[start]))
    height, width = first.shape[:2]
    ffplay = shutil.which("ffplay")
    if not ffplay:
        raise RuntimeError("ffplay was not found; install ffmpeg or run with --viewer opencv on a GUI OpenCV build")

    cmd = [
        ffplay,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-autoexit",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        process.stdin.write(first.tobytes())
        for _, frame in iter_frames(dataset, start + 1, stop):
            process.stdin.write(as_bgr24(frame).tobytes())
    except BrokenPipeError:
        pass
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        process.wait()


def resolve_viewer(requested: str) -> str:
    if requested != "auto":
        return requested
    if opencv_has_gui():
        return "opencv"
    if shutil.which("ffplay"):
        return "ffplay"
    return "opencv"


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play JPEG frames stored in a CD HDF5 file.")
    parser.add_argument("hdf5", nargs="?", default=str(DEFAULT_HDF5), help=f"HDF5 file to play. Default: {DEFAULT_HDF5}")
    parser.add_argument("--frames-key", default=DEFAULT_FRAMES_KEY, help=f"HDF5 JPEG dataset. Default: {DEFAULT_FRAMES_KEY}")
    parser.add_argument("--fps", type=positive_float, default=30.0, help="Playback frames per second. Default: 30")
    parser.add_argument("--start", type=nonnegative_int, default=0, help="Frame index to start from. Default: 0")
    parser.add_argument("--limit", type=nonnegative_int, help="Maximum number of frames to play.")
    parser.add_argument("--viewer", choices=("auto", "ffplay", "opencv"), default="auto", help="Playback backend. Default: auto")
    parser.add_argument("--check-only", action="store_true", help="Decode one frame and print status without opening a player.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hdf5_path = Path(args.hdf5)
    if not hdf5_path.exists():
        print(f"error: file not found: {hdf5_path}", file=sys.stderr)
        return 2

    with h5py.File(hdf5_path, "r") as h5:
        if args.frames_key not in h5:
            print(f"error: frames key not found: {args.frames_key}", file=sys.stderr)
            return 2
        dataset = h5[args.frames_key]
        if not isinstance(dataset, h5py.Dataset) or len(dataset.shape) != 1:
            print(f"error: frames key is not a 1D dataset: {args.frames_key}", file=sys.stderr)
            return 2

        frame_count = int(dataset.shape[0])
        if frame_count == 0:
            print("error: frame dataset is empty", file=sys.stderr)
            return 2
        if args.start >= frame_count:
            print(f"error: --start {args.start} is outside frame count {frame_count}", file=sys.stderr)
            return 2

        stop = frame_count if args.limit is None else min(frame_count, args.start + args.limit)
        print_file_status(h5, hdf5_path, args.frames_key, frame_count)

        sample = decode_jpeg(dataset[args.start])
        print(f"decoded frame {args.start}: shape={sample.shape} dtype={sample.dtype}")
        if args.check_only:
            return 0

        viewer = resolve_viewer(args.viewer)
        print(f"playing frames {args.start}:{stop} at {args.fps:g} fps with {viewer}; press q/Esc in OpenCV or close ffplay to stop")
        if viewer == "opencv":
            if not opencv_has_gui():
                print("warning: OpenCV was built without GUI support; playback may fail. Try --viewer ffplay.", file=sys.stderr)
            play_with_opencv(dataset, args.start, stop, args.fps, "CD JPEG Playback")
        else:
            play_with_ffplay(dataset, args.start, stop, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
