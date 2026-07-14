from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class CDBackendConfig:
    device: str = "cpu"
    nightrider_path: str | None = None
    nightrider_ref: str = "origin/gpu_pipeline"
    th_pos: float = 0.4
    th_neg: float = 0.4
    ref_us: int = 100
    dt_us: int = 16667

    def normalized_device(self) -> str:
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("--device must be one of: auto, cuda, cpu")
        return self.device

    def resolved_nightrider_path(self) -> Path:
        configured = self.nightrider_path or os.environ.get("NIGHTRIDER_PATH")
        return Path(configured or "/home/labpcadm/Desktop/NIGHTRIDER").expanduser()


class CDFrameBackend(Protocol):
    name: str
    actual_device: str

    def process_frame(self, rgb_uint8: np.ndarray) -> np.ndarray:
        ...


class _NightriderSource:
    def __init__(self, repo_path: Path, ref: str):
        self.repo_path = repo_path
        self.ref = ref
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None

    def event_camera_dir(self) -> Path:
        direct = self.repo_path / "EventCamera"
        if self._has_gpu_backend(direct):
            return direct

        if not (self.repo_path / ".git").exists():
            raise FileNotFoundError(
                f"NIGHTRIDER GPU files were not found in {direct}, and {self.repo_path} is not a Git checkout"
            )

        self._tmpdir = tempfile.TemporaryDirectory(prefix="hdf52cd-nightrider-")
        archive = subprocess.run(
            ["git", "-C", str(self.repo_path), "archive", self.ref, "EventCamera"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        target_root = Path(self._tmpdir.name).resolve()
        with tarfile.open(fileobj=BytesIO(archive.stdout)) as tar:
            for member in tar.getmembers():
                member_target = (target_root / member.name).resolve()
                if not str(member_target).startswith(str(target_root)):
                    raise RuntimeError(f"Unsafe path in NIGHTRIDER archive: {member.name}")
            tar.extractall(target_root)

        extracted = target_root / "EventCamera"
        if not self._has_gpu_backend(extracted):
            raise FileNotFoundError(
                f"NIGHTRIDER ref {self.ref!r} does not contain the expected GPU backend files"
            )
        return extracted

    @staticmethod
    def _has_gpu_backend(event_camera_dir: Path) -> bool:
        return (event_camera_dir / "IECBSLiveGPU.py").exists() and (
            event_camera_dir / "src" / "dvs_sensor_gpu.py"
        ).exists()
    

class NightriderGPUCDBackend:
    """CD-frame backend backed by NIGHTRIDER gpu_pipeline.

    The simulator math comes from NIGHTRIDER's `IECBSLiveGPU.process_batch`
    and `DvsSensorGPU.step`. This class only adapts HDF5 JPEG frames into the
    tensor shape that NIGHTRIDER expects and renders the returned EventBuffer
    with the same binary visualization convention used by
    `EventCameraInterfaceGPU`: gray background, OFF=0, ON=255.
    """

    name = "nightrider_gpu_pipeline"

    def __init__(self, width: int, height: int, config: CDBackendConfig):
        self.width = width
        self.height = height
        self.config = config
        self._source = _NightriderSource(
            config.resolved_nightrider_path(),
            config.nightrider_ref,
        )
        self._torch = self._import_torch()
        self.actual_device = self._resolve_device(config.normalized_device())
        event_camera_dir = self._source.event_camera_dir()
        live_class = self._load_live_class(event_camera_dir)
        dvs_params = {
            "th_pos": config.th_pos,
            "th_neg": config.th_neg,
            "ref_us": config.ref_us,
            "dt_us": config.dt_us,
        }
        self._simulator = live_class(
            width=width,
            height=height,
            dvs_params=dvs_params,
            device=self.actual_device,
        )

    def process_frame(self, rgb_uint8: np.ndarray) -> np.ndarray:
        if rgb_uint8.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Expected RGB frame shape {(self.height, self.width, 3)}, got {rgb_uint8.shape}"
            )
        if not rgb_uint8.flags.c_contiguous or not rgb_uint8.flags.writeable:
            rgb_uint8 = np.array(rgb_uint8, copy=True, order="C")
        rgb = self._torch.as_tensor(
            rgb_uint8,
            dtype=self._torch.float32,
            device=self.actual_device,
        ).div(255.0)
        events_list = self._simulator.process_batch(rgb.unsqueeze(0), dt_us=self.config.dt_us)
        if not events_list:
            return np.full((self.height, self.width), 125, dtype=np.uint8)
        return render_event_buffer(events_list[0], self.width, self.height)

    @staticmethod
    def _import_torch():
        try:
            return importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError(
                "NIGHTRIDER gpu_pipeline requires PyTorch. Install the project dependencies first."
            ) from exc

    def _resolve_device(self, requested: str) -> str:
        if requested == "auto":
            return "cuda" if self._torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not self._torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false")
        return requested

    @staticmethod
    def _load_live_class(event_camera_dir: Path):
        event_camera_path = str(event_camera_dir.resolve())
        if event_camera_path not in sys.path:
            sys.path.insert(0, event_camera_path)

        # Avoid reusing a same-named module from another checkout/ref in long-lived processes.
        for module_name in (
            "IECBSLiveGPU",
            "src.dvs_sensor_gpu",
            "src.event_buffer",
            "src.dat_files",
        ):
            sys.modules.pop(module_name, None)

        try:
            module = importlib.import_module("IECBSLiveGPU")
        except ImportError as exc:
            raise RuntimeError(
                f"Could not import NIGHTRIDER gpu_pipeline from {event_camera_dir}. "
                "Make sure torch and opencv-python-headless are installed."
            ) from exc
        return module.IECBSLiveGPU


def render_event_buffer(events, width: int, height: int) -> np.ndarray:
    frame = np.full((height, width), 125, dtype=np.uint8)
    event_count = int(getattr(events, "i", 0) or 0)
    if event_count <= 0:
        return frame

    x = _array_to_numpy(events.x[:event_count]).astype(np.int64, copy=False)
    y = _array_to_numpy(events.y[:event_count]).astype(np.int64, copy=False)
    p = _array_to_numpy(events.p[:event_count]).astype(np.uint8, copy=False)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    frame[y[valid], x[valid]] = p[valid] * 255
    return frame


def _array_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    return np.asarray(value)
