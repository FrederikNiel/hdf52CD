# hdf52cd

Convert HRDF-style HDF5 files that contain compressed JPEG video frames into
index-stable CD-frame HDF5 files for AI training.

The converter writes new files and keeps the original HDF5 structure, action
arrays, frame indexes, attributes, and metadata unchanged. Only the selected
JPEG frame dataset is replaced with grayscale CD-frame JPEGs.

CD simulation is backed by the NIGHTRIDER `gpu_pipeline` implementation:

- `EventCamera/IECBSLiveGPU.py`
- `EventCamera/src/dvs_sensor_gpu.py`
- `IECBSLiveGPU.process_batch(...)`
- `DvsSensorGPU.step(...)`

This tool does not use a frame-difference approximation. On CPU it still calls the same NIGHTRIDER `gpu_pipeline` PyTorch code with `device="cpu"`. It loads the exact
NIGHTRIDER GPU branch code from a local NIGHTRIDER checkout. If the checkout is
on another branch, it reads `origin/gpu_pipeline` through Git without changing
the NIGHTRIDER working tree.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
# Install CPU PyTorch first from the official CPU wheel index.
pip install torch
pip install opencv-python
pip install -e ".[test]"
```

The NIGHTRIDER checkout is expected at `/home/labpcadm/Desktop/NIGHTRIDER` by
default. Override it with `--nightrider-path` or `NIGHTRIDER_PATH`.

## Usage

```bash
hdf5-to-cd-dataset \
  --input "/home/frederik/Documents/rover_rgbd_hd720_expert_400k_10pct.hdf5" \
  --output-dir /home/frederik/Documents/ \
  --frames-key /observations/rgb_jpeg \
  --nightrider-path /home/frederik/Documents/NIGHTRIDER \
  --nightrider-ref origin/gpu_pipeline \
  --device cuda \
  --jpeg-quality 95
```

Use `--dry-run` to inspect candidate frame datasets without writing output.

```bash
hdf5-to-cd-dataset --input sample.hdf5 --dry-run
```

## NIGHTRIDER Parameters

Defaults match the `gpu_pipeline` branch values in `IECBSLiveGPU`:

- `--th-pos 0.4`
- `--th-neg 0.4`
- `--ref-us 100`
- `--dt-us 16667`

`--device cpu` is the default and is the expected mode on this PC.
`--device auto` uses CUDA only when PyTorch reports CUDA availability, otherwise CPU.
`--device cuda` fails if CUDA is unavailable.

## CD Frame Rendering

NIGHTRIDER emits event buffers. HDF5 training output needs compressed image
frames, so each event buffer is rendered using the same binary convention as
`EventCameraInterfaceGPU`:

- no event: gray value `125`
- OFF event: gray value `0`
- ON event: gray value `255`

Frames are stored as grayscale JPEGs.

## Alignment Policy

For `N` input JPEG frames, the output contains exactly `N` CD JPEG frames.

- The first source frame initializes NIGHTRIDER DVS state.
- CD frame `i` is generated from the NIGHTRIDER events produced while processing source frame `i`.
- Output index `0` repeats the first generated CD frame, so frame indexes stay action-aligned.
- If there is only one input frame, CD frame `0` is a black grayscale JPEG.

This keeps JPEG indexes aligned with action/index datasets for behavior cloning.
