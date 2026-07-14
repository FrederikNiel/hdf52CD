from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .hdf5_convert import (
    ConversionConfig,
    ConversionResult,
    convert_file,
    discover_inputs,
    list_frame_candidates,
    write_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hdf5-to-cd-dataset",
        description="Convert HDF5 JPEG frame datasets using NIGHTRIDER gpu_pipeline CD simulation.",
    )
    parser.add_argument("--input", action="append", required=True, help="Input file, glob, or directory. May be repeated.")
    parser.add_argument("--output-dir", default="cd_hdf5", help="Directory for converted HDF5 files.")
    parser.add_argument("--frames-key", help="HDF5 dataset path containing compressed JPEG frames.")
    parser.add_argument("--device", default="cpu", choices=["auto", "cuda", "cpu"], help="NIGHTRIDER simulation device. Defaults to CPU for non-CUDA machines.")
    parser.add_argument("--nightrider-path", help="Path to a NIGHTRIDER checkout. Defaults to $NIGHTRIDER_PATH or /home/labpcadm/Desktop/NIGHTRIDER.")
    parser.add_argument("--nightrider-ref", default="origin/gpu_pipeline", help="Git ref containing NIGHTRIDER gpu_pipeline files when the checkout is on another branch.")
    parser.add_argument("--th-pos", type=float, default=0.4, help="NIGHTRIDER positive log threshold.")
    parser.add_argument("--th-neg", type=float, default=0.4, help="NIGHTRIDER negative log threshold.")
    parser.add_argument("--ref-us", type=int, default=100, help="NIGHTRIDER refractory period in microseconds.")
    parser.add_argument("--dt-us", type=int, default=16667, help="NIGHTRIDER frame delta in microseconds.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="Output JPEG quality, 1-100.")
    parser.add_argument("--manifest", help="JSONL manifest path. Defaults to <output-dir>/manifest.jsonl.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing converted files.")
    parser.add_argument("--dry-run", action="store_true", help="List detected inputs/candidate frame datasets without writing outputs.")
    return parser


def progress_printer(input_path: Path):
    last_percent = -1

    def report(completed: int, total: int) -> None:
        nonlocal last_percent
        if total <= 0:
            return
        percent = min(100, int(completed * 100 / total))
        if percent == last_percent and completed < total:
            return
        last_percent = percent
        print(f"progress {input_path.name}: {percent:3d}% ({completed}/{total} frames)", flush=True)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.ref_us < 0:
        parser.error("--ref-us must be non-negative")
    if args.dt_us <= 0:
        parser.error("--dt-us must be positive")

    inputs = discover_inputs(args.input)
    if not inputs:
        print("No input HDF5 files found.", file=sys.stderr)
        return 2

    if args.dry_run:
        for input_path in inputs:
            try:
                candidates = list_frame_candidates(input_path)
                candidate_text = ", ".join(candidates) if candidates else "(none)"
                print(f"{input_path}: {candidate_text}")
            except Exception as exc:
                print(f"{input_path}: error: {exc}", file=sys.stderr)
        return 0

    config = ConversionConfig(
        frames_key=args.frames_key,
        device=args.device,
        jpeg_quality=args.jpeg_quality,
        nightrider_path=args.nightrider_path,
        nightrider_ref=args.nightrider_ref,
        th_pos=args.th_pos,
        th_neg=args.th_neg,
        ref_us=args.ref_us,
        dt_us=args.dt_us,
        overwrite=args.overwrite,
    )
    output_dir = Path(args.output_dir)
    manifest = Path(args.manifest) if args.manifest else output_dir / "manifest.jsonl"

    results: list[ConversionResult] = []
    exit_code = 0
    for input_path in inputs:
        try:
            result = convert_file(input_path, output_dir, config, progress_callback=progress_printer(input_path))
            print(f"converted {result.input_path} -> {result.output_path} ({result.frame_count} frames, {result.device})")
            results.append(result)
        except Exception as exc:
            exit_code = 1
            error_result = ConversionResult(
                input_path=str(input_path),
                output_path=str(output_dir / f"{input_path.stem}_cd{input_path.suffix}"),
                frames_key=args.frames_key or "",
                frame_count=0,
                width=0,
                height=0,
                backend="nightrider_gpu_pipeline",
                device=args.device,
                status="error",
                error=str(exc),
            )
            print(f"failed {input_path}: {exc}", file=sys.stderr)
            results.append(error_result)

    write_manifest(manifest, results)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
