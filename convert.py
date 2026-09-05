#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Korean audio/video → English subtitles (.srt + .vtt)

Uses faster-whisper offline: no API key needed.
ffmpeg must be on PATH for video files.

Usage:
    python convert.py <file> [file ...]
    python convert.py video.mp4 audio.mp3 --model medium --output-dir ./out
"""
import argparse
import sys
from pathlib import Path

from transcriber import load_model, process_file, VIDEO_EXTS, AUDIO_EXTS, MODEL_SIZES


def process(input_path: Path, model, output_dir: Path | None, vad_filter: bool) -> None:
    ext = input_path.suffix.lower()
    if ext not in VIDEO_EXTS and ext not in AUDIO_EXTS:
        print(f"\n[{input_path.name}] skipped — unsupported extension '{ext}'")
        return

    print(f"\n[{input_path.name}] processing …")

    def progress_cb(n: int, total, fraction) -> None:
        if fraction is not None:
            print(f"  [segment {n}] {fraction * 100:.0f}%")
        else:
            print(f"  [segment {n}]")

    # Use keyword args to avoid positional-slot confusion (model_size is arg 3,
    # not output_dir).
    process_file(
        input_path,
        model=model,
        output_dir=output_dir,
        progress_cb=progress_cb,
        vad_filter=vad_filter,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Korean audio/video to English subtitles (.srt + .vtt)"
    )
    parser.add_argument("files", nargs="+", help="Input audio or video file(s)")
    parser.add_argument(
        "--model",
        default="small",
        choices=list(MODEL_SIZES),
        help="Whisper model size (default: small)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Directory for output files (default: same as input)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Compute device (default: auto-detect)",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help=(
            "Disable voice-activity filtering. Use for synthetic/TTS or "
            "heavily processed audio that VAD wrongly discards as silence."
        ),
    )
    args = parser.parse_args()

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    print(f"Loading Whisper model '{args.model}' on {device} …")
    model = load_model(args.model, device=device)

    output_dir = Path(args.output_dir) if args.output_dir else None

    errors = []
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"[{f}] file not found — skipping")
            continue
        try:
            process(p, model, output_dir, vad_filter=not args.no_vad)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((f, e))

    if errors:
        print(f"\n{len(errors)} file(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
