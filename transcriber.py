"""
transcriber.py — reusable library for Korean audio/video → English subtitle conversion.

Imported by both the CLI (convert.py) and the FastAPI web server (server.py).
Requires faster-whisper and ffmpeg on PATH for video files.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, NamedTuple, Optional


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}

MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v2", "large-v3")


class Segment(NamedTuple):
    index: int
    start: float
    end: float
    text: str


class TranscribeResult(NamedTuple):
    segments: list[Segment]
    duration: float
    language: str


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def ts_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    total_ms = int(round(max(0.0, seconds) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def ts_vtt(seconds: float) -> str:
    """Format seconds as WebVTT timestamp: HH:MM:SS.mmm"""
    total_ms = int(round(max(0.0, seconds) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


# ---------------------------------------------------------------------------
# Subtitle writers
# ---------------------------------------------------------------------------

def write_srt(segments: list[Segment], path: Path) -> None:
    """Write segments to an SRT subtitle file."""
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"{seg.index}\n")
            f.write(f"{ts_srt(seg.start)} --> {ts_srt(seg.end)}\n")
            f.write(seg.text.strip() + "\n\n")


def write_vtt(segments: list[Segment], path: Path) -> None:
    """Write segments to a WebVTT subtitle file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            f.write(f"{seg.index}\n")
            f.write(f"{ts_vtt(seg.start)} --> {ts_vtt(seg.end)}\n")
            f.write(seg.text.strip() + "\n\n")


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: Path, tmp_wav: Path) -> None:
    """Extract mono 16 kHz PCM audio from a video file using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ar", "16000", "-ac", "1",
        str(tmp_wav),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed:\n{result.stderr.decode(errors='replace')}"
        )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_size: str = "small",
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
):
    """Load and return a WhisperModel.

    Parameters
    ----------
    model_size:
        One of "tiny", "base", "small", "medium", "large-v2", "large-v3".
    device:
        "cuda" or "cpu". When None, auto-detected: cuda if torch is available
        and reports a GPU, otherwise cpu.
    compute_type:
        "float16" for cuda, "int8" for cpu. Inferred from *device* when None.

    Returns
    -------
    faster_whisper.WhisperModel
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is not installed. Run:  pip install faster-whisper"
        ) from exc

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    return WhisperModel(model_size, device=device, compute_type=compute_type)


# ---------------------------------------------------------------------------
# Core transcription
# ---------------------------------------------------------------------------

def transcribe(
    audio_path: "str | Path",
    model=None,
    model_size: str = "small",
    device: Optional[str] = None,
    progress_cb: Optional[Callable[[int, Optional[int], Optional[float]], None]] = None,
    vad_filter: bool = True,
    vad_parameters: Optional[dict] = None,
) -> TranscribeResult:
    """Transcribe Korean audio to English text using faster-whisper.

    Parameters
    ----------
    audio_path:
        Path to a WAV or other audio file accepted by faster-whisper.
    model:
        A pre-loaded WhisperModel. When None, ``load_model`` is called with
        *model_size* and *device*.
    model_size:
        Whisper model size used when *model* is None.
    device:
        Compute device used when *model* is None.
    progress_cb:
        Optional callback invoked after each segment is processed.
        Signature: ``progress_cb(done_segments: int, total_hint: int | None,
        fraction: float | None)``. *total_hint* is None when the total cannot
        be determined ahead of time. *fraction* is an estimate of overall
        progress (0.0-0.99) based on segment end time vs. audio duration, or
        None when duration is unknown.
    vad_filter:
        Whether to apply voice-activity detection to skip silence before
        transcribing. Passed explicitly to ``model.transcribe`` because the
        underlying default differs between WhisperModel (False) and
        BatchedInferencePipeline (True), and has changed across releases.
        Note that VAD can discard synthetic/TTS or heavily noise-gated audio
        entirely if it doesn't look like natural speech — set this to False
        for such inputs.
    vad_parameters:
        Optional dict of VAD tuning parameters, forwarded to
        ``model.transcribe`` only when not None.

    Returns
    -------
    TranscribeResult
    """
    if model is None:
        model = load_model(model_size, device)

    audio_path = Path(audio_path)

    transcribe_kwargs = dict(
        language="ko",
        task="translate",
        beam_size=5,
        vad_filter=vad_filter,
    )
    if vad_parameters is not None:
        transcribe_kwargs["vad_parameters"] = vad_parameters

    raw_segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)

    duration = float(info.duration or 0.0)
    language = info.language or "ko"

    segments: list[Segment] = []
    for raw in raw_segments:
        if not raw.text.strip():
            continue
        idx = len(segments) + 1
        seg = Segment(
            index=idx,
            start=raw.start,
            end=raw.end,
            text=raw.text,
        )
        segments.append(seg)
        if progress_cb is not None:
            fraction = min(0.99, raw.end / duration) if duration > 0 else None
            progress_cb(idx, None, fraction)

    return TranscribeResult(segments=segments, duration=duration, language=language)


# ---------------------------------------------------------------------------
# High-level file processor
# ---------------------------------------------------------------------------

def process_file(
    input_path: "str | Path",
    model=None,
    model_size: str = "small",
    device: Optional[str] = None,
    output_dir: "Optional[str | Path]" = None,
    progress_cb: Optional[Callable[[int, Optional[int], Optional[float]], None]] = None,
    vad_filter: bool = True,
) -> tuple[Path, Path]:
    """Convert a Korean audio or video file to .srt and .vtt subtitle files.

    Parameters
    ----------
    input_path:
        Path to the source audio or video file.
    model:
        Pre-loaded WhisperModel; loaded automatically when None.
    model_size:
        Whisper model size used when *model* is None.
    device:
        Compute device used when *model* is None.
    output_dir:
        Directory for output .srt/.vtt files. Defaults to the same directory
        as *input_path*.
    progress_cb:
        Forwarded to :func:`transcribe`.
    vad_filter:
        Forwarded to :func:`transcribe`. Note that VAD can discard
        synthetic/TTS or heavily noise-gated audio entirely; disable it for
        such inputs.

    Returns
    -------
    tuple[Path, Path]
        ``(srt_path, vtt_path)``

    Raises
    ------
    ValueError
        If the file extension is not in VIDEO_EXTS or AUDIO_EXTS.
    RuntimeError
        If ffmpeg fails during audio extraction.
    """
    input_path = Path(input_path)
    ext = input_path.suffix.lower()

    tmp_file = None
    try:
        if ext in VIDEO_EXTS:
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_file.close()
            extract_audio(input_path, Path(tmp_file.name))
            audio_path = tmp_file.name
        elif ext in AUDIO_EXTS:
            audio_path = str(input_path)
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported video: {sorted(VIDEO_EXTS)}, "
                f"audio: {sorted(AUDIO_EXTS)}"
            )

        result = transcribe(
            audio_path,
            model=model,
            model_size=model_size,
            device=device,
            progress_cb=progress_cb,
            vad_filter=vad_filter,
        )

        out_dir = Path(output_dir) if output_dir is not None else input_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = input_path.stem

        srt_path = out_dir / f"{stem}.srt"
        vtt_path = out_dir / f"{stem}.vtt"

        write_srt(result.segments, srt_path)
        write_vtt(result.segments, vtt_path)

        return srt_path, vtt_path

    finally:
        if tmp_file is not None and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
