#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_convert.py — stdlib-only smoke/regression tests for transcriber.py,
convert.py and server.py.

No pytest. Run directly:

    python3 test_convert.py            # full run (loads the tiny model)
    python3 test_convert.py --fast     # tier 1 only, no model, sub-second
    python3 test_convert.py --model base --device cpu
    python3 test_convert.py --keep     # keep the temp dir and print its path
    python3 test_convert.py --strict   # treat skips as failures

Tests that need the fixture, a cached/downloadable model, ffmpeg, or
fastapi+httpx are SKIPPED (not failed) when that dependency is unavailable,
so this file behaves sanely in minimal environments.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import transcriber
from transcriber import (
    Segment,
    ts_srt,
    ts_vtt,
    write_srt,
    write_vtt,
    process_file,
    load_model,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ko_sample.flac"

SRT_TS_RE = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}$")
VTT_TS_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}$")
SRT_ARROW_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$"
)
VTT_ARROW_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}$"
)

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

RESULTS = {"pass": 0, "fail": 0, "skip": 0}
ARGS = None


class Skip(Exception):
    """Raised inside a section to mark it (or the rest of it) skipped."""


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        RESULTS["pass"] += 1
        print(f"  PASS  {name}")
    else:
        RESULTS["fail"] += 1
        print(f"  FAIL  {name}{(' - ' + detail) if detail else ''}")


def skip(name: str, reason: str) -> None:
    if ARGS is not None and ARGS.strict:
        RESULTS["fail"] += 1
        print(f"  FAIL  {name} - SKIPPED ({reason}) [--strict]")
    else:
        RESULTS["skip"] += 1
        print(f"  SKIP  {name} - {reason}")


class section:
    """Context manager: turns an unexpected exception into one FAIL instead
    of aborting the whole run. Use `raise Skip(reason)` inside to mark the
    remainder of the section skipped instead."""

    def __init__(self, title: str):
        self.title = title

    def __enter__(self):
        print(f"\n=== {self.title} ===")
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            return False
        if exc_type is Skip:
            skip(self.title, str(exc))
            return True
        # Unexpected exception: record as one FAIL with a traceback line,
        # don't abort the rest of the run.
        RESULTS["fail"] += 1
        last_line = traceback.format_exc().strip().splitlines()[-1]
        print(f"  FAIL  {self.title} - unhandled exception: {last_line}")
        return True


# ---------------------------------------------------------------------------
# Shared lazy model
# ---------------------------------------------------------------------------

_shared_model = None
_shared_model_load_failed = None  # Exception, once attempted


def get_shared_model():
    """Load the tiny (or --model) model once and cache it for all tiers
    that need a model. Raises Skip on load failure that looks like a
    network/offline problem."""
    global _shared_model, _shared_model_load_failed

    if _shared_model is not None:
        return _shared_model
    if _shared_model_load_failed is not None:
        raise Skip(f"model previously failed to load: {_shared_model_load_failed}")

    try:
        print(f"  (loading model '{ARGS.model}' on {ARGS.device} ...)")
        _shared_model = load_model(ARGS.model, device=ARGS.device)
        return _shared_model
    except Exception as e:
        msg = str(e)
        offline_markers = ("ConnectionError", "OSError", "couldn't connect", "offline")
        is_offline = isinstance(e, (ConnectionError, OSError)) or any(
            m.lower() in msg.lower() for m in offline_markers
        )
        _shared_model_load_failed = e
        if is_offline:
            raise Skip(f"model not cached and no network: {e}")
        # Not clearly a network problem - let it surface as a real failure.
        raise


# ---------------------------------------------------------------------------
# Tier 1 - pure functions, no model
# ---------------------------------------------------------------------------

def tier1(tmp_dir: Path) -> None:
    with section("ts_srt / ts_vtt basic formatting"):
        check("ts_srt(0.0)", ts_srt(0.0) == "00:00:00,000", ts_srt(0.0))
        check(
            "ts_srt(3661.5)",
            ts_srt(3661.5) == "01:01:01,500",
            ts_srt(3661.5),
        )
        check(
            "ts_vtt(3661.5)",
            ts_vtt(3661.5) == "01:01:01.500",
            ts_vtt(3661.5),
        )

    with section("ts_srt rounding edges"):
        check(
            "ts_srt(1.9999) rounds up to next second",
            ts_srt(1.9999) == "00:00:02,000",
            ts_srt(1.9999),
        )
        check(
            "ts_srt(59.9999) rounds up to next minute",
            ts_srt(59.9999) == "00:01:00,000",
            ts_srt(59.9999),
        )

    with section("ts_srt / ts_vtt format regex, incl. negatives"):
        values = [0.0, 0.001, 1.0, 59.999, 60.0, 3599.999, 3600.0, 86399.999,
                  -1.0, -0.5, -100.0]
        srt_ok = all(SRT_TS_RE.match(ts_srt(v)) for v in values)
        vtt_ok = all(VTT_TS_RE.match(ts_vtt(v)) for v in values)
        check("ts_srt matches HH:MM:SS,mmm for all sample values", srt_ok)
        check("ts_vtt matches HH:MM:SS.mmm for all sample values", vtt_ok)
        # Negatives should clamp to zero, not go negative or raise.
        check("ts_srt(-1.0) clamps to 00:00:00,000", ts_srt(-1.0) == "00:00:00,000")
        check("ts_vtt(-0.5) clamps to 00:00:00.000", ts_vtt(-0.5) == "00:00:00.000")

    with section("write_srt / write_vtt on hand-built segments"):
        segs = [
            Segment(index=1, start=0.0, end=1.5, text="Hello there."),
            Segment(index=2, start=1.5, end=3.25, text="General Kenobi."),
            Segment(index=3, start=3.25, end=4.0, text="You are a bold one."),
        ]
        srt_path = tmp_dir / "manual.srt"
        vtt_path = tmp_dir / "manual.vtt"
        write_srt(segs, srt_path)
        write_vtt(segs, vtt_path)

        check("SRT file exists", srt_path.exists())
        check("VTT file exists", vtt_path.exists())
        check("SRT file non-empty", srt_path.stat().st_size > 0)
        check("VTT file non-empty", vtt_path.stat().st_size > 0)

        srt_text = srt_path.read_text(encoding="utf-8")
        blocks = [b for b in srt_text.strip().split("\n\n") if b.strip()]
        check("SRT has 3 blocks", len(blocks) == 3, f"got {len(blocks)}")
        parses = True
        for i, block in enumerate(blocks, start=1):
            lines = block.splitlines()
            if len(lines) < 3:
                parses = False
                break
            if lines[0] != str(i):
                parses = False
                break
            if not SRT_ARROW_RE.match(lines[1]):
                parses = False
                break
            if not lines[2].strip():
                parses = False
                break
        check("SRT block structure parses (index/timestamp/text)", parses)

        vtt_text = vtt_path.read_text(encoding="utf-8")
        vtt_lines = vtt_text.splitlines()
        check("VTT first line is exactly 'WEBVTT'", vtt_lines and vtt_lines[0] == "WEBVTT")

    with section("process_file rejects unsupported extension"):
        bogus = tmp_dir / "not_a_media_file.xyz"
        bogus.write_bytes(b"not really media")
        raised = False
        try:
            process_file(bogus, model=None, output_dir=tmp_dir)
        except ValueError:
            raised = True
        check("process_file(.xyz) raises ValueError", raised)


# ---------------------------------------------------------------------------
# Tier 2 - end-to-end audio, default params
# ---------------------------------------------------------------------------

def _parse_ts(ts: str, sep: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(sep)
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _check_srt_structure(srt_path: Path, label: str) -> list[tuple[float, float]]:
    """Run the structural checks shared by tier 2 and tier 3 and return the
    list of (start, end) tuples parsed from the SRT."""
    text = srt_path.read_text(encoding="utf-8")
    blocks = [b for b in text.strip().split("\n\n") if b.strip()]
    check(f"{label}: at least one SRT block", len(blocks) >= 1, f"got {len(blocks)}")

    indices = []
    timestamps_ok = True
    text_lines_ok = True
    cues: list[tuple[float, float]] = []

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            timestamps_ok = False
            text_lines_ok = False
            continue
        try:
            indices.append(int(lines[0]))
        except ValueError:
            indices.append(None)
        if not SRT_ARROW_RE.match(lines[1]):
            timestamps_ok = False
        else:
            start_s, end_s = lines[1].split(" --> ")
            cues.append((_parse_ts(start_s, ","), _parse_ts(end_s, ",")))
        if not any(l.strip() for l in lines[2:]):
            text_lines_ok = False

    check(f"{label}: block index lines are integers", all(i is not None for i in indices))
    check(
        f"{label}: indices contiguous 1..N ascending",
        indices == list(range(1, len(indices) + 1)),
        str(indices),
    )
    check(f"{label}: timestamp lines match anchored regex", timestamps_ok)
    check(f"{label}: every block has non-empty text after timestamp", text_lines_ok)

    return cues


def _check_timestamps_sane(cues: list[tuple[float, float]], duration: float, label: str) -> None:
    starts = [c[0] for c in cues]
    ends = [c[1] for c in cues]

    check(f"{label}: start <= end per cue", all(s <= e for s, e in cues))
    check(
        f"{label}: starts non-decreasing",
        all(starts[i] <= starts[i + 1] for i in range(len(starts) - 1)),
    )
    check(
        f"{label}: ends non-decreasing",
        all(ends[i] <= ends[i + 1] for i in range(len(ends) - 1)),
    )
    check(f"{label}: first start >= 0", starts[0] >= 0 if starts else False)

    # Deliberately no upper bound tied tightly to audio duration: Whisper's
    # final window can legitimately run past the end of the source audio
    # (measured 20.70s of cue time on this project's 12.4s fixture with
    # vad_filter=False). Only a loose sanity ceiling, if any.
    ceiling = 2 * duration + 30
    check(
        f"{label}: ends within loose sanity ceiling ({ceiling:.1f}s)",
        all(e <= ceiling for e in ends),
    )


def _check_vtt_matches_srt(vtt_path: Path, srt_cues: list[tuple[float, float]], label: str) -> None:
    text = vtt_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    check(f"{label}: VTT first line exactly 'WEBVTT'", bool(lines) and lines[0] == "WEBVTT")

    ts_lines = [l for l in lines if "-->" in l]
    no_comma = all("," not in l for l in ts_lines)
    check(f"{label}: no comma in any VTT timestamp line", no_comma)

    all_match = all(VTT_ARROW_RE.match(l) for l in ts_lines)
    check(f"{label}: all VTT timestamp lines match .mmm regex", all_match)

    check(
        f"{label}: VTT cue count equals SRT cue count",
        len(ts_lines) == len(srt_cues),
        f"vtt={len(ts_lines)} srt={len(srt_cues)}",
    )

    within_tol = True
    for line, (s_start, s_end) in zip(ts_lines, srt_cues):
        start_s, end_s = line.split(" --> ")
        v_start = _parse_ts(start_s, ".")
        v_end = _parse_ts(end_s, ".")
        if abs(v_start - s_start) > 0.001 or abs(v_end - s_end) > 0.001:
            within_tol = False
            break
    check(f"{label}: VTT cue times equal SRT cue times within 1ms", within_tol)


def tier2(tmp_dir: Path) -> None:
    with section("Tier 2 setup: fixture + model"):
        if not FIXTURE.exists():
            raise Skip(f"fixture not found: {FIXTURE}")
        model = get_shared_model()

    with section("process_file end-to-end on ko_sample.flac (default output_dir)"):
        audio_copy = tmp_dir / "ko_sample.flac"
        shutil.copyfile(FIXTURE, audio_copy)

        result = process_file(audio_copy, model=model, output_dir=None)
        check("process_file returns a 2-tuple", isinstance(result, tuple) and len(result) == 2)
        srt_path, vtt_path = result
        check("srt_path is a Path", isinstance(srt_path, Path))
        check("vtt_path is a Path", isinstance(vtt_path, Path))

        expected_srt = tmp_dir / "ko_sample.srt"
        expected_vtt = tmp_dir / "ko_sample.vtt"
        check("srt output at ko_sample.srt", srt_path == expected_srt, str(srt_path))
        check("vtt output at ko_sample.vtt", vtt_path == expected_vtt, str(vtt_path))
        check("srt file exists", srt_path.exists())
        check("vtt file exists", vtt_path.exists())
        check("srt file non-empty", srt_path.exists() and srt_path.stat().st_size > 0)
        check("vtt file non-empty", vtt_path.exists() and vtt_path.stat().st_size > 0)

    with section("Tier 2: SRT structure"):
        cues = _check_srt_structure(srt_path, "ko_sample")

    with section("Tier 2: timestamps sane"):
        if not cues:
            raise Skip("no cues parsed from SRT, cannot check timestamp sanity")
        # 12.4s fixture duration per the task brief.
        _check_timestamps_sane(cues, duration=12.4, label="ko_sample")

    with section("Tier 2: VTT matches SRT"):
        _check_vtt_matches_srt(vtt_path, cues, "ko_sample")

    with section("Tier 2: both files decode as UTF-8"):
        try:
            srt_path.read_text(encoding="utf-8")
            vtt_path.read_text(encoding="utf-8")
            ok = True
        except UnicodeDecodeError:
            ok = False
        check("srt/vtt decode as UTF-8", ok)


# ---------------------------------------------------------------------------
# Tier 3 - video path + server
# ---------------------------------------------------------------------------

def tier3_video(tmp_dir: Path) -> None:
    with section("Tier 3: mux fixture into mp4"):
        if not FIXTURE.exists():
            raise Skip(f"fixture not found: {FIXTURE}")
        if shutil.which("ffmpeg") is None:
            raise Skip("ffmpeg not on PATH")
        model = get_shared_model()

        mp4_path = tmp_dir / "ko_sample.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=160x120:r=5",
            "-i", str(FIXTURE),
            "-shortest",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
            "-c:a", "aac",
            str(mp4_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise Skip(
                "ffmpeg mux failed: "
                + result.stderr.decode(errors="replace")[-500:]
            )

    with section("Tier 3: process_file on muxed mp4 (video/extract_audio path)"):
        out_dir = tmp_dir / "video_out"
        srt_path, vtt_path = process_file(mp4_path, model=model, output_dir=out_dir)
        check("video srt file exists", srt_path.exists())
        check("video vtt file exists", vtt_path.exists())
        check("video srt non-empty", srt_path.exists() and srt_path.stat().st_size > 0)
        check("video vtt non-empty", vtt_path.exists() and vtt_path.stat().st_size > 0)

    with section("Tier 3: video SRT structure"):
        _check_srt_structure(srt_path, "video")


def tier3_server(tmp_dir: Path) -> None:
    with section("Tier 3: server.py via TestClient"):
        try:
            import fastapi  # noqa: F401
            import httpx  # noqa: F401
        except ImportError as e:
            raise Skip(f"fastapi/httpx not available: {e}")
        if not FIXTURE.exists():
            raise Skip(f"fixture not found: {FIXTURE}")

        from fastapi.testclient import TestClient
        import server

        client = TestClient(server.app)

        with open(FIXTURE, "rb") as f:
            resp = client.post(
                "/jobs",
                files={"file": ("ko_sample.flac", f, "audio/flac")},
                data={"model": "tiny", "device": ARGS.device},
            )
        check("POST /jobs returns 200", resp.status_code == 200, str(resp.status_code))
        job_id = resp.json()["job_id"]

        import time as _time
        deadline = _time.time() + 300
        status = None
        while _time.time() < deadline:
            r = client.get(f"/jobs/{job_id}")
            status = r.json()["status"]
            if status in ("done", "error"):
                break
            _time.sleep(0.5)

        check("job reaches terminal state within 300s", status in ("done", "error"), str(status))
        check("job status is 'done'", status == "done", str(status))

        if status == "done":
            dl = client.get(f"/jobs/{job_id}/download/srt")
            check("GET download/srt returns 200", dl.status_code == 200, str(dl.status_code))
            if dl.status_code == 200:
                srt_tmp = tmp_dir / "server_job.srt"
                srt_tmp.write_bytes(dl.content)
                _check_srt_structure(srt_tmp, "server-srt")

        dl_bad = client.get(f"/jobs/{job_id}/download/xml")
        check("GET download/xml returns 400", dl_bad.status_code == 400, str(dl_bad.status_code))

        dl_404 = client.get("/jobs/nonexistent")
        check("GET /jobs/nonexistent returns 404", dl_404.status_code == 404, str(dl_404.status_code))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global ARGS
    parser = argparse.ArgumentParser(description="Tests for the sub-converter")
    parser.add_argument("--fast", action="store_true", help="Tier 1 only, no model, sub-second")
    parser.add_argument("--model", default="tiny", help="Whisper model size for model-dependent tiers")
    parser.add_argument("--device", default="cpu", help="Compute device for model-dependent tiers")
    parser.add_argument("--keep", action="store_true", help="Preserve and print the temp dir")
    parser.add_argument("--strict", action="store_true", help="Treat skips as failures")
    ARGS = parser.parse_args()

    tmp_ctx = tempfile.TemporaryDirectory(prefix="sub_converter_test_")
    tmp_dir = Path(tmp_ctx.name)

    try:
        print(f"Temp dir: {tmp_dir}")
        tier1(tmp_dir)

        if not ARGS.fast:
            tier2(tmp_dir)
            tier3_video(tmp_dir)
            tier3_server(tmp_dir)
        else:
            print("\n=== --fast: skipping tiers 2 and 3 (model-dependent) ===")

        print(
            f"\n{RESULTS['pass']} passed, {RESULTS['fail']} failed, "
            f"{RESULTS['skip']} skipped"
        )

        if ARGS.keep:
            # Detach so TemporaryDirectory's cleanup doesn't remove it.
            tmp_ctx._finalizer.detach()
            print(f"Kept temp dir: {tmp_dir}")

        sys.exit(1 if RESULTS["fail"] > 0 else 0)
    finally:
        if not ARGS.keep:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
