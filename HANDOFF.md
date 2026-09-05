# Handoff — sub-converter

Korean audio/video → English subtitles (`.srt` + `.vtt`), fully offline via `faster-whisper`. No API key.

**Location:** `/workspace/sub-converter` (moved here from `/workspace/english/src/sub-converter`; the old path is now empty).
**Not a git repository.** Nothing is committed anywhere. `git init` if you want history.
**Scope:** this project only. It has no relationship to the English for Life site that still lives in `/workspace/english` — do not pull that work into this session.

---

## Status: complete and verified

Full suite passes at this path, including the live HTTP server tier:

```
62 passed, 0 failed, 0 skipped
```

Verified end to end, not just unit-tested: a Korean FLAC uploaded over HTTP to the running server came back as English SRT.

```
1
00:00:00,000 --> 00:00:02,840
In the story of the body, he says,
```

Translation quality is mediocre only because that run used the `tiny` model. Use `small` (the default) or larger for real work.

---

## Files

| File | Role |
|---|---|
| `transcriber.py` | Core library. `load_model()`, `transcribe()`, `process_file()`, `Segment`, `TranscribeResult`, `ts_srt`/`ts_vtt`, `write_srt`/`write_vtt`, `extract_audio`, `MODEL_SIZES`, `VIDEO_EXTS`/`AUDIO_EXTS` |
| `convert.py` | CLI. `--model --output-dir --device --no-vad` |
| `server.py` | FastAPI. `POST /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/download/{srt\|vtt}` |
| `static/index.html` | Web UI — drag-drop, model picker, progress polling, preview |
| `test_convert.py` | 62 assertions, stdlib only, no pytest |
| `fixtures/ko_sample.flac` | 12.4 s Korean speech test fixture |
| `README.md` | User docs incl. a detailed "How it works" |
| `Dockerfile`, `.dockerignore`, `requirements.txt` | Packaging |

---

## Read this before touching VAD — a wrong diagnosis already cost time here

During development, a Korean test clip returned **0 segments**. This was initially blamed on `vad_filter`. **That was wrong**, and the mistake is easy to repeat:

- `faster-whisper` 1.2.1 defaults `vad_filter=False` on `WhisperModel` (it defaults `True` on `BatchedInferencePipeline`). VAD was never running.
- Re-tested both files with VAD both on and off: **all four combinations returned 0 segments.**
- The real cause was **`no_speech_threshold`** — Whisper's own no-speech head scored the audio as non-speech and dropped every segment. `no_speech_threshold=None` recovered 7 segments.
- The root problem was the **audio**: espeak-ng formant synthesis isn't speech-like enough for Whisper's detector. Real neural TTS works with entirely stock parameters.

VAD and `no_speech_threshold` are **two independent gates at different pipeline stages** — VAD pre-filters the waveform, `no_speech_threshold` gates per 30 s window at decode time. Either can silently produce an empty subtitle file. If you hit 0 segments, check both, and suspect the audio before the parameters.

`vad_filter` is now passed **explicitly** everywhere (default `True`) precisely because the library default is inconsistent across interfaces and has moved between releases — never let it be inherited.

---

## The fixture — has license terms

`fixtures/ko_sample.flac` — 12.4 s, 16 kHz mono 16-bit FLAC, 238 KB, md5 `d9246697c0e760e0a9b3399e47d2dbca`.

Derived from Piper's `ko_KR-kss-medium` voice (HF `rhasspy/piper-voices`, path `ko/ko_KR/kss/medium`), trained on the KSS dataset — **CC BY-NC-SA 4.0**. Attribution is required, share-alike applies, and it is **non-commercial only**. Fine for local/personal use. If this project ever goes commercial or public, replace the fixture.

Verified behaviour with `tiny`, `language="ko"`, `task="translate"`, otherwise default params:
- `vad_filter=True` → 2 segments, last cue ends 12.38 s
- `vad_filter=False` → 2 segments, last cue ends **20.70 s** — overruns the 12.4 s file

That overrun is a genuine Whisper final-window quirk, **not a bug**. The test therefore does **not** assert an upper timestamp bound against audio duration. Don't "fix" this by adding one.

---

## Environment

Installed in this container with `pip3 install <pkg> --break-system-packages` (PEP 668 blocks plain `pip3 install`):

```
faster-whisper>=1.0.0,<2   fastapi   uvicorn[standard]   python-multipart   httpx
```

`ffmpeg` is on PATH. The whisper `tiny` model is cached at `~/.cache/huggingface/hub/models--Systran--faster-whisper-tiny`. `torch` is **not** installed — `load_model`'s auto-detect therefore always falls back to CPU, so a GPU box needs `--device cuda` passed explicitly.

**No Docker binary or daemon exists in this workspace.** `Dockerfile` is written but has **never been built**. Do not claim it works until someone runs `docker build`.

---

## Running it

```bash
cd /workspace/sub-converter

python3 test_convert.py            # full suite, ~3 min (loads the model)
python3 test_convert.py --fast     # tier 1 only, sub-second, no model
# other flags: --model --device --keep --strict

python3 convert.py <file> --model small
python3 -m uvicorn server:app --host 127.0.0.1 --port 8000   # then open http://localhost:8000/
```

The test suite is mutation-checked: reintroducing the millisecond-overflow bug produced 4 failures, so it catches real regressions rather than passing vacuously. `test_convert.py` resolves its fixture via `Path(__file__).parent`, so it runs from any working directory.

---

## Bugs already found and fixed — don't reintroduce

1. `ts_srt` emitted invalid `00:00:01,1000` for `1.9999`. Both formatters now derive every field from one rounded integer-millisecond value and clamp negatives.
2. `vad_filter` was never passed → behaviour inherited from an inconsistent library default. Now explicit.
3. Empty-text segments produced structurally invalid cues. Now skipped **before** index assignment so indices stay contiguous 1..N.
4. Progress bar could never move — the callback always passed `total=None` and the server guarded on `total > 0`. Callback is now `(done, total_hint, fraction)` with `fraction` derived from `info.duration`.
5. `server.py` reloaded the whole model on every job. Now a `threading.Lock`-guarded cache keyed by `(model_size, device)`.
6. Job temp dirs, the `jobs` dict, and uploaded files leaked forever. Now a 1-hour TTL sweep plus upload deletion in a `finally`.
7. Client-controlled filename was interpolated into `Content-Disposition`, and the header was emitted twice. Now sanitized to `[A-Za-z0-9._-]`, capped at 100 chars, falling back to `subtitles`; the duplicate header was removed.

---

## Known limitations

- Progress reads 0 → 100 on short files. Not a defect: model loading dominates, and progress can't advance before the first segment decodes.
- Server is single-process and keeps jobs in memory — restarting loses them.
- No speaker diarization, no subtitle line-length wrapping.
- Utterance-level timestamps only unless `word_timestamps=True`.
- `tiny`/`base` degrade notably on the `translate` task specifically, more than on `transcribe`.

## Possible next steps

None are blocking — the tool is working and fully tested.

- Build and actually verify the Docker image somewhere with a daemon.
- `git init` and make a first commit.
- Add a VAD checkbox to `static/index.html` (the `vad` form field already exists server-side and defaults to true).
- Persist jobs so a server restart doesn't lose them.
- Expose `no_speech_threshold` as a CLI flag for the difficult-audio case described above.
