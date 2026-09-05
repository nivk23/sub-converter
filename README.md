# sub-converter

Korean audio/video → English subtitles (`.srt` + `.vtt`), fully offline, no API key.

## Install

```bash
pip install -r requirements.txt
```

- `ffmpeg` must be on `PATH` if you want to feed it video files (it's used to extract audio; audio-only inputs like `.flac`/`.wav`/`.mp3` don't need it).
- Whisper model weights auto-download on first use to `~/.cache/huggingface` (from the `Systran/faster-whisper-*` repos, in CTranslate2 format — see below). Subsequent runs reuse the cache and work offline.

## Usage

### CLI

```bash
python convert.py video.mp4 audio.mp3 --model medium --output-dir ./out
```

Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `small` | One of `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` |
| `--output-dir` | same dir as input | Where to write `.srt`/`.vtt` |
| `--device` | auto-detect | `cpu` or `cuda` |
| `--no-vad` | off (VAD on) | Disable voice-activity filtering — see [How it works](#how-it-works) for why you'd need this |

### Web UI

```bash
uvicorn server:app --reload
```

Then open `http://localhost:8000`. Upload a file, pick a model/device, and poll the job until it's done; download links for `.srt`/`.vtt` appear when it finishes.

### Docker

```bash
docker build -t sub-converter .
docker run -p 8000:8000 sub-converter
```

The image bundles `ffmpeg`; model weights still download to the container's cache on first use (mount a volume at `/root/.cache/huggingface` to persist them across runs).

## How it works

```
input file
   │
   ▼
[video?] ── extract_audio (ffmpeg) ──► 16 kHz mono float32 PCM
   │no                                        │
   └────────────────────────────────────────► │
                                               ▼
                                    optional VAD gate (Silero)
                                               │
                                               ▼
                                 log-mel spectrogram (80 x 3000)
                                               │
                                               ▼
                                   transformer encoder (per 30s window)
                                               │
                                               ▼
                        autoregressive decoder, prompted <|ko|><|translate|>
                           (English out directly — no Korean intermediate)
                                               │
                                               ▼
                          timestamp tokens (0.02s grid) + beam search
                                               │
                                               ▼
                      Segment(index, start, end, text) list, gaps stitched
                       via condition_on_previous_text across windows
                                               │
                                               ▼
                          write_srt() ──► .srt   write_vtt() ──► .vtt
```

1. **Decode + resample.** `extract_audio()` (in `transcriber.py`) shells out to `ffmpeg -vn -ar 16000 -ac 1` to produce mono 16 kHz PCM. This isn't an arbitrary choice: Whisper's encoder was trained exclusively on 16 kHz mono audio, so anything else must be resampled down to match before the model can use it. Audio files that are already in a supported container skip this step and go straight to the model.

2. **Optional VAD gate.** Before any transformer work happens, faster-whisper can run a Silero voice-activity-detection pass that finds speech regions and skips silence, saving compute and reducing hallucinated segments in quiet stretches. This tool passes `vad_filter=True` by default (`transcriber.transcribe`/`process_file` both take a `vad_filter` argument passed explicitly to `model.transcribe`, since the underlying library's own default has changed across releases and differs between its interfaces). **Caveat:** VAD is a speech classifier, and it can decide that heavily noise-gated or synthetic (TTS/formant-synthesis) audio isn't speech at all, silently discarding the whole file. `--no-vad` (CLI) or `vad_filter=False` (library) is the escape hatch.

   This is a **separate mechanism** from Whisper's own internal `no_speech_threshold`, which runs *after* VAD, per 30-second window, using the decoder's own no-speech-probability head to decide whether a window has any speech worth transcribing at all. Both mechanisms can independently and silently empty a subtitle file, for different reasons and at different pipeline stages — VAD as a pre-filter on the waveform, `no_speech_threshold` as a per-window decode-time gate. Conflating the two costs real debugging time: an earlier investigation into a 0-segment failure on synthetic speech initially (and incorrectly) blamed VAD, when disabling VAD alone did not fix it — the actual cause was `no_speech_threshold` classifying the audio as non-speech. This tool doesn't override `no_speech_threshold`; if you hit that failure mode, it's a faster-whisper `transcribe()` kwarg you can pass through `transcriber.transcribe`.

3. **Log-mel spectrogram.** The (possibly VAD-trimmed) waveform is chopped into 30-second windows and converted to an 80-bin log-mel spectrogram with a 10 ms hop, producing an 80x3000 matrix per window. This spectrogram, not the raw waveform, is what the model actually consumes.

4. **Transformer encoder.** Two convolutional layers followed by self-attention blocks turn each window's spectrogram into audio embeddings. This runs once per 30-second window.

5. **Autoregressive decoder.** The decoder is prompted with task tokens — `<|ko|>` selects the source language, `<|translate|>` selects the task — and then autoregressively emits English text **directly, in a single decoding pass**. There is no intermediate Korean transcript and no separate translation model or API call involved: Whisper was multitask-trained to translate any of ~100 languages into English end-to-end, and `<|translate|>` invokes that behavior directly. This is different from the transcribe-then-translate architecture most people assume (ASR model → separate MT model/API). The tradeoff is that you can't get both a Korean transcript and an English translation from one pass — swapping `<|translate|>` for `<|transcribe|>` gets you Korean text instead, and getting both means running the audio through twice.

6. **Timestamps and search.** Special timestamp tokens on a 0.02 s grid are interleaved with the text tokens to delimit where each utterance starts and ends. Beam search (`beam_size=5`) explores multiple decoding paths, with temperature fallback re-attempting a window at higher temperature if the beam search output looks degenerate (repetition, low confidence). Consecutive windows are stitched together via `condition_on_previous_text`, which feeds the previous window's text back in as context for the next.

7. **Output.** Non-empty decoder segments become `Segment(index, start, end, text)` tuples (empty-text segments are dropped so indices stay contiguous), which `write_srt()` renders as `NN\nHH:MM:SS,mmm --> HH:MM:SS,mmm\ntext\n\n` blocks and `write_vtt()` renders the same way under a `WEBVTT` header with `HH:MM:SS.mmm` timestamps.

## Choosing a model size

| Model | Params | Disk | Relative CPU speed | Use when |
|---|---|---|---|---|
| `tiny` | 39M | ~75 MB | fastest | Smoke tests / wiring checks only |
| `base` | 74M | ~145 MB | very fast | Smoke tests; quick iteration |
| `small` | 244M | ~480 MB | moderate | **Default** — the right balance of speed and quality for most CPU use |
| `medium` | 769M | ~1.5 GB | slow | Real work, especially with a GPU |
| `large-v3` | 1550M | ~3 GB | slowest | Best quality; wants a GPU |

`small` is the default for a reason: it's the point where translate-task quality stops being noticeably rough while still running at usable speed on a CPU. `tiny` and `base` are fine for confirming the pipeline works end to end, but expect visibly worse output on the `translate` task specifically — small encoders have a harder time with translation than with same-language transcription, so the quality gap between `tiny`/`base` and `small`+ is wider here than it would be for plain Korean-to-Korean transcription. `medium` and `large-v3` are worth the cost for anything you actually care about, provided you have a GPU. On CPU, runtime scales roughly with parameter count, so `large-v3` can easily run slower than real time (i.e., minutes of processing per minute of audio) — plan accordingly if you're CPU-only.

## Why faster-whisper / CTranslate2

This project uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a reimplementation of Whisper's inference on top of [CTranslate2](https://github.com/OpenNMT/CTranslate2) rather than PyTorch. Same model weights, same architecture, same accuracy — but roughly 4x faster and using substantially less memory, thanks to int8/float16 quantization, layer fusion, and a more efficient key-value cache in the transformer decoder.

Practical upshots:

- `compute_type="int8"` (the default this project picks for `device="cpu"` in `load_model()`) is what makes CPU-only translation usable at all in reasonable time.
- No PyTorch dependency is required at runtime — this repo doesn't list `torch` in `requirements.txt`. `load_model()`/`convert.py` do `try: import torch` purely to opportunistically auto-detect a CUDA GPU when `--device` isn't given explicitly, and fall back to CPU cleanly if `torch` isn't installed.
- Models are pulled from the `Systran/faster-whisper-*` Hugging Face repos, which ship weights already converted to CTranslate2's format — not the original OpenAI PyTorch checkpoints.

## Testing

```bash
python3 test_convert.py          # full run: loads a real model, exercises audio + video + server
python3 test_convert.py --fast   # tier 1 only: pure functions, no model, sub-second
```

No `pytest` — the test file is stdlib-only by design, since `requirements.txt` is intentionally minimal and pytest isn't part of it. Also accepts `--model`/`--device` (which model/device to use for the model-dependent tiers; default `tiny`/`cpu`), `--keep` (preserve and print the temp working dir instead of deleting it), and `--strict` (treat skips as failures, useful in CI to catch an environment that's silently missing a dependency).

What it checks: timestamp formatting and rounding at second/minute boundaries, negative-input clamping, SRT/VTT writer output structure, rejection of unsupported file extensions, and — on real audio — that `process_file()` produces well-formed, contiguously-indexed SRT and VTT files with sane, monotonically non-decreasing timestamps, matching cue counts and cue times between the two formats, and valid UTF-8 output. It repeats the structural checks on a muxed video file to exercise the `extract_audio`/ffmpeg path, and (only if `fastapi`+`httpx` are both installed, which they aren't in this project's default environment) drives the actual FastAPI job endpoints end to end.

What it deliberately does **not** check: the translated text content itself. Translation output is model- and version-dependent, so asserting on specific words or even on the text being ASCII would make the suite flaky for the wrong reasons — the tests only assert that each cue has *some* non-empty text.

**Test fixture:** `fixtures/ko_sample.flac` (12.4 s, 16 kHz mono, 16-bit FLAC) is synthetic Korean speech generated with [Piper](https://github.com/rhasspy/piper)'s `ko_KR-kss-medium` neural TTS voice ([rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices), path `ko/ko_KR/kss/medium`), trained on the [KSS Dataset](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset). The sample was trimmed to 12.4 s and resampled for this repo.

This voice model is licensed **CC BY-NC-SA 4.0** (https://creativecommons.org/licenses/by-nc-sa/4.0/) — **non-commercial use only**. The fixture audio derived from it inherits that restriction: don't redistribute or use it commercially.

## Limitations

- VAD (`vad_filter`) and Whisper's internal `no_speech_threshold` are two independent mechanisms, either of which can silently produce a 0-segment (empty) subtitle file on unusual audio — see [How it works](#how-it-works). If output is suspiciously empty, try `--no-vad` first; if that doesn't help, the issue is likely `no_speech_threshold` on the decode side.
- No speaker diarization — multi-speaker audio is transcribed as one continuous stream with no speaker labels.
- No subtitle line-length wrapping — long utterances become long single-line cues rather than being split for readability.
- Timestamps are utterance-level, not word-level, unless you pass `word_timestamps=True` through to `transcribe()` yourself (not exposed as a CLI/server flag).
- The final cue's end timestamp can overrun the actual audio duration — a known Whisper quirk on the last window of a file, observed directly on this project's own fixture (20.70 s of cue time on a 12.4 s file with VAD off).
- `server.py` keeps job state in an in-memory dict and is single-process — jobs don't survive a restart, and it won't horizontally scale without adding shared job storage.
