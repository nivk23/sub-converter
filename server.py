import os
import re
import time
import uuid
import threading
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from transcriber import process_file, load_model, MODEL_SIZES

app = FastAPI(title="Korean Subtitle Converter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

jobs: dict = {}

DEVICES = ("cpu", "cuda")

JOB_TTL_SECONDS = 60 * 60  # 1 hour

# Cache of loaded WhisperModel instances keyed by (model_size, device), so
# concurrent/successive jobs don't each pay the model-load cost (seconds of
# load time and hundreds of MB of memory per instance).
_model_cache: dict = {}
_model_cache_lock = threading.Lock()


def get_model(model_size: str, device: str):
    key = (model_size, device)
    with _model_cache_lock:
        model = _model_cache.get(key)
        if model is None:
            model = load_model(model_size, device=device)
            _model_cache[key] = model
        return model


def sweep_expired_jobs() -> None:
    """Drop jobs older than JOB_TTL_SECONDS and remove their work dirs."""
    now = time.time()
    expired = [
        job_id
        for job_id, job in jobs.items()
        if now - job.get("created_at", now) > JOB_TTL_SECONDS
    ]
    for job_id in expired:
        job = jobs.pop(job_id, None)
        if job and job.get("work_dir"):
            shutil.rmtree(job["work_dir"], ignore_errors=True)


def run_transcription(
    job_id: str,
    input_path: str,
    stem: str,
    work_dir: str,
    model_size: str,
    device: str,
    vad_filter: bool,
):
    job = jobs[job_id]
    job["status"] = "processing"
    job["progress"] = 0

    def progress_cb(done: int, total: Optional[int], fraction: Optional[float]) -> None:
        if fraction is not None:
            job["progress"] = int(fraction * 100)

    try:
        model = get_model(model_size, device)
        # process_file returns (srt_path, vtt_path) as Path objects
        srt_path, vtt_path = process_file(
            input_path=input_path,
            model=model,
            model_size=model_size,
            device=device,
            output_dir=work_dir,
            progress_cb=progress_cb,
            vad_filter=vad_filter,
        )
        job["status"] = "done"
        job["progress"] = 100
        job["srt_path"] = str(srt_path)
        job["vtt_path"] = str(vtt_path)
        job["stem"] = stem
        job["srt_url"] = f"/jobs/{job_id}/download/srt"
        job["vtt_url"] = f"/jobs/{job_id}/download/vtt"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)


@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return RedirectResponse(url="/static/index.html")


@app.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    model: str = Form("small"),
    device: str = Form("cpu"),
    vad: str = Form("true"),
):
    if model not in MODEL_SIZES:
        raise HTTPException(
            status_code=400,
            detail=f"model must be one of {sorted(MODEL_SIZES)}",
        )
    if device not in DEVICES:
        raise HTTPException(
            status_code=400,
            detail=f"device must be one of {sorted(DEVICES)}",
        )

    vad_filter = vad.lower() not in ("false", "0", "no")

    sweep_expired_jobs()

    job_id = str(uuid.uuid4())
    work_dir = tempfile.mkdtemp(prefix=f"sub_{job_id}_")

    original_name = file.filename or "upload"
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix or ".tmp"
    # Save with the original stem so process_file produces {stem}.srt / {stem}.vtt
    input_path = os.path.join(work_dir, f"{stem}{suffix}")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "error": None,
        "srt_url": None,
        "vtt_url": None,
        "srt_path": None,
        "vtt_path": None,
        "stem": stem,
        "work_dir": work_dir,
        "created_at": time.time(),
    }

    t = threading.Thread(
        target=run_transcription,
        args=(job_id, input_path, stem, work_dir, model, device, vad_filter),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "error": job["error"],
        "srt_url": job["srt_url"],
        "vtt_url": job["vtt_url"],
    }


@app.get("/jobs/{job_id}/download/{fmt}")
async def download_subtitle(job_id: str, fmt: str):
    if fmt not in ("srt", "vtt"):
        raise HTTPException(status_code=400, detail="fmt must be 'srt' or 'vtt'")

    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Job not complete")

    path_key = f"{fmt}_path"
    file_path = job.get(path_key)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Subtitle file not found")

    raw_stem = job.get("stem", "subtitles")
    safe_stem = re.sub(r"[^A-Za-z0-9._-]", "", raw_stem)[:100] or "subtitles"
    filename = f"{safe_stem}.{fmt}"

    return FileResponse(
        path=file_path,
        media_type="text/plain",
        filename=filename,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
