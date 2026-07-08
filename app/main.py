"""
main.py — FastAPI application

Single-job model:
  - Press Record → recording starts immediately
  - Press Stop   → processing starts immediately (transcribe → notes)
  - No queue, no job states beyond recording/processing/done/error
  - One job at a time, enforced by is_processing flag
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

from app.db import (
    init_db, create_job, update_job_status,
    save_transcript, get_transcript,
    save_notes, get_all_notes, get_notes,
    get_all_jobs,
)
from app.recorder import AudioRecorder
from app.transcriber import preload as transcriber_preload, run_transcription, diarization_enabled, load_diarization
from app.notes_gen import preload_ollama, generate_notes

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH    = Path(os.environ.get("CONFIG_PATH", "/app/app/config.json"))
DEFAULT_OUTPUT = Path(os.environ.get("OUTPUT_DIR", "/app/output"))

VALID_DEVICES       = {"cpu", "cuda", "mps"}
VALID_COMPUTE_TYPES = {"int8", "float16", "float32", "int8_float16"}
VALID_MODES         = {"auto", "manual"}
COMPUTE_DEFAULTS    = {"cpu": "int8", "cuda": "float16", "mps": "float32"}

_BASE_OUTPUT_ROOT = Path("/app/output").resolve()

_pending_save_task: asyncio.Task | None = None


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return {
        "output_path":   str(DEFAULT_OUTPUT),
        "device":        "cpu",
        "compute_type":  "int8",
        "whisperx_mode": "auto",
    }


async def _save_config(cfg: dict) -> None:
    await asyncio.sleep(0.02)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def _schedule_save(cfg: dict) -> None:
    global _pending_save_task
    if _pending_save_task and not _pending_save_task.done():
        _pending_save_task.cancel()
    _pending_save_task = asyncio.create_task(_save_config(cfg))


# ── App ───────────────────────────────────────────────────────────────────────

app       = FastAPI(title="Audio Notes")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

active_connections: list[WebSocket] = []
config: dict       = _load_config()
models_loaded: bool = False
is_processing: bool = False   # True while transcription/notes pipeline is running

recorder = AudioRecorder(output_dir=Path(config["output_path"]))


# ── WebSocket broadcast ───────────────────────────────────────────────────────

async def ws_broadcast(payload: dict[str, Any]) -> None:
    dead = []
    for ws in active_connections:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for d in dead:
        active_connections.remove(d)


# ── Processing pipeline — triggered directly on stop ─────────────────────────

async def _process(wav_path: str) -> None:
    """
    Called by recorder.on_complete when the user hits Stop.
    Runs transcription then note generation in sequence.
    No queue — this is the only job that can ever run.
    """
    global is_processing

    job_id     = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    create_job(job_id, created_at)
    is_processing = True

    try:
        cfg_snapshot = dict(config)

        await ws_broadcast({"event": "processing_started", "id": job_id})

        update_job_status(job_id, "transcribing")
        segments = await run_transcription(wav_path, ws_broadcast, cfg_snapshot)
        save_transcript(job_id, segments)
        update_job_status(job_id, "summarizing")
        await ws_broadcast({"event": "transcript_ready", "id": job_id})

        notes_md   = await generate_notes(job_id, segments, ws_broadcast)
        save_notes(job_id, notes_md)
        update_job_status(job_id, "done")
        await ws_broadcast({"event": "note_ready", "id": job_id})
        logger.info("[pipeline] Job %s complete", job_id)

    except Exception as exc:
        logger.exception("[pipeline] Job %s failed: %s", job_id, exc)
        update_job_status(job_id, "error")
        safe_msg = type(exc).__name__ + ": " + str(exc).split("\n")[0][:120]
        await ws_broadcast({"event": "error", "id": job_id, "detail": safe_msg})
    finally:
        is_processing = False
        # Clean up WAV after processing — no longer needed
        try:
            Path(wav_path).unlink(missing_ok=True)
        except Exception:
            pass


async def _recorder_error(msg: str) -> None:
    global is_processing
    is_processing = False
    logger.error("[recorder_error] %s", msg)
    await ws_broadcast({"event": "error", "detail": f"Recording failed: {msg}"})


async def _recorder_complete(wav_path: str) -> None:
    """Fires when recorder finishes writing the WAV — kick off processing."""
    asyncio.create_task(_process(wav_path))


recorder.on_error    = _recorder_error
recorder.on_complete = _recorder_complete


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    init_db()

    mode = config.get("whisperx_mode", "auto")
    if mode == "auto":
        logger.info("[startup] auto mode — preloading models in background")
        asyncio.create_task(_background_preload())
    else:
        logger.info("[startup] manual mode — models not loaded yet")


async def _background_preload() -> None:
    """Load WhisperX + Ollama in the background so uvicorn serves immediately."""
    global models_loaded
    loop = asyncio.get_running_loop()
    try:
        # No ws_broadcast(models_loading) here — no clients are connected at startup.
        # The WS 'init' event already sends models_loaded=False on connect.
        await loop.run_in_executor(None, lambda: (
            transcriber_preload(config),
            preload_ollama(),
        ))
        models_loaded = True
        await ws_broadcast({"event": "models_ready"})
        logger.info("[startup] Models ready")
    except Exception as exc:
        logger.exception("[startup] Background preload failed: %s", exc)
        await ws_broadcast({"event": "error", "detail": f"Model preload failed: {exc}"})


# ── Path helpers ──────────────────────────────────────────────────────────────

def _get_allowed_roots() -> list[Path]:
    roots: set[Path] = {_BASE_OUTPUT_ROOT}
    live = config.get("output_path", "")
    if live:
        try:
            roots.add(Path(live).resolve())
        except Exception:
            pass
    return list(roots)


def _validate_output_path(raw: str) -> Path:
    try:
        resolved = Path(raw).resolve()
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid path")
    allowed = _get_allowed_roots()
    for root in allowed:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise HTTPException(
        status_code=422,
        detail=f"Output path must be inside {', '.join(str(r) for r in allowed)}. Got: {resolved}",
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "output_path":    config.get("output_path", str(DEFAULT_OUTPUT)),
            "device":         config.get("device", "cpu"),
            "compute_type":   config.get("compute_type", "int8"),
            "whisperx_mode":  config.get("whisperx_mode", "auto"),
            "models_loaded":  models_loaded,
            "hf_token_set":   bool(config.get("hf_token", "").strip()),
            "diarization_ok": diarization_enabled(),
        },
    )


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/config")
async def get_config() -> JSONResponse:
    return JSONResponse({
        "output_path":    config.get("output_path"),
        "device":         config.get("device", "cpu"),
        "compute_type":   config.get("compute_type", "int8"),
        "whisperx_mode":  config.get("whisperx_mode", "auto"),
        "models_loaded":  models_loaded,
        "hf_token_set":   bool(config.get("hf_token", "").strip()),
        "diarization_ok": diarization_enabled(),
    })


@app.get("/devices")
async def list_devices() -> JSONResponse:
    try:
        from .recorder import list_input_devices
        devices = list_input_devices()
    except Exception as exc:
        logger.warning("[devices] Could not enumerate devices: %s", exc)
        devices = []
    return JSONResponse({"devices": devices})


@app.get("/status")
async def status() -> JSONResponse:
    return JSONResponse({
        "recording":     recorder.is_recording,
        "elapsed":       round(recorder.elapsed_seconds, 1),
        "peak":          round(recorder.peak, 4),
        "is_processing": is_processing,
        "models_loaded": models_loaded,
    })


@app.post("/set-hf-token")
async def set_hf_token(body: dict) -> JSONResponse:
    """
    Save HF token to config.json and immediately hot-load the diarization
    pipeline — no container restart needed.
    Token is stored in config.json (persists across restarts).
    Never echoed back in responses — only a boolean status is returned.
    """
    token = body.get("token", "").strip()
    if not token:
        raise HTTPException(status_code=422, detail="token is required")

    config["hf_token"] = token
    _schedule_save(config)

    # Hot-load diarization on the currently loaded device (cpu fallback)
    from app import transcriber as _t
    device = _t._loaded_device or config.get("device", "cpu")

    loop = asyncio.get_running_loop()
    ok = await loop.run_in_executor(None, load_diarization, token, device)

    if ok:
        await ws_broadcast({"event": "diarization_ready"})
        logger.info("[set-hf-token] Diarization pipeline loaded successfully")
        return JSONResponse({"status": "ok", "diarization": True})
    else:
        await ws_broadcast({"event": "hf_token_missing"})
        return JSONResponse(
            {"status": "error", "detail": "Token saved but diarization failed — check token permissions."},
            status_code=400,
        )


@app.post("/load-models")
async def load_models() -> JSONResponse:
    if models_loaded:
        return JSONResponse({"status": "already_loaded"})
    asyncio.create_task(_background_preload())
    return JSONResponse({"status": "loading"})


@app.post("/start-recording")
async def start_recording(body: dict) -> JSONResponse:
    if not models_loaded:
        raise HTTPException(status_code=409, detail="Models not loaded yet.")
    if recorder.is_recording:
        return JSONResponse({"status": "already_recording"}, status_code=409)
    if is_processing:
        raise HTTPException(status_code=409, detail="Processing in progress — wait for it to finish.")

    mic_name = body.get("mic") or None
    recorder.set_mic(mic_name)
    recorder.start(asyncio.get_running_loop())
    await ws_broadcast({"event": "recording_started", "mic": mic_name})
    return JSONResponse({"status": "recording"})


@app.post("/stop-recording")
async def stop_recording() -> JSONResponse:
    if not recorder.is_recording:
        return JSONResponse({"status": "not_recording"}, status_code=409)
    recorder.stop()
    await ws_broadcast({"event": "recording_stopped"})
    return JSONResponse({"status": "processing"})


@app.post("/set-output-path")
async def set_output_path(body: dict) -> JSONResponse:
    raw = body.get("path", "").strip()
    if not raw:
        raise HTTPException(status_code=422, detail="path is required")
    safe_path = _validate_output_path(raw)
    config["output_path"] = str(safe_path)
    _schedule_save(config)
    recorder.set_output_dir(safe_path)
    return JSONResponse({"status": "ok", "output_path": str(safe_path)})


@app.post("/set-device")
async def set_device(body: dict) -> JSONResponse:
    device = body.get("device", "").lower().strip()
    if device not in VALID_DEVICES:
        raise HTTPException(status_code=422, detail=f"device must be one of {sorted(VALID_DEVICES)}")
    compute_type = body.get("compute_type", "").strip() or COMPUTE_DEFAULTS[device]
    if compute_type not in VALID_COMPUTE_TYPES:
        raise HTTPException(status_code=422, detail=f"compute_type must be one of {sorted(VALID_COMPUTE_TYPES)}")
    config["device"]       = device
    config["compute_type"] = compute_type
    _schedule_save(config)
    await ws_broadcast({"event": "device_changed", "device": device, "compute_type": compute_type})
    return JSONResponse({"status": "ok", "device": device, "compute_type": compute_type})


@app.post("/set-mode")
async def set_mode(body: dict) -> JSONResponse:
    mode = body.get("mode", "").lower().strip()
    if mode not in VALID_MODES:
        raise HTTPException(status_code=422, detail="mode must be 'auto' or 'manual'")
    config["whisperx_mode"] = mode
    _schedule_save(config)
    return JSONResponse({"status": "ok", "whisperx_mode": mode, "note": "Restart container to apply."})


@app.get("/notes")
async def notes_list() -> JSONResponse:
    return JSONResponse({"notes": get_all_notes()})


@app.get("/notes/{job_id}")
async def note_detail(job_id: str) -> JSONResponse:
    note = get_notes(job_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return JSONResponse(note)


@app.get("/history")
async def history() -> JSONResponse:
    return JSONResponse({"jobs": get_all_jobs()})


@app.get("/transcript/{job_id}")
async def transcript_detail(job_id: str) -> JSONResponse:
    segments = get_transcript(job_id)
    if segments is None:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return JSONResponse({"segments": segments})


@app.get("/export/notes/{job_id}")
async def export_notes(job_id: str):
    """Download notes for a job as a .md file. Generated on-demand from DB."""
    note = get_notes(job_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return PlainTextResponse(
        content=note["content"],
        headers={"Content-Disposition": f'attachment; filename="notes_{job_id[:8]}.md"'},
        media_type="text/markdown",
    )


@app.get("/export/transcript/{job_id}")
async def export_transcript(job_id: str):
    """Download transcript for a job as a plain-text file. Generated on-demand from DB."""
    segments = get_transcript(job_id)
    if segments is None:
        raise HTTPException(status_code=404, detail="Transcript not found")
    lines = []
    for seg in segments:
        start = int(seg.get("start", 0))
        m, s  = divmod(start, 60)
        lines.append(f"[{m:02d}:{s:02d}] {seg.get('speaker','UNKNOWN')}: {seg.get('text','').strip()}")
    return PlainTextResponse(
        content="\n".join(lines),
        headers={"Content-Disposition": f'attachment; filename="transcript_{job_id[:8]}.txt"'},
        media_type="text/plain",
    )


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    active_connections.append(ws)
    await ws.send_json({
        "event":          "init",
        "device":         config.get("device", "cpu"),
        "compute_type":   config.get("compute_type", "int8"),
        "output_path":    config.get("output_path", str(DEFAULT_OUTPUT)),
        "whisperx_mode":  config.get("whisperx_mode", "auto"),
        "models_loaded":  models_loaded,
        "hf_token_set":   bool(config.get("hf_token", "").strip()),
        "diarization_ok": diarization_enabled(),
    })
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in active_connections:
            active_connections.remove(ws)
