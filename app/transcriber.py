"""
transcriber.py — WhisperX 3-step pipeline with model singleton.

Changes:
- whisperx model is loaded ONCE at startup (via preload()) and cached.
  Subsequent jobs reuse the cached model — no 10-30s reload per job.
- Language is hardcoded to English ("en") as requested.
- Device resolved from live config at call time.
- Diarization pipeline also cached after first load.
"""

import asyncio
import logging
import os
from typing import Callable, Coroutine, Any

import whisperx
import whisperx.diarize  # explicit submodule import — DiarizationPipeline moved here in v3.3.4+

logger = logging.getLogger(__name__)

HF_TOKEN         = os.environ.get("HF_TOKEN", "")
LANGUAGE         = "en"
BATCH_SIZE       = 4   # halved from 8 — cuts peak RAM ~50% with no quality impact
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"  # pinned — community-1 is separately gated

_COMPUTE_DEFAULTS: dict[str, str] = {
    "cpu":  "int8",
    "cuda": "float16",
    "mps":  "float32",
}

# ── Singletons — populated by preload() at startup ────────────────────────────
_whisper_model    = None
_align_model      = None
_align_metadata   = None
_diarize_model    = None
_loaded_device    = None
# _loaded_compute not tracked: device mismatch warning uses _loaded_device only


def _resolve_device(config: dict) -> tuple[str, str]:
    device = config.get("device", "cpu").lower()
    if device not in _COMPUTE_DEFAULTS:
        logger.warning("[transcriber] Unknown device %r — falling back to cpu", device)
        device = "cpu"
    compute_type = config.get("compute_type") or _COMPUTE_DEFAULTS[device]
    return device, compute_type


def preload(config: dict) -> None:
    """
    Load WhisperX model, alignment model, and diarization pipeline at startup.
    Called once from FastAPI startup event. Blocks until all models are ready.
    This replaces per-job loading entirely.
    """
    global _whisper_model, _align_model, _align_metadata
    global _diarize_model, _loaded_device

    device, compute_type = _resolve_device(config)
    logger.info("[transcriber] Preloading WhisperX on device=%s compute_type=%s", device, compute_type)

    _whisper_model = whisperx.load_model(
        "small", device, compute_type=compute_type, language=LANGUAGE
    )  # language pre-set avoids per-file detection overhead
    logger.info("[transcriber] Whisper model ready")

    _align_model, _align_metadata = whisperx.load_align_model(
        language_code=LANGUAGE, device=device
    )
    logger.info("[transcriber] Alignment model ready")

    if HF_TOKEN:
        _diarize_model = whisperx.diarize.DiarizationPipeline(
            model_name=DIARIZATION_MODEL, token=HF_TOKEN, device=device
        )
        logger.info("[transcriber] Diarization pipeline ready (%s)", DIARIZATION_MODEL)
    else:
        logger.warning("[transcriber] HF_TOKEN not set — diarization disabled")

    _loaded_device = device
    logger.info("[transcriber] All models preloaded")


def _build_segments(result: dict) -> list[dict]:
    segments = []
    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "UNKNOWN")
        start   = round(seg.get("start", 0.0), 2)
        end     = round(seg.get("end",   0.0), 2)
        text    = seg.get("text", "").strip()
        if text:
            segments.append({"start": start, "end": end, "speaker": speaker, "text": text})
    return segments


async def run_transcription(
    audio_path: str,
    ws_broadcast: Callable[[dict], Coroutine[Any, Any, None]],
    config: dict,
) -> list[dict]:
    """
    Run transcription using preloaded models.
    Accepts any audio file whisperx.load_audio() supports (WAV, MP3, etc.).
    Falls back to loading on-demand if preload() was never called (dev mode).
    """
    # No global declarations needed: run_transcription only reads these values.
    loop = asyncio.get_running_loop()  # fix #9
    device, _ = _resolve_device(config)  # compute_type unused here; device used for mismatch check

    # Warn if device changed since preload — models are locked to original device
    if _loaded_device and _loaded_device != device:
        logger.warning(
            "[transcriber] Device changed to %s but models are loaded on %s — "
            "using loaded models. Restart container to apply new device.",
            device, _loaded_device,
        )
        await ws_broadcast({
            "event": "warn",
            "message": (
                f"WhisperX is loaded on {_loaded_device.upper()} — "
                f"restart the container to switch to {device.upper()}."
            ),
        })

    # On-demand load if preload was skipped.
    # Run in an executor so the blocking model load doesn't freeze the event loop.
    if _whisper_model is None:
        logger.warning("[transcriber] Models not preloaded — loading now (will be slow)")
        await loop.run_in_executor(None, preload, config)

    await ws_broadcast({"status": "transcribing", "device": _loaded_device or device})

    def _pipeline() -> list[dict]:
        audio  = whisperx.load_audio(audio_path)
        result = _whisper_model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)
        logger.info("[transcriber] Transcription done")

        result = whisperx.align(
            result["segments"], _align_model, _align_metadata,
            audio, _loaded_device or device,
            return_char_alignments=False,
        )
        logger.info("[transcriber] Alignment done")

        if _diarize_model is not None:
            diarize_segments = _diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            logger.info("[transcriber] Diarization done")

        return _build_segments(result)

    segments = await loop.run_in_executor(None, _pipeline)
    logger.info("[transcriber] Pipeline complete — %d segments", len(segments))
    return segments
