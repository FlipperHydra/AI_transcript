"""
transcriber.py — WhisperX 3-step pipeline with model singleton.

- whisperx model is loaded ONCE at startup (via preload()) and cached.
  Subsequent jobs reuse the cached model — no 10-30s reload per job.
- Language is hardcoded to English ("en").
- Device resolved from live config at call time.
- Diarization pipeline cached after first load; can be hot-swapped via
  load_diarization(token, device) without restarting the container.
- HF token is NOT read from the environment — it comes from config.json
  and is set by the user through the web UI.
"""

import asyncio
import logging
import threading
from typing import Callable, Coroutine, Any

import whisperx
import whisperx.diarize  # explicit submodule import — DiarizationPipeline moved here in v3.3.4+

logger = logging.getLogger(__name__)

LANGUAGE          = "en"
BATCH_SIZE        = 4   # halved from 8 — cuts peak RAM ~50% with no quality impact
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
_diarize_lock     = threading.Lock()  # prevents concurrent diarization loads


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
    HF token is read from config dict (set via UI, saved to config.json).
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

    _loaded_device = device

    # Attempt diarization load if token present in config
    hf_token = config.get("hf_token", "").strip()
    if hf_token:
        load_diarization(hf_token, device)
    else:
        logger.warning(
            "[transcriber] No HF token in config — diarization disabled. "
            "Set token via the web UI to enable speaker labels."
        )

    logger.info("[transcriber] All models preloaded")


def load_diarization(token: str, device: str = "cpu") -> bool:
    """
    Load (or reload) the diarization pipeline with the given token.
    Thread-safe — can be called from the /set-hf-token route at any time.
    Returns True on success, False on failure (logs the error).
    """
    global _diarize_model
    with _diarize_lock:
        try:
            logger.info("[transcriber] Loading diarization pipeline (%s)...", DIARIZATION_MODEL)
            _diarize_model = whisperx.diarize.DiarizationPipeline(
                model_name=DIARIZATION_MODEL, token=token, device=device
            )
            logger.info("[transcriber] Diarization pipeline ready")
            return True
        except Exception as exc:
            logger.error("[transcriber] Diarization load failed: %s", exc)
            _diarize_model = None
            return False


def diarization_enabled() -> bool:
    """Returns True if the diarization pipeline is currently loaded."""
    return _diarize_model is not None


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
    Falls back to loading on-demand if preload() was never called (manual mode).
    """
    loop = asyncio.get_running_loop()
    device, _ = _resolve_device(config)

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

    # On-demand load if preload was skipped (manual mode).
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
        else:
            logger.info("[transcriber] Diarization skipped — no token set")

        return _build_segments(result)

    segments = await loop.run_in_executor(None, _pipeline)
    logger.info("[transcriber] Pipeline complete — %d segments", len(segments))
    return segments
