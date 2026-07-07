"""
recorder.py — PortAudio (sounddevice) capture, streaming WAV write.

Pipeline:
  mic → sounddevice.InputStream → sf.SoundFile (WAV, 16kHz mono)
                                       ↓
                              pushed to pipeline queue as WAV path
                                       ↓
                           transcriber.py reads WAV directly

No MP3 conversion, no pydub, no double-buffering. WhisperX resamples
to 16kHz mono internally anyway, so recording at CD quality (44100Hz)
only wastes 2.75× disk and RAM per recording.
"""

import asyncio
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Coroutine, Any

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)

SAMPLE_RATE  = 16000      # matches WhisperX internal rate — no resample overhead
CHANNELS     = 1          # mono — sufficient for speech
CHUNK_FRAMES = 1024       # ~64 ms per chunk at 16000 Hz
DTYPE        = "float32"  # sounddevice native; sf writes PCM_16


def list_input_devices() -> list[dict]:
    """
    Return all available input devices.
    Used by the /devices route. PortAudio backend — no PulseAudio needed.
    """
    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append({"id": i, "name": dev["name"]})
    return devices


class AudioRecorder:
    def __init__(self, output_dir: Path):
        self._output_dir   = output_dir
        self._recording    = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._start_time: float | None = None
        self._peak: float  = 0.0
        self._mic_name: str | None = None
        # Callbacks set by main.py
        self.on_error: Callable[[str], Coroutine[Any, Any, None]] | None = None
        self.on_complete: Callable[[str], Coroutine[Any, Any, None]] | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def set_output_dir(self, path: Path) -> None:
        self._output_dir = path

    def set_mic(self, name: str | None) -> None:
        self._mic_name = name

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None or not self._recording:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def peak(self) -> float:
        return self._peak

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._recording:
            logger.warning("Recorder already running — ignoring start()")
            return
        self._recording  = True
        self._loop       = loop
        self._start_time = time.monotonic()
        t = threading.Thread(target=self._record_thread, daemon=True)
        t.start()
        logger.info("Recording started")

    def stop(self) -> None:
        self._recording = False
        logger.info("Recording stop signal sent")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fail(self, msg: str) -> None:
        self._recording = False
        self._peak      = 0.0
        logger.error("[recorder] Fatal: %s", msg)
        if self._loop and not self._loop.is_closed() and self.on_error:
            asyncio.run_coroutine_threadsafe(self.on_error(msg), self._loop)

    def _done(self, wav_path: str) -> None:
        """Called when WAV is ready — hands path to main.py for processing."""
        if self._loop and not self._loop.is_closed() and self.on_complete:
            asyncio.run_coroutine_threadsafe(self.on_complete(wav_path), self._loop)

    def _resolve_device_index(self) -> int | None:
        if not self._mic_name:
            return None
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0 and dev["name"] == self._mic_name:
                return i
        logger.warning("Named mic %r not found — falling back to system default", self._mic_name)
        return None

    def _record_thread(self) -> None:
        uid     = uuid.uuid4().hex
        wav_path = self._output_dir / f"{uid}.wav"

        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._fail(f"Cannot create output directory '{self._output_dir}': {exc}")
            return

        try:
            device_index = self._resolve_device_index()
        except Exception as exc:
            self._fail(f"Cannot enumerate audio devices: {exc}")
            return

        logger.info(
            "Capturing from: %s",
            sd.query_devices(device_index)["name"] if device_index is not None
            else "system default"
        )

        try:
            with sf.SoundFile(
                str(wav_path), mode="w",
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                subtype="PCM_16",
            ) as wav_file:
                with sd.InputStream(
                    device=device_index,
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype=DTYPE,
                    blocksize=CHUNK_FRAMES,
                ) as stream:
                    while self._recording:
                        data, _ = stream.read(CHUNK_FRAMES)
                        wav_file.write(data)
                        # np.dot avoids allocating a squared array each chunk
                        self._peak = float(np.sqrt(np.dot(data.ravel(), data.ravel()) / data.size))

        except Exception as exc:
            self._fail(f"Audio capture error: {exc}")
            wav_path.unlink(missing_ok=True)
            return

        self._peak = 0.0
        logger.info("WAV ready: %s", wav_path)
        self._done(str(wav_path))
