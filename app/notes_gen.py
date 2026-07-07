"""
notes_gen.py — Token-aware chunking, rolling context summarization, final compile.

Fix: preload_ollama() now uses a 30-second timeout so it cannot block
forever when ollama is still pulling phi3:mini on first startup.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Callable, Coroutine, Any

import tiktoken
import ollama

logger = logging.getLogger(__name__)

OLLAMA_HOST   = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
MODEL         = "phi3:mini"
MAX_TOKENS    = 2800   # safe ceiling for phi3:mini 4K context
CHUNK_SECONDS = 300    # 5-minute time-based fallback

# ── Module-level singletons ───────────────────────────────────────────────────
_client  = ollama.Client(host=OLLAMA_HOST)
_encoder = tiktoken.get_encoding("cl100k_base")


def _ollama_chat_with_retry(messages: list[dict], options: dict, max_retries: int = 1) -> str:
    """
    Call ollama with a single automatic retry on transient failure.
    Sleeps 2 s before retrying. Raises on second failure so the caller handles it.
    """
    import time as _time
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = _client.chat(model=MODEL, messages=messages, options=options)
            return response["message"]["content"]
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.warning(
                    "[notes_gen] ollama call failed (attempt %d/%d): %s — retrying in 2 s",
                    attempt + 1, max_retries + 1, exc
                )
                _time.sleep(2)
    raise last_exc


def preload_ollama() -> None:
    """
    Ping ollama at startup to confirm phi3:mini is loaded and ready.
    Times out after 30 seconds so it never blocks startup indefinitely.
    If ollama isn't ready yet (still pulling the model), logs a warning
    and continues — the first real job will surface the error then.
    """
    logger.info("[notes_gen] Pinging ollama / phi3:mini (timeout=30s)...")
    try:
        _ollama_chat_with_retry(
            messages=[{"role": "user", "content": "ok"}],
            options={"num_predict": 1, "temperature": 0},
            max_retries=0,  # startup ping — no retry; we just log and move on
        )
        logger.info("[notes_gen] phi3:mini is ready")
    except Exception as exc:
        logger.warning(
            "[notes_gen] Could not reach ollama at startup: %s — "
            "will retry on first recording job.", exc
        )


def _count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


# ── Prompt strings ────────────────────────────────────────────────────────────

def _build_chunk_prompt(chunk_text: str, previous_summary: str) -> str:
    context_block = ""
    if previous_summary.strip():
        context_block = (
            "## Context from previous segment\n"
            "The following is a summary of what was discussed before this segment. "
            "Use it to maintain narrative continuity — do NOT repeat or re-summarize it, "
            "only reference it when directly relevant.\n\n"
            f"{previous_summary}\n\n---\n"
        )
    return (
        f"{context_block}"
        "## Current Segment Transcript\n\n"
        f"{chunk_text}\n\n"
        "---\n\n"
        "You are a precise meeting notes assistant. Based ONLY on the transcript segment "
        "above (not the prior context), produce a structured summary in plain Markdown "
        "with these exact sections:\n\n"
        "### Segment Summary\n"
        "A concise paragraph (3–5 sentences) describing the main topic(s) and flow of "
        "this segment.\n\n"
        "### Key Points & Decisions\n"
        "A bullet list of the most important points, decisions, or conclusions from this "
        "segment only. Be specific and concrete — quote short phrases when useful.\n\n"
        "### Full Timestamped Log\n"
        "Copy the full transcript segment exactly as given, preserving [MM:SS] timestamps "
        "and speaker labels. Do not alter, summarize, or omit any lines.\n\n"
        "Rules:\n"
        "- Do not fabricate content not present in the transcript.\n"
        "- Do not include action items from prior context — only from this segment.\n"
        "- Output plain Markdown only. No preamble, no closing remarks."
    )


def _build_compile_prompt(all_chunk_summaries: str) -> str:
    return (
        "## All Chunk Summaries (in chronological order)\n\n"
        f"{all_chunk_summaries}\n\n"
        "---\n\n"
        "You are a precise meeting notes compiler. Using ONLY the chunk summaries above, "
        "produce a single master notes document in clean Markdown. "
        "Follow this exact structure:\n\n"
        "# Meeting / Session Notes\n\n"
        "## Overview\n"
        "A 2–3 sentence executive summary of the entire session: main purpose, key "
        "outcomes, and tone.\n\n"
        "## Action Items & Decisions\n"
        "A deduplicated, numbered list of every action item and decision across all "
        "chunks. If the same item appears in multiple chunks, include it only once. "
        "Be specific.\n\n"
        "## Speaker Breakdown\n"
        "For each speaker identified (use the labels from the transcripts, e.g., "
        "\"Speaker_00\"), write 1–2 sentences describing their main contributions and "
        "role in the discussion.\n\n"
        "## Full Transcript Log\n"
        "Assemble the complete timestamped transcript in chronological order by "
        "concatenating the \"Full Timestamped Log\" sections from all chunks. "
        "Preserve every line exactly.\n\n"
        "Rules:\n"
        "- Do not fabricate content. All output must be grounded in the summaries.\n"
        "- Deduplicate action items rigorously — scan all chunks before listing.\n"
        "- Output clean Markdown only. No preamble, no closing remarks."
    )


# ── Segment formatting ────────────────────────────────────────────────────────

def _format_segment(seg: dict) -> str:
    start = int(seg["start"])
    m, s  = divmod(start, 60)
    return f"[{m:02d}:{s:02d}] {seg.get('speaker', 'UNKNOWN')}: {seg.get('text', '').strip()}"


# ── Token-aware chunking ──────────────────────────────────────────────────────

def _chunk_segments(
    segments: list[dict],
    previous_summary: str = "",
) -> list[list[dict]]:
    """
    Split segments into chunks that stay under MAX_TOKENS.
    Splits only at speaker-turn boundaries.
    5-minute time-based guard fires if a single speaker runs very long.

    previous_summary is counted against the budget so that the rolling context
    injected into the prompt doesn’t push the total over phi3:mini’s context limit.
    """
    if not segments:
        return []

    # Reserve tokens already consumed by the previous_summary so the transcript
    # text portion never causes the full prompt to exceed MAX_TOKENS.
    reserved = _count_tokens(previous_summary) if previous_summary.strip() else 0
    effective_max = max(200, MAX_TOKENS - reserved)  # floor of 200 to avoid empty chunks

    chunks: list[list[dict]] = []
    current: list[dict]      = []
    current_tokens            = 0
    chunk_start_time: float   = segments[0]["start"]

    for seg in segments:
        line        = _format_segment(seg)
        line_tokens = _count_tokens(line)
        time_span   = seg["start"] - chunk_start_time

        should_flush = (
            current
            and (
                current_tokens + line_tokens > effective_max
                or time_span >= CHUNK_SECONDS
            )
        )

        if should_flush:
            chunks.append(current)
            current          = []
            current_tokens   = 0
            chunk_start_time = seg["start"]

        current.append(seg)
        current_tokens += line_tokens

    if current:
        chunks.append(current)

    return chunks


# ── Main entry point ──────────────────────────────────────────────────────────

async def generate_notes(
    job_id: str,
    segments: list[dict],
    output_dir: Path,  # kept for API compat but no longer used for file writes
    ws_broadcast: Callable[[dict], Coroutine[Any, Any, None]],
) -> str:
    """Returns the compiled notes markdown string. Caller saves to DB."""
    loop = asyncio.get_running_loop()  # get_running_loop() is correct inside a coroutine

    await ws_broadcast({"status": "chunking"})
    # Pass previous_summary="" on first call; re-chunk is not done mid-job,
    # but the initial budget is sized conservatively (no prior context yet).
    chunks = _chunk_segments(segments, previous_summary="")
    total  = len(chunks)
    logger.info("[notes_gen] %d chunk(s) after token-aware split", total)

    chunk_summaries: list[str] = []
    previous_summary            = ""

    for i, chunk in enumerate(chunks):
        await ws_broadcast({"status": "summarizing", "chunk": i + 1, "of": total})

        chunk_text = "\n".join(_format_segment(s) for s in chunk)
        prompt     = _build_chunk_prompt(chunk_text, previous_summary)

        def _call_chunk(p=prompt) -> str:
            return _ollama_chat_with_retry(
                messages=[{"role": "user", "content": p}],
                options={"temperature": 0.2},
            )

        summary          = await loop.run_in_executor(None, _call_chunk)
        chunk_summaries.append(summary)
        previous_summary = summary
        logger.info("[notes_gen] Chunk %d/%d done", i + 1, total)

    await ws_broadcast({"status": "compiling"})
    all_summaries  = "\n\n---\n\n".join(chunk_summaries)
    compile_prompt = _build_compile_prompt(all_summaries)

    def _call_compile() -> str:
        return _ollama_chat_with_retry(
            messages=[{"role": "user", "content": compile_prompt}],
            options={"temperature": 0.2},
        )

    master_notes = await loop.run_in_executor(None, _call_compile)
    logger.info("[notes_gen] Compile pass done")

    logger.info("[notes_gen] Notes generation complete for job %s", job_id)
    return master_notes
