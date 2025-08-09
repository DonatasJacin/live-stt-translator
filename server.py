
import os
import asyncio
import json
import re
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from faster_whisper import WhisperModel

# ---- Configuration ----
SAMPLE_RATE = 16000
CHUNK_SEC = float(os.getenv("STT_CHUNK_SEC", "0.8"))
OVERLAP_SEC = float(os.getenv("STT_OVERLAP_SEC", "0.25"))
MAX_BUFFER_SEC = float(os.getenv("STT_MAX_BUFFER_SEC", "12.0"))
MAX_UTTERANCE_CHARS = int(os.getenv("STT_MAX_UTTERANCE_CHARS", "1200"))

MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")     # e.g., small, medium, large-v3
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", str(os.cpu_count() or 4)))

# Optional decode knobs
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
INITIAL_PROMPT = os.getenv("WHISPER_INITIAL_PROMPT", None)

# ---- Load model once ----
print(f"[server] Loading Whisper model '{MODEL_NAME}' with compute_type={COMPUTE_TYPE}, cpu_threads={CPU_THREADS} ...")
asr_model = WhisperModel(
    MODEL_NAME,
    compute_type=COMPUTE_TYPE,
    cpu_threads=CPU_THREADS,
)
print("[server] Model loaded.")

app = FastAPI()

@app.get("/", response_class=PlainTextResponse)
def root():
    return "Live STT server is up. Connect a WebSocket client to /ws"

def _int16_bytes_to_float32_numpy(buf: bytes) -> np.ndarray:
    """Convert raw little-endian int16 PCM to float32 numpy in [-1, 1]."""
    if not buf:
        return np.zeros(0, dtype=np.float32)
    audio_i16 = np.frombuffer(buf, dtype=np.int16)
    return (audio_i16.astype(np.float32) / 32768.0)

def _longest_common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

_word_re = re.compile(r"[A-Za-z0-9]+")

def _last_word(s: str) -> str:
    m = None
    for m in _word_re.finditer(s):
        pass
    return m.group(0) if m else ""

def _first_word(s: str) -> str:
    m = _word_re.search(s)
    return m.group(0) if m else ""

def _clean_join(agg: str, delta: str) -> str:
    """Join agg + delta with sane spacing and no immediate word duplication.
    - Insert a space if both sides are alnum without boundary space.
    - Drop duplicated first word in delta if it's equal to agg's last word (case-insensitive).
    - Fix 'andand' / 'thereThere' no-space duplicates.
    """
    if not delta:
        return agg

    # If delta is identical to the last chunk (nothing new), return as-is
    if not agg:
        joined = delta.lstrip()
    else:
        # Ensure a space boundary if needed
        need_space = agg and agg[-1].isalnum() and delta[0].isalnum()
        joined = (agg + (" " if need_space else "") + delta)

    # Fix immediate no-space duplicates at the join (e.g., "andand")
    joined = re.sub(r"(\b\w{2,})(\1\b)", r"\1", joined, flags=re.IGNORECASE)

    # Drop duplicated first word of delta if it equals last word of agg
    lw = _last_word(agg).lower()
    fw = _first_word(delta).lower()
    if lw and fw and lw == fw:
        # remove first occurrence of that word in the just-appended region
        # We only strip at the boundary: find the boundary index
        boundary = len(agg)
        # If a space was inserted, boundary += 1
        if boundary < len(joined) and joined[boundary] == " ":
            boundary += 1
        # Remove the duplicate word starting at boundary
        after = joined[boundary:]
        # If the next token starts with that word (case-insensitive), strip it
        if after.lower().startswith(fw):
            after = after[len(fw):]
            # Also remove one following space if present
            if after.startswith(" "):
                after = after[1:]
            joined = joined[:boundary] + after

    return joined

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("[server] Client connected.")

    raw_buffer = bytearray()
    last_text: str = ""       # last window's full text
    agg_text: str = ""        # utterance-wide running transcript
    lang_hint: Optional[str] = None

    chunk_samples = int(CHUNK_SEC * SAMPLE_RATE)
    overlap_samples = int(OVERLAP_SEC * SAMPLE_RATE)
    max_buffer_bytes = int(MAX_BUFFER_SEC * SAMPLE_RATE * 2)

    try:
        while True:
            message = await ws.receive()

            if message.get("bytes") is not None:
                frame = message["bytes"]
                raw_buffer += frame

                if len(raw_buffer) > max_buffer_bytes:
                    raw_buffer = raw_buffer[-max_buffer_bytes:]

                if len(raw_buffer) >= 2 * chunk_samples:
                    audio = _int16_bytes_to_float32_numpy(raw_buffer)
                    segments, _ = asr_model.transcribe(
                        audio,
                        beam_size=BEAM_SIZE,
                        vad_filter=True,
                        language=lang_hint,           # None => auto-detect
                        condition_on_previous_text=True,
                        temperature=0.0,
                    )
                    window_text = "".join(s.text for s in segments).strip()

                    # Compute the newly added suffix vs previous window
                    i = _longest_common_prefix(window_text, last_text)
                    delta = window_text[i:]
                    if delta:
                        agg_text = _clean_join(agg_text, delta)
                        last_text = window_text
                        try:
                            await ws.send_json({"type": "partial", "delta": delta, "text": window_text, "agg": agg_text})
                        except WebSocketDisconnect:
                            break

                    # Keep small overlap so words at chunk boundary survive
                    if overlap_samples > 0 and len(raw_buffer) >= overlap_samples * 2:
                        raw_buffer = raw_buffer[-overlap_samples*2:]
                    else:
                        raw_buffer.clear()

                    # Auto-finalize if utterance is getting huge (safety)
                    if len(agg_text) >= MAX_UTTERANCE_CHARS:
                        try:
                            await ws.send_json({"type": "final", "text": agg_text})
                        except WebSocketDisconnect:
                            break
                        agg_text = ""
                        last_text = ""

            elif message.get("text") is not None:
                try:
                    evt = json.loads(message["text"])
                except Exception:
                    evt = {}

                if evt.get("type") == "config":
                    lang_hint = evt.get("language")
                    await ws.send_json({"type": "ack", "language": lang_hint})

                if evt.get("type") == "segment_end":
                    # Emit the entire utterance we've built so far
                    try:
                        await ws.send_json({"type": "final", "text": agg_text})
                    except WebSocketDisconnect:
                        break
                    raw_buffer.clear()
                    last_text = ""
                    agg_text = ""

    except WebSocketDisconnect:
        print("[server] Client disconnected.")
    except Exception as e:
        print(f"[server] Error: {e}")
        try:
            await ws.close()
        except Exception:
            pass
