
import os
import json
import re
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

import sherpa_onnx

# ---- Configuration ----
SAMPLE_RATE = 16000
FEATURE_DIM = 80
CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", str(os.cpu_count() or 4)))  # reuse same env knob
PARAKEET_DIR = os.getenv("PARAKEET_DIR", os.path.join(os.getcwd(), "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"))
DECODE_METHOD = os.getenv("PARAKEET_DECODING", "greedy_search")  # or "modified_beam_search"
MAX_UTTERANCE_SEC = float(os.getenv("STT_MAX_UTTERANCE_SEC", "30.0"))

# Filler filtering knobs
ENABLE_FILLER_FILTER = os.getenv("ENABLE_FILLER_FILTER", "1") != "0"
FILLER_MAX_DUR_SEC = float(os.getenv("FILLER_MAX_DUR_SEC", "1.2"))   # only suppress if short
MIN_FINAL_CHARS = int(os.getenv("MIN_FINAL_CHARS", "8"))             # too-short finals are likely noise
FILLER_SET = {
    "uh","um","erm","mm","mmm","hmm","hmmm","uhh","uh-huh","mm-hmm",
    "ah","oh","huh","yeah","yep","nope","ok","okay","right","mmkay","hmmm",
    "hmmmmm","mmmmm","hmmhmm","hmm-mm","hmm-um","hmm-uh","hmmhmmhmm"
}

def _abspath(p: str) -> str:
    return os.path.abspath(p) if not os.path.isabs(p) else p

PARAKEET_DIR = _abspath(PARAKEET_DIR)

ENC = os.path.join(PARAKEET_DIR, "encoder.int8.onnx")
DEC = os.path.join(PARAKEET_DIR, "decoder.int8.onnx")
JOIN = os.path.join(PARAKEET_DIR, "joiner.int8.onnx")
TOK = os.path.join(PARAKEET_DIR, "tokens.txt")

def _int16_bytes_to_float32_numpy(buf: bytes) -> np.ndarray:
    if not buf:
        return np.zeros(0, dtype=np.float32)
    audio_i16 = np.frombuffer(buf, dtype=np.int16)
    return (audio_i16.astype(np.float32) / 32768.0)

# --- Filler suppression ---
_word_only = re.compile(r"[^a-z]+")

def _norm_text(s: str) -> str:
    return _word_only.sub(" ", s.lower()).strip()

def _should_suppress(text: str, dur_sec: float) -> bool:
    """Return True if this final should be dropped as filler/noise."""
    if not ENABLE_FILLER_FILTER:
        return False
    t = text.strip()
    if not t:
        return True
    # Very short raw length + short duration: likely noise
    if len(t) < MIN_FINAL_CHARS and dur_sec <= FILLER_MAX_DUR_SEC:
        # Allow simple 'hi', 'no' if you want; for now, treat as suppressible.
        n = _norm_text(t)
        toks = n.split()
        if not toks:
            return True
        # If all tokens are fillers or <=2 letters, drop
        if all(tok in FILLER_SET or len(tok) <= 2 for tok in toks):
            return True
        # Single-token cases like "okay", "yeah", "huh", "mm"
        if len(toks) <= 2 and all(tok in FILLER_SET for tok in toks):
            return True
    # Pure filler regardless of duration (rare), e.g., "mm", "uh-huh"
    n = _norm_text(t)
    if n in FILLER_SET and dur_sec <= FILLER_MAX_DUR_SEC * 1.5:
        return True
    return False

# ---- Load model once (support old and new sherpa-onnx APIs) ----
def create_recognizer():
    version = getattr(sherpa_onnx, "__version__", "unknown")
    print(f"[server-parakeet] sherpa-onnx version: {version}")
    print(f"[server-parakeet] Loading Parakeet TDT 0.6B V2 from {PARAKEET_DIR} with threads={CPU_THREADS} ...")
    print(f"[server-parakeet] Filler filter: {'ON' if ENABLE_FILLER_FILTER else 'OFF'} "
          f"(MIN_FINAL_CHARS={MIN_FINAL_CHARS}, FILLER_MAX_DUR_SEC={FILLER_MAX_DUR_SEC})")

    # Newer API path: config objects
    try:
        FeatureExtractorConfig = getattr(sherpa_onnx, "FeatureExtractorConfig", None)
        OfflineModelConfig = getattr(sherpa_onnx, "OfflineModelConfig", None)
        OfflineTransducerModelConfig = getattr(sherpa_onnx, "OfflineTransducerModelConfig", None)
        OfflineRecognizerConfig = getattr(sherpa_onnx, "OfflineRecognizerConfig", None)

        if FeatureExtractorConfig and OfflineModelConfig and OfflineTransducerModelConfig and OfflineRecognizerConfig:
            feat_cfg = FeatureExtractorConfig(sampling_rate=SAMPLE_RATE, feature_dim=FEATURE_DIM)
            model_cfg = OfflineModelConfig(
                transducer=OfflineTransducerModelConfig(
                    encoder_filename=ENC,
                    decoder_filename=DEC,
                    joiner_filename=JOIN,
                ),
                tokens=TOK,
                num_threads=CPU_THREADS,
                provider="cpu",
                model_type="nemo_transducer",
            )
            recog_cfg = OfflineRecognizerConfig(
                feat_config=feat_cfg,
                model_config=model_cfg,
                decoding_method=DECODE_METHOD,
            )
            rec = sherpa_onnx.OfflineRecognizer(recog_cfg)
            print("[server-parakeet] API mode: config-based")
            return rec

    except TypeError as e:
        print(f"[server-parakeet] Config API not available ({e}); trying factory constructor.")

    # Older API path: classmethod factory
    if hasattr(sherpa_onnx.OfflineRecognizer, "from_transducer"):
        # Try with explicit model_type/sample_rate/feature_dim first
        try:
            rec = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=ENC,
                decoder=DEC,
                joiner=JOIN,
                tokens=TOK,
                num_threads=CPU_THREADS,
                provider="cpu",
                decoding_method=DECODE_METHOD,
                model_type="nemo_transducer",
                sample_rate=SAMPLE_RATE,
                feature_dim=FEATURE_DIM,
            )
            print("[server-parakeet] API mode: factory-based (from_transducer, with nemo_transducer)")
            return rec
        except TypeError as e:
            # Fall back to minimal signature
            print(f"[server-parakeet] from_transducer with model_type failed ({e}); trying minimal signature.")
            rec = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=ENC,
                decoder=DEC,
                joiner=JOIN,
                tokens=TOK,
                num_threads=CPU_THREADS,
                provider="cpu",
                decoding_method=DECODE_METHOD,
            )
            print("[server-parakeet] API mode: factory-based (minimal)")
            return rec

    # If both paths failed:
    raise RuntimeError("Your installed sherpa-onnx does not provide a supported OfflineRecognizer API. "
                       "Please upgrade: pip install --upgrade --no-cache-dir sherpa-onnx")

recognizer = create_recognizer()
print("[server-parakeet] Model loaded.")

app = FastAPI()

@app.get("/", response_class=PlainTextResponse)
def root():
    return "Live STT (Parakeet) server is up. Connect a WebSocket client to /ws"

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("[server-parakeet] Client connected.")

    utt_pcm = bytearray()     # full utterance buffer (int16 LE)
    seconds_accum = 0.0       # track utterance length
    lang_hint: Optional[str] = None  # ignored, Parakeet is English-only

    try:
        while True:
            message = await ws.receive()

            if message.get("bytes") is not None:
                frame = message["bytes"]
                utt_pcm += frame
                seconds_accum = len(utt_pcm) / 2 / SAMPLE_RATE

                # Optional: auto-finalize if too long
                if MAX_UTTERANCE_SEC > 0 and seconds_accum >= MAX_UTTERANCE_SEC:
                    audio = _int16_bytes_to_float32_numpy(bytes(utt_pcm))
                    stream = recognizer.create_stream()
                    stream.accept_waveform(SAMPLE_RATE, audio)
                    recognizer.decode_stream(stream)
                    text = (stream.result.text or "").strip()

                    # Apply filler suppression
                    if not _should_suppress(text, seconds_accum):
                        await ws.send_json({"type": "final", "text": text})

                    # reset
                    utt_pcm.clear()
                    seconds_accum = 0.0

            elif message.get("text") is not None:
                try:
                    evt = json.loads(message["text"])
                except Exception:
                    evt = {}

                if evt.get("type") == "config":
                    lang_hint = evt.get("language")
                    await ws.send_json({"type": "ack", "language": lang_hint})

                if evt.get("type") == "segment_end":
                    # Decode the full utterance we have so far
                    audio = _int16_bytes_to_float32_numpy(bytes(utt_pcm))
                    stream = recognizer.create_stream()
                    stream.accept_waveform(SAMPLE_RATE, audio)
                    recognizer.decode_stream(stream)
                    text = (stream.result.text or "").strip()

                    # Apply filler suppression
                    if not _should_suppress(text, seconds_accum):
                        await ws.send_json({"type": "final", "text": text})

                    # reset buffers
                    utt_pcm.clear()
                    seconds_accum = 0.0

    except WebSocketDisconnect:
        print("[server-parakeet] Client disconnected.")
    except Exception as e:
        print(f"[server-parakeet] Error: {e}")
        try:
            await ws.close()
        except Exception:
            pass
