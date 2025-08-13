import os
import json
import re
from typing import Optional, List, Dict

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

import sherpa_onnx
import ctranslate2
from transformers import AutoTokenizer

# ---------------- STT (Parakeet) config ----------------
SAMPLE_RATE = 16000
FEATURE_DIM = 80
CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", str(os.cpu_count() or 4)))
PARAKEET_DIR = os.getenv(
    "PARAKEET_DIR",
    os.path.join(os.getcwd(), "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"),
)
DECODE_METHOD = os.getenv("PARAKEET_DECODING", "greedy_search")
# Keep chunks short for live translation
MAX_UTTERANCE_SEC = float(os.getenv("STT_MAX_UTTERANCE_SEC", "5.0"))

# ---------------- Filler filtering ----------------
ENABLE_FILLER_FILTER = os.getenv("ENABLE_FILLER_FILTER", "1") != "0"
FILLER_MAX_DUR_SEC = float(os.getenv("FILLER_MAX_DUR_SEC", "1.2"))
MIN_FINAL_CHARS = int(os.getenv("MIN_FINAL_CHARS", "8"))
FILLER_SET = {
    "uh","um","erm","mm","mmm","hmm","hmmm","uhh","uh-huh","mm-hmm",
    "ah","oh","huh","yeah","yep","nope","ok","okay","right","mmkay"
}

# ---------------- MT (M2M100 via CTranslate2) ----------------
# ISO 639-1 codes, comma-separated (e.g., "fr,es,de,lt,ko")
TARGET_LANGS = [s.strip() for s in os.getenv("TARGET_LANGS", "fr,es,de").split(",") if s.strip()]
MT_DIR = os.getenv("MT_MODEL_DIR", os.path.join(os.getcwd(), "m2m100_418_ct2"))  # CTranslate2 model dir
MT_BEAM_SIZE = int(os.getenv("MT_BEAM_SIZE", "4"))

def _abspath(p: str) -> str:
    return os.path.abspath(p) if not os.path.isabs(p) else p

PARAKEET_DIR = _abspath(PARAKEET_DIR)
MT_DIR = _abspath(MT_DIR)

ENC = os.path.join(PARAKEET_DIR, "encoder.int8.onnx")
DEC = os.path.join(PARAKEET_DIR, "decoder.int8.onnx")
JOIN = os.path.join(PARAKEET_DIR, "joiner.int8.onnx")
TOK = os.path.join(PARAKEET_DIR, "tokens.txt")

# ---------------- Utils ----------------
def _int16_bytes_to_float32_numpy(buf: bytes) -> np.ndarray:
    if not buf:
        return np.zeros(0, dtype=np.float32)
    audio_i16 = np.frombuffer(buf, dtype=np.int16)
    return (audio_i16.astype(np.float32) / 32768.0)

_word_only = re.compile(r"[^a-z]+")
def _norm_text(s: str) -> str:
    return _word_only.sub(" ", s.lower()).strip()

def _should_suppress(text: str, dur_sec: float) -> bool:
    if not ENABLE_FILLER_FILTER:
        return False
    t = (text or "").strip()
    if not t:
        return True
    if len(t) < MIN_FINAL_CHARS and dur_sec <= FILLER_MAX_DUR_SEC:
        toks = _norm_text(t).split()
        if not toks:
            return True
        if all(tok in FILLER_SET or len(tok) <= 2 for tok in toks):
            return True
    n = _norm_text(t)
    if n in FILLER_SET and dur_sec <= FILLER_MAX_DUR_SEC * 1.5:
        return True
    return False

# ---------------- Init STT ----------------
def create_recognizer():
    print(f"[server-mt] Loading Parakeet from {PARAKEET_DIR} threads={CPU_THREADS}")
    print(f"[server-mt] Filler filter: {'ON' if ENABLE_FILLER_FILTER else 'OFF'}")
    # New sherpa-onnx config API
    try:
        feat_cfg = sherpa_onnx.FeatureExtractorConfig(sampling_rate=SAMPLE_RATE, feature_dim=FEATURE_DIM)
        model_cfg = sherpa_onnx.OfflineModelConfig(
            transducer=sherpa_onnx.OfflineTransducerModelConfig(
                encoder_filename=ENC, decoder_filename=DEC, joiner_filename=JOIN
            ),
            tokens=TOK,
            num_threads=CPU_THREADS,
            provider="cpu",
            model_type="nemo_transducer",
        )
        recog_cfg = sherpa_onnx.OfflineRecognizerConfig(
            feat_config=feat_cfg,
            model_config=model_cfg,
            decoding_method=DECODE_METHOD,
        )
        return sherpa_onnx.OfflineRecognizer(recog_cfg)
    except TypeError:
        # Older factory API
        if hasattr(sherpa_onnx.OfflineRecognizer, "from_transducer"):
            return sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=ENC, decoder=DEC, joiner=JOIN, tokens=TOK,
                num_threads=CPU_THREADS, provider="cpu",
                decoding_method=DECODE_METHOD,
                model_type="nemo_transducer", sample_rate=SAMPLE_RATE, feature_dim=FEATURE_DIM,
            )
        raise

recognizer = create_recognizer()

# ---------------- Init MT ----------------
if not os.path.isdir(MT_DIR):
    raise RuntimeError(f"MT model directory not found: {MT_DIR}\n"
                       f"Set MT_MODEL_DIR or convert M2M100 to CTranslate2 (see README steps).")

print(f"[server-mt] Loading MT model from {MT_DIR} (beam_size={MT_BEAM_SIZE})")
translator = ctranslate2.Translator(
    MT_DIR, device="cpu",
    inter_threads=CPU_THREADS, intra_threads=CPU_THREADS
)
tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
tokenizer.src_lang = "en"  # ASR source language

def translate_all(text: str, targets: List[str]) -> Dict[str, str]:
    # Encode with src_lang so the input contains the proper >>en<< token
    enc = tokenizer(text, return_tensors="pt")  # add_special_tokens=True (default)
    src_tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

    # Build per-target language prefixes as token *strings*
    prefixes = []
    tlangs = []
    for t in targets:
        try:
            lang_id = tokenizer.get_lang_id(t)  # e.g., 'fr' -> id for >>fr<<
        except KeyError:
            continue
        lang_tok = tokenizer.convert_ids_to_tokens([lang_id])  # ['>>fr<<']
        prefixes.append(lang_tok)
        tlangs.append(t)

    if not prefixes:
        return {}

    # Repeat the same source tokens per target and translate
    batch = [src_tokens for _ in prefixes]
    results = translator.translate_batch(batch, target_prefix=prefixes, beam_size=MT_BEAM_SIZE)

    out: Dict[str, str] = {}
    for t, res in zip(tlangs, results):
        hyp = res.hypotheses[0] if res.hypotheses else []
        # First generated token is the target language tag; drop it
        toks = hyp[1:] if hyp else []
        out[t] = tokenizer.decode(tokenizer.convert_tokens_to_ids(toks), skip_special_tokens=True)
    return out


# ---------------- App ----------------
app = FastAPI()

@app.get("/", response_class=PlainTextResponse)
def root():
    langs = ",".join(TARGET_LANGS) or "(none)"
    return f"Live STT+MT is up. Targets: {langs}. Connect a WebSocket client to /ws"

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("[server-mt] Client connected.")

    utt_pcm = bytearray()
    seconds_accum = 0.0

    try:
        while True:
            message = await ws.receive()

            if message.get("bytes") is not None:
                frame = message["bytes"]
                utt_pcm += frame
                seconds_accum = len(utt_pcm) / 2 / SAMPLE_RATE

                # Time cap (finalize even without silence)
                if MAX_UTTERANCE_SEC > 0 and seconds_accum >= MAX_UTTERANCE_SEC:
                    audio = _int16_bytes_to_float32_numpy(bytes(utt_pcm))
                    stream = recognizer.create_stream()
                    stream.accept_waveform(SAMPLE_RATE, audio)
                    recognizer.decode_stream(stream)
                    text = (stream.result.text or "").strip()

                    if not _should_suppress(text, seconds_accum):
                        await ws.send_json({"type": "final", "text": text})
                        if TARGET_LANGS:
                            mts = translate_all(text, TARGET_LANGS)
                            for lang, mt in mts.items():
                                await ws.send_json({"type": "mt", "lang": lang, "text": mt, "src": text})

                    utt_pcm.clear()
                    seconds_accum = 0.0

            elif message.get("text") is not None:
                try:
                    evt = json.loads(message["text"])
                except Exception:
                    evt = {}

                if evt.get("type") == "config":
                    await ws.send_json({"type": "ack", "targets": TARGET_LANGS})

                if evt.get("type") == "segment_end":
                    audio = _int16_bytes_to_float32_numpy(bytes(utt_pcm))
                    stream = recognizer.create_stream()
                    stream.accept_waveform(SAMPLE_RATE, audio)
                    recognizer.decode_stream(stream)
                    text = (stream.result.text or "").strip()

                    if not _should_suppress(text, seconds_accum):
                        await ws.send_json({"type": "final", "text": text})
                        if TARGET_LANGS:
                            mts = translate_all(text, TARGET_LANGS)
                            for lang, mt in mts.items():
                                await ws.send_json({"type": "mt", "lang": lang, "text": mt, "src": text})

                    utt_pcm.clear()
                    seconds_accum = 0.0

    except WebSocketDisconnect:
        print("[server-mt] Client disconnected.")
    except Exception as e:
        print(f"[server-mt] Error: {e}")
        try:
            await ws.close()
        except Exception:
            pass
