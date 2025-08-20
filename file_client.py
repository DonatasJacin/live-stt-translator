import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import websocket

# Ensure UTF-8 output (for Hangul etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320
BYTES_PER_SAMPLE = 2
FRAME_BYTES = FRAME_SAMPLES * BYTES_PER_SAMPLE  # 640

def dbfs_from_int16(frame_bytes: bytes) -> float:
    if not frame_bytes:
        return -120.0
    x = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return -120.0
    rms = np.sqrt(np.mean((x / 32768.0) ** 2) + 1e-12)
    return 20.0 * math.log10(rms + 1e-12)

def stream_file(ws_url: str, file_path: str, targets, silence_ms: int, max_segment_ms: int, vad_db: float):
    # Open WebSocket
    ws = websocket.WebSocket()
    ws.connect(ws_url, ping_interval=20)

    # Tell server which languages to translate to
    cfg = {"type": "config", "language": "en", "targets": targets}
    ws.send(json.dumps(cfg))

    # Use ffmpeg to decode to 16k mono s16le PCM to stdout
    # Requires ffmpeg executable in PATH (conda-forge ffmpeg works)
    cmd = [
        "ffmpeg",
        "-i", file_path,
        "-vn",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-hide_banner",
        "-loglevel", "error",
        "pipe:1",
    ]
    print(f"[file] Decoding with: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    # Endpointing state
    silence_frames_needed = max(1, silence_ms // FRAME_MS)
    force_cut_enabled = max_segment_ms is not None and max_segment_ms > 0
    voiced = False
    silence_count = 0
    voiced_ms = 0

    # Non-blocking receive loop params
    ws.settimeout(0.001)

    def flush_segment():
        # Ask the server to decode whatever we sent so far
        try:
            ws.send(json.dumps({"type": "segment_end"}))
        except websocket._exceptions.WebSocketConnectionClosedException:
            print("\n[file] Server closed connection.")
            return False
        return True

    # Main read-send loop
    total_frames = 0
    try:
        while True:
            frame = proc.stdout.read(FRAME_BYTES)
            if not frame:
                # EOF: flush any remainder
                if voiced or silence_count > 0:
                    flush_segment()
                break

            total_frames += 1
            level = dbfs_from_int16(frame)

            if level > vad_db:
                try:
                    ws.send(frame, opcode=websocket.ABNF.OPCODE_BINARY)
                except websocket._exceptions.WebSocketTimeoutException:
                    print("\n[file] WS send timed out (server likely closed).")
                    return
                except websocket._exceptions.WebSocketConnectionClosedException:
                    print("\n[file] WS connection closed by server.")
                    return

                if not voiced:
                    voiced = True
                    voiced_ms = 0
                else:
                    voiced_ms += FRAME_MS
                silence_count = 0

                if force_cut_enabled and voiced_ms >= max_segment_ms:
                    if not flush_segment():
                        break
                    voiced_ms = 0  # continue new segment without dropping frames

            else:
                # silence
                if voiced:
                    silence_count += 1
                    if silence_count >= silence_frames_needed:
                        if not flush_segment():
                            break
                        voiced = False
                        silence_count = 0
                        voiced_ms = 0

            # Poll for server messages
            try:
                msg = ws.recv()
                if msg:
                    try:
                        data = json.loads(msg)
                        t = data.get("type")
                        if t == "final":
                            print(f"[final] {data.get('text','')}")
                        elif t == "mt":
                            print(f"[mt:{data.get('lang','?')}] {data.get('text','')}")
                        elif t == "ack":
                            targets = data.get("targets", targets)
                            unsupported = data.get("unsupported", [])
                            if unsupported:
                                print(f"[ack] targets set: {targets} | unsupported: {unsupported}")
                            else:
                                print(f"[ack] targets set: {targets}")
                    except json.JSONDecodeError:
                        pass
            except websocket._exceptions.WebSocketTimeoutException:
                pass

        # Drain remaining messages for a short period
        t0 = time.time()
        while time.time() - t0 < 2.0:
            try:
                msg = ws.recv()
                if not msg:
                    break
                try:
                    data = json.loads(msg)
                    t = data.get("type")
                    if t == "final":
                        print(f"[final] {data.get('text','')}")
                    elif t == "mt":
                        print(f"[mt:{data.get('lang','?')}] {data.get('text','')}")
                except json.JSONDecodeError:
                    pass
            except websocket._exceptions.WebSocketTimeoutException:
                time.sleep(0.01)
            except websocket._exceptions.WebSocketConnectionClosedException:
                break

    finally:
        try:
            ws.close()
        except Exception:
            pass
        if proc and proc.stdout:
            proc.stdout.close()
        if proc:
            proc.terminate()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stream an audio file to the STT+MT WebSocket server.")
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws", help="WebSocket URL (use wss://<ngrok-domain>/ws for ngrok)")
    p.add_argument("--file", required=True, help="Path to input audio (mp3/wav/etc.; ffmpeg must support it)")
    p.add_argument("--targets", default="lt,ko", help="Comma-separated target languages (e.g., lt,ko,fr)")
    p.add_argument("--silence_ms", type=int, default=400, help="Silence duration to end a segment")
    p.add_argument("--max_segment_ms", type=int, default=4000, help="Force a segment end after this many ms of continuous speech (0 disables)")
    p.add_argument("--vad_db", type=float, default=-45.0, help="Energy VAD threshold in dBFS (more negative = more sensitive)")
    args = p.parse_args()

    targets = [s.strip() for s in args.targets.split(",") if s.strip()]
    if args.max_segment_ms <= 0:
        args.max_segment_ms = None

    stream_file(args.url, args.file, targets, args.silence_ms, args.max_segment_ms, args.vad_db)
