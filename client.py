
import argparse
import json
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import webrtcvad
import websocket

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
BYTES_PER_SAMPLE = 2

def audio_capture(q: queue.Queue, device: int | None):
    def callback(indata, frames, time_info, status):
        if status:
            print(f"[client] Audio status: {status}", file=sys.stderr)
        q.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        dtype='int16',
        channels=1,
        callback=callback,
        device=device
    ):
        while True:
            time.sleep(0.1)

def run(ws_url: str, device: int | None, vad_level: int, silence_ms: int, language: str | None, max_segment_ms: int):
    q = queue.Queue()
    cap_thread = threading.Thread(target=audio_capture, args=(q, device), daemon=True)
    cap_thread.start()

    vad = webrtcvad.Vad(vad_level)
    ws = websocket.WebSocket()
    ws.connect(ws_url, ping_interval=20)

    if language:
        ws.send(json.dumps({"type": "config", "language": language}))

    voiced = False
    silence_frames_needed = max(1, silence_ms // FRAME_MS)
    silence_count = 0

    # Track how long we've been continuously voiced; force a cut when exceeded.
    voiced_ms = 0
    force_cut_enabled = max_segment_ms is not None and max_segment_ms > 0

    current_line = ""

    print("[client] Connected. Speak into your mic. Press Ctrl+C to stop.")
    print(f"[client] Endpointing: silence_ms={silence_ms}, max_segment_ms={max_segment_ms if force_cut_enabled else 'disabled'}")
    try:
        while True:
            frame = q.get()
            if len(frame) != FRAME_SAMPLES * BYTES_PER_SAMPLE:
                continue

            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if is_speech:
                ws.send(frame, opcode=websocket.ABNF.OPCODE_BINARY)
                if not voiced:
                    voiced = True
                    voiced_ms = 0  # start timing on speech onset
                else:
                    voiced_ms += FRAME_MS

                # Time-based forced segmentation (no silence needed)
                if force_cut_enabled and voiced_ms >= max_segment_ms:
                    try:
                        ws.send(json.dumps({"type": "segment_end"}))
                        # Immediately start a new segment without dropping frames
                        voiced_ms = 0
                        # Do NOT flip 'voiced' to False; we are still speaking
                    except websocket._exceptions.WebSocketConnectionClosedException:
                        print("\n[client] Server closed connection.")
                        break
                # reset silence counter while speaking
                silence_count = 0

            else:
                if voiced:
                    silence_count += 1
                    if silence_count >= silence_frames_needed:
                        try:
                            ws.send(json.dumps({"type": "segment_end"}))
                        except websocket._exceptions.WebSocketConnectionClosedException:
                            print("\n[client] Server closed connection.")
                            break
                        voiced = False
                        silence_count = 0
                        voiced_ms = 0

            # Non-blocking receive of server messages
            ws.settimeout(0.001)
            try:
                msg = ws.recv()
                if msg:
                    try:
                        data = json.loads(msg)
                        if data.get("type") == "partial":
                            current_line = data.get("agg") or data.get("text","")
                            print(f"\r[partial] {current_line[:120]:<120}", end="", flush=True)
                        elif data.get("type") == "final":
                            final_text = data.get("text","")
                            print(f"\n[final] {final_text}")
                            current_line = ""
                    except json.JSONDecodeError:
                        pass
            except websocket._exceptions.WebSocketTimeoutException:
                pass
            except websocket._exceptions.WebSocketConnectionClosedException:
                print("\n[client] Server closed connection.")
                break

    except KeyboardInterrupt:
        print("\n[client] Stopping.")
    finally:
        try:
            ws.close()
        except Exception:
            pass

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws", help="WebSocket URL")
    p.add_argument("--device", type=int, default=None, help="sounddevice input device index (optional)")
    p.add_argument("--vad", type=int, default=2, choices=[0,1,2,3], help="WebRTC VAD aggressiveness (0-3)")
    p.add_argument("--silence_ms", type=int, default=300, help="Silence duration to end a segment")
    p.add_argument("--language", type=str, default=None, help="Optional language hint like 'en', 'es', etc.")
    p.add_argument("--max_segment_ms", type=int, default=3500, help="Force a segment end after this many ms of continuous speech (0 disables)")
    args = p.parse_args()
    run(args.url, args.device, args.vad, args.silence_ms, args.language, args.max_segment_ms)
