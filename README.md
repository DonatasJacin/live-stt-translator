
# Live STT Starter (Python)

A minimal, production-leaning starter to do **live speech → text** with:
- **Client**: Microphone capture, **WebRTC VAD** gating, streams voiced frames via WebSocket.
- **Server**: **FastAPI** WebSocket endpoint using **faster-whisper** for fast chunked ASR, returns partial + final transcripts.