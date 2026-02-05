#!/usr/bin/env python3
"""
WebSocket Server for Real-time ASR Streaming from YouTube.

This script:
1. Downloads audio from a YouTube URL
2. Streams audio chunks to Qwen3-ASR
3. Sends transcription results via WebSocket

Usage:
    # Start the server
    python youtube_websocket_streaming.py

    # Connect with wscat or browser:
    # ws://localhost:8765?url=YOUR_YOUTUBE_URL

Dependencies:
    pip install yt-dlp websockets soundfile numpy
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from typing import Optional

import numpy as np
import soundfile as sf
import websockets

from qwen_asr import Qwen3ASRModel


# Configuration
WEBSOCKET_PORT = 8765
CHUNK_MS = 500  # 500ms chunks for low latency
SAMPLE_RATE = 16000

# Global model (loaded once)
asr_model: Optional[Qwen3ASRModel] = None


def download_youtube_audio(url: str) -> str:
    """Download audio from YouTube URL and return path to WAV file."""
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "audio.wav")
    
    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", output_path.replace(".wav", ".%(ext)s"),
        "--postprocessor-args", "-ar 16000 -ac 1",  # 16kHz mono
        url
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Find the output file
    for f in os.listdir(temp_dir):
        if f.endswith(".wav"):
            return os.path.join(temp_dir, f)
    
    raise FileNotFoundError("Failed to download audio")


def load_and_prepare_audio(audio_path: str) -> np.ndarray:
    """Load audio file and prepare for streaming."""
    wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    
    # Convert stereo to mono
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        duration = wav.shape[0] / float(sr)
        n_samples = int(round(duration * SAMPLE_RATE))
        x_old = np.linspace(0.0, duration, num=wav.shape[0], endpoint=False)
        x_new = np.linspace(0.0, duration, num=n_samples, endpoint=False)
        wav = np.interp(x_new, x_old, wav)
    
    return wav.astype(np.float32)


async def stream_transcription(websocket, youtube_url: str):
    """Stream transcription results for a YouTube video."""
    global asr_model
    
    await websocket.send(json.dumps({
        "type": "status",
        "message": "Downloading YouTube audio..."
    }))
    
    try:
        # Download audio
        start_download = time.time()
        audio_path = download_youtube_audio(youtube_url)
        download_time = time.time() - start_download
        
        await websocket.send(json.dumps({
            "type": "status",
            "message": f"Download complete ({download_time:.1f}s). Starting transcription..."
        }))
        
        # Load audio
        wav = load_and_prepare_audio(audio_path)
        duration = wav.shape[0] / SAMPLE_RATE
        
        await websocket.send(json.dumps({
            "type": "info",
            "duration": duration,
            "chunk_ms": CHUNK_MS
        }))
        
        # Initialize streaming state
        state = asr_model.init_streaming_state(
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=2.0,
        )
        
        # Stream audio chunks
        chunk_samples = int(CHUNK_MS / 1000.0 * SAMPLE_RATE)
        pos = 0
        last_text = ""
        
        while pos < wav.shape[0]:
            chunk_start_time = time.time()
            
            # Get chunk
            chunk = wav[pos : pos + chunk_samples]
            pos += chunk.shape[0]
            current_time = pos / SAMPLE_RATE
            
            # Process chunk
            asr_model.streaming_transcribe(chunk, state)
            
            # Calculate latency
            latency_ms = (time.time() - chunk_start_time) * 1000
            
            # Send result if text changed
            if state.text != last_text:
                new_text = state.text[len(last_text):]
                last_text = state.text
                
                await websocket.send(json.dumps({
                    "type": "transcription",
                    "timestamp": round(current_time, 2),
                    "text": new_text,
                    "full_text": state.text,
                    "language": state.language,
                    "latency_ms": round(latency_ms, 1)
                }))
            
            # Small delay to simulate real-time (optional)
            # await asyncio.sleep(CHUNK_MS / 1000.0)
        
        # Finalize
        asr_model.finish_streaming_transcribe(state)
        
        await websocket.send(json.dumps({
            "type": "final",
            "language": state.language,
            "text": state.text
        }))
        
        # Cleanup
        os.remove(audio_path)
        os.rmdir(os.path.dirname(audio_path))
        
    except Exception as e:
        await websocket.send(json.dumps({
            "type": "error",
            "message": str(e)
        }))


async def handler(websocket):
    """Handle WebSocket connections."""
    from urllib.parse import parse_qs, urlparse
    
    # Get path from websocket request (new API)
    path = websocket.request.path if hasattr(websocket, 'request') else "/"
    parsed = urlparse(path)
    params = parse_qs(parsed.query)
    
    youtube_url = params.get("url", [None])[0]
    
    if not youtube_url:
        await websocket.send(json.dumps({
            "type": "error",
            "message": "Missing 'url' parameter. Use: ws://localhost:8765?url=YOUTUBE_URL"
        }))
        return
    
    await stream_transcription(websocket, youtube_url)


async def main():
    global asr_model
    
    print("=" * 60)
    print("WebSocket ASR Server (YouTube Streaming)")
    print("=" * 60)
    
    # Load model
    print("\nLoading Qwen3-ASR model...")
    asr_model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.5,
        max_new_tokens=64,
        enforce_eager=True,
        max_model_len=16384,
    )
    print("Model loaded!\n")
    
    # Start WebSocket server
    print(f"Starting WebSocket server on ws://localhost:{WEBSOCKET_PORT}")
    print(f"Connect with: ws://localhost:{WEBSOCKET_PORT}?url=YOUR_YOUTUBE_URL")
    print("-" * 60)
    
    async with websockets.serve(handler, "localhost", WEBSOCKET_PORT):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
