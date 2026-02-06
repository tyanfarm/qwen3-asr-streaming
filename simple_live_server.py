#!/usr/bin/env python3
"""
Simplified Live Screen Audio Server - Uses raw PCM audio for low latency.

Features:
- Backpressure: drops old chunks when processing can't keep up
- Rate limiting: prevents VRAM overflow from chunk accumulation
- Silero VAD: detects speech to avoid processing silence

Usage:
    python simple_live_server.py
"""

import asyncio
import json
import time
from collections import deque
from typing import Optional

import numpy as np
import torch
import websockets

from qwen_asr import Qwen3ASRModel


WEBSOCKET_PORT = 8765
SAMPLE_RATE = 16000
MAX_QUEUE_SIZE = 40  # Max chunks to buffer before dropping

# VAD settings
VAD_THRESHOLD = 0.5  # Speech probability threshold (0-1)
MIN_SPEECH_DURATION_MS = 250  # Minimum speech duration to process

asr_model: Optional[Qwen3ASRModel] = None
vad_model = None


class TranscriptionSession:
    RESET_INTERVAL_SEC = 30  # Reset streaming state every 30 seconds
    
    def __init__(self, asr_model: Qwen3ASRModel):
        self.asr_model = asr_model
        self.state = asr_model.init_streaming_state(
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=1.0,
        )
        self.last_text = ""
        self.start_time = time.time()
        self.state_start_time = time.time()  # Track when state was last reset
        self.chunk_count = 0
        
        # VAD audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        self.min_buffer_size = int(SAMPLE_RATE * MIN_SPEECH_DURATION_MS / 1000)
    
    def _is_speech_at_edge(self, samples: np.ndarray) -> bool:
        """Check if the chunk ends with speech (mid-word).
        
        Only checks the last 512 samples (32ms) - single VAD call for speed.
        Returns True if speech is ongoing at the end of the chunk.
        """
        global vad_model
        if vad_model is None:
            return False  # If VAD not loaded, don't buffer
        
        if len(samples) < 512:
            return False
        
        try:
            # Only check the LAST 512 samples
            window = samples[-512:]
            audio_tensor = torch.from_numpy(window).float()
            speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()
            return speech_prob > VAD_THRESHOLD
        except Exception as e:
            print(f"VAD error: {e}")
            return False
    
    def process_pcm_chunk(self, pcm_bytes: bytes) -> dict:
        """Process raw PCM audio chunk with edge-aware VAD."""
        chunk_start = time.time()
        
        try:
            # Reset state periodically to prevent token overflow
            if time.time() - self.state_start_time > self.RESET_INTERVAL_SEC:
                print(f"🔄 Resetting streaming state (after {self.RESET_INTERVAL_SEC}s)")
                self.asr_model.finish_streaming_transcribe(self.state)
                self.state = self.asr_model.init_streaming_state(
                    unfixed_chunk_num=2,
                    unfixed_token_num=5,
                    chunk_size_sec=1.0,
                )
                self.state_start_time = time.time()
                self.last_text = ""
            
            # Convert bytes to int16 array
            int16_data = np.frombuffer(pcm_bytes, dtype=np.int16)
            
            # Convert to float32 [-1, 1]
            samples = int16_data.astype(np.float32) / 32768.0
            
            # Skip if too short
            if len(samples) < 1600:  # Less than 100ms @ 16kHz
                return None
            
            # Add samples to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, samples])
            
            # Edge-aware VAD: Check if we're mid-speech at the end
            speech_at_end = self._is_speech_at_edge(samples)
            
            # Process conditions:
            # 1. Speech ends (silence at edge) - process at word boundary
            # 2. Buffer too large (2 seconds) - force process
            should_process = (
                (not speech_at_end and len(self.audio_buffer) >= self.min_buffer_size) or
                (len(self.audio_buffer) >= SAMPLE_RATE * 2)  # Max 2 seconds
            )
            
            if should_process:
                # Process the buffered audio
                self.asr_model.streaming_transcribe(self.audio_buffer, self.state)
                self.audio_buffer = np.array([], dtype=np.float32)
            
            # Calculate metrics
            latency_ms = (time.time() - chunk_start) * 1000
            current_time = time.time() - self.start_time
            self.chunk_count += 1
            
            # Check for new text
            if self.state.text != self.last_text:
                new_text = self.state.text[len(self.last_text):]
                self.last_text = self.state.text
                
                return {
                    "type": "transcription",
                    "timestamp": round(current_time, 2),
                    "text": new_text,
                    "full_text": self.state.text,
                    "language": self.state.language,
                    "latency_ms": round(latency_ms, 1),
                    "chunk_count": self.chunk_count
                }
            
            return None
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def finalize(self) -> dict:
        self.asr_model.finish_streaming_transcribe(self.state)
        return {
            "type": "final",
            "language": self.state.language,
            "text": self.state.text,
            "total_chunks": self.chunk_count,
            "duration": round(time.time() - self.start_time, 2)
        }


async def handler(websocket):
    global asr_model
    
    print(f"Client connected: {websocket.remote_address}")
    session = TranscriptionSession(asr_model)
    
    # Queue for incoming audio chunks (with max size for backpressure)
    chunk_queue = deque(maxlen=MAX_QUEUE_SIZE)
    is_processing = False
    dropped_chunks = 0
    
    async def process_queue():
        """Process chunks from queue one at a time."""
        nonlocal is_processing, dropped_chunks
        
        while chunk_queue:
            is_processing = True
            pcm_bytes = chunk_queue.popleft()
            
            # Process in executor to not block event loop
            result = await asyncio.get_event_loop().run_in_executor(
                None, session.process_pcm_chunk, pcm_bytes
            )
            
            if result:
                # Add dropped chunks info if any
                if dropped_chunks > 0:
                    result["dropped_chunks"] = dropped_chunks
                    dropped_chunks = 0
                try:
                    await websocket.send(json.dumps(result))
                except:
                    pass
        
        is_processing = False
    
    try:
        await websocket.send(json.dumps({
            "type": "status",
            "message": "Ready to receive audio..."
        }))
        
        async for message in websocket:
            if isinstance(message, bytes):
                # Check if queue is full (will auto-drop oldest)
                if len(chunk_queue) >= MAX_QUEUE_SIZE:
                    dropped_chunks += 1
                    print(f"⚠ Dropping chunk (queue full: {len(chunk_queue)})")
                
                chunk_queue.append(message)
                print(f"LENGTH CHUNK: {len(chunk_queue)}")
                # Start processing if not already running
                if not is_processing:
                    asyncio.create_task(process_queue())
    
    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected")
    finally:
        final = session.finalize()
        print(f"Session: {final['total_chunks']} chunks, {final['duration']}s, {dropped_chunks} dropped")


async def main():
    global asr_model, vad_model
    
    print("=" * 60)
    print("Simple Live Screen Audio Server (with Silero VAD)")
    print("=" * 60)
    
    # Load Silero VAD model
    print("\nLoading Silero VAD model...")
    try:
        vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        vad_model.eval()
        print("VAD model loaded!")
    except Exception as e:
        print(f"Warning: Could not load VAD model: {e}")
        print("Continuing without VAD...")
        vad_model = None
    
    print("\nLoading ASR model...")
    asr_model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.5,
        max_new_tokens=64,
        enforce_eager=True,
        max_inference_batch_size=2,
        max_model_len=4096,
    )
    print("ASR model loaded!\n")
    
    print(f"Server: ws://localhost:{WEBSOCKET_PORT}")
    print("Open simple_live_client.html in browser")
    print("-" * 60)
    
    async with websockets.serve(handler, "localhost", WEBSOCKET_PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
