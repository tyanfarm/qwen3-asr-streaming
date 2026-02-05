#!/usr/bin/env python3
"""
Simplified Live Screen Audio Server - Uses raw PCM audio for low latency.

Features:
- Backpressure: drops old chunks when processing can't keep up
- Rate limiting: prevents VRAM overflow from chunk accumulation

Usage:
    python simple_live_server.py
"""

import asyncio
import json
import time
from collections import deque
from typing import Optional

import numpy as np
import websockets

from qwen_asr import Qwen3ASRModel


WEBSOCKET_PORT = 8765
SAMPLE_RATE = 16000
MAX_QUEUE_SIZE = 40  # Max chunks to buffer before dropping

asr_model: Optional[Qwen3ASRModel] = None


class TranscriptionSession:
    RESET_INTERVAL_SEC = 30  # Reset streaming state every 30 seconds
    
    def __init__(self, asr_model: Qwen3ASRModel):
        self.asr_model = asr_model
        self.state = asr_model.init_streaming_state(
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=2.0,
        )
        self.last_text = ""
        self.start_time = time.time()
        self.state_start_time = time.time()  # Track when state was last reset
        self.chunk_count = 0
    
    def process_pcm_chunk(self, pcm_bytes: bytes) -> dict:
        """Process raw PCM audio chunk."""
        chunk_start = time.time()
        
        try:
            # Reset state periodically to prevent token overflow
            if time.time() - self.state_start_time > self.RESET_INTERVAL_SEC:
                print(f"🔄 Resetting streaming state (after {self.RESET_INTERVAL_SEC}s)")
                self.asr_model.finish_streaming_transcribe(self.state)
                self.state = self.asr_model.init_streaming_state(
                    unfixed_chunk_num=2,
                    unfixed_token_num=5,
                    chunk_size_sec=2.0,
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
            
            # Process with ASR
            self.asr_model.streaming_transcribe(samples, self.state)
            
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
    global asr_model
    
    print("=" * 60)
    print("Simple Live Screen Audio Server")
    print("=" * 60)
    
    print("\nLoading model...")
    asr_model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.5,
        max_new_tokens=64,
        enforce_eager=True,
        max_inference_batch_size=2,
        max_model_len=4096,
    )
    print("Model loaded!\n")
    
    print(f"Server: ws://localhost:{WEBSOCKET_PORT}")
    print("Open simple_live_client.html in browser")
    print("-" * 60)
    
    async with websockets.serve(handler, "localhost", WEBSOCKET_PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
