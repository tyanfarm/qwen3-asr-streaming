#!/bin/bash
# Start vLLM server for Qwen3-ASR with optimized parameters for streaming

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Parameters explained:
# --model: The model path
# --max-model-len 16384: Reduced from 65536 to prevent KV cache OOM
# --gpu-memory-utilization 0.7: Balanced memory usage
# --enforce-eager: Required for stable execution in this environment
# --disable-log-stats: Disable periodic stats logging to keep output clean, enable if debugging
# --port 8000: Default vLLM port

echo "Starting qwen-asr-serve for Qwen3-ASR..."
qwen-asr-serve Qwen/Qwen3-ASR-1.7B \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.7 \
    --enforce-eager \
    --allowed-local-media-path / \
    --port 8000 \
    --disable-log-stats
