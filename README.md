# Qwen3-ASR WebSocket Client

## Streaming Tab Chrome

The WebSocket API is available at `ws://localhost:8765`.

### Usage

1. Start the WebSocket server:
   ```bash
   python simple_live_server.py
   ```

2. Start the WebSocket client:
   ```bash
   python -m http.server 9000
   ```

   - Open `http://localhost:9000/simple_live_client.html` in your browser.

3. Enter a Youtube video URL and start streaming.

## Simple Streaming with Youtube Audio

### Usage

1. Start the WebSocket server:
   ```bash
   youtube_websocket_streaming.py
   ```

2. Start the WebSocket client:
   ```bash
   python -m http.server 9000
   ```

   - Open `http://localhost:9000/websocket_client.html` in your browser.

3. Enter a YouTube URL and click Start to begin transcription.