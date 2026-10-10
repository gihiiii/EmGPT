"""
MiniGPT-Em - Interactive Web Demo

Usage:
    python app.py

Features:
- Dual-mode architecture:
  1. Primary: Uses Gradio when available (ideal for Hugging Face Spaces deployment).
  2. Fallback: Automatically falls back to built-in Python HTTP server with a modern,
     responsive web UI if Gradio C-extensions/DLLs are restricted by system policies.
"""

import sys
import json
import argparse
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from model.transformer import MiniGPT
from tokenizer.bpe import BPETokenizer
from generate import load_model, generate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_model.pt"
if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "latest_model.pt"

TOKENIZER_DIR = BASE_DIR / "tokenizer"

print(f"[App] Loading model from {CHECKPOINT_PATH} on {DEVICE}...")
model, config = load_model(CHECKPOINT_PATH, device=DEVICE)
tokenizer = BPETokenizer.load(TOKENIZER_DIR)
print(f"[App] Model & Tokenizer loaded! (~{model.num_parameters() / 1e6:.1f}M params)")


def run_generation(
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
) -> str:
    """Generate lyrics string from prompt."""
    if not prompt or not prompt.strip():
        return "Please enter a prompt to begin generation."

    try:
        return generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt.strip(),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k) if top_k > 0 else None,
            top_p=float(top_p) if top_p < 1.0 else None,
            repetition_penalty=float(repetition_penalty),
            device=DEVICE,
        )
    except Exception as e:
        return f"Generation Error: {str(e)}"


# Built-in lightweight modern Web Server (Fallback for local environments)
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EmGPT - Eminem-Style GPT</title>
    <style>
        :root {
            --bg: #0f172a;
            --surface: #1e293b;
            --border: #334155;
            --primary: #f59e0b;
            --text: #f8fafc;
            --muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 2rem 1rem; display: flex; justify-content: center; }
        .container { max-width: 900px; width: 100%; }
        header { text-align: center; margin-bottom: 2rem; }
        h1 { font-size: 2.2rem; color: var(--primary); margin-bottom: 0.5rem; }
        p.subtitle { color: var(--muted); font-size: 1rem; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
        .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem; }
        label { display: block; font-size: 0.875rem; font-weight: 600; margin-bottom: 0.4rem; color: #e2e8f0; }
        textarea, input[type=range] { width: 100%; }
        textarea { background: #0f172a; border: 1px solid var(--border); border-radius: 8px; color: var(--text); padding: 0.75rem; font-size: 0.95rem; resize: vertical; }
        .slider-group { margin-bottom: 1rem; }
        .slider-header { display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--muted); margin-bottom: 0.25rem; }
        button.btn { width: 100%; background: var(--primary); color: #000; border: none; font-weight: 700; padding: 0.85rem; border-radius: 8px; cursor: pointer; font-size: 1rem; transition: opacity 0.2s; margin-top: 0.5rem; }
        button.btn:hover { opacity: 0.9; }
        .output-box { min-height: 280px; white-space: pre-wrap; word-break: break-word; font-family: 'Courier New', Courier, monospace; background: #0b1120; border: 1px solid var(--border); border-radius: 8px; padding: 1rem; font-size: 0.95rem; line-height: 1.5; color: #38bdf8; }
        .examples { margin-top: 1.5rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }
        .example-pill { background: #334155; padding: 0.35rem 0.75rem; border-radius: 20px; font-size: 0.8rem; cursor: pointer; border: 1px solid transparent; }
        .example-pill:hover { border-color: var(--primary); color: var(--primary); }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>🎤 EmGPT: ~12.7M Parameter Language Model</h1>
        <p class="subtitle">Trained from scratch in PyTorch on Eminem lyrics with custom BPE Tokenization.</p>
    </header>

    <div class="grid">
        <div class="card">
            <div style="margin-bottom: 1.2rem;">
                <label for="prompt">Prompt</label>
                <textarea id="prompt" rows="3">Look, if you had one shot</textarea>
            </div>

            <div class="slider-group">
                <div class="slider-header"><span>Max New Tokens</span><span id="val-tokens">80</span></div>
                <input type="range" id="tokens" min="20" max="200" step="5" value="80" oninput="document.getElementById('val-tokens').innerText = this.value">
            </div>

            <div class="slider-group">
                <div class="slider-header"><span>Temperature (Creativity)</span><span id="val-temp">0.80</span></div>
                <input type="range" id="temp" min="0.1" max="1.5" step="0.05" value="0.80" oninput="document.getElementById('val-temp').innerText = this.value">
            </div>

            <div class="slider-group">
                <div class="slider-header"><span>Top-K Filter</span><span id="val-topk">40</span></div>
                <input type="range" id="topk" min="0" max="100" step="5" value="40" oninput="document.getElementById('val-topk').innerText = this.value">
            </div>

            <div class="slider-group">
                <div class="slider-header"><span>Repetition Penalty</span><span id="val-rep">1.15</span></div>
                <input type="range" id="rep" min="1.0" max="2.0" step="0.05" value="1.15" oninput="document.getElementById('val-rep').innerText = this.value">
            </div>

            <button class="btn" id="generate-btn" onclick="submitPrompt()">🔥 Spit Bars</button>

            <div class="examples">
                <span class="example-pill" onclick="setPrompt(this.innerText)">Look, if you had one shot</span>
                <span class="example-pill" onclick="setPrompt(this.innerText)">I'm not afraid to</span>
                <span class="example-pill" onclick="setPrompt(this.innerText)">Guess who's back</span>
                <span class="example-pill" onclick="setPrompt(this.innerText)">Hi! My name is</span>
            </div>
        </div>

        <div class="card">
            <label>Generated Lyrics</label>
            <div class="output-box" id="output">Click 'Spit Bars' to generate rhymes...</div>
        </div>
    </div>
</div>

<script>
    function setPrompt(text) {
        document.getElementById('prompt').value = text;
    }

    async function submitPrompt() {
        const btn = document.getElementById('generate-btn');
        const out = document.getElementById('output');
        btn.disabled = true;
        btn.innerText = "⏳ Generating...";
        out.innerText = "Thinking and composing bars...";

        try {
            const res = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: document.getElementById('prompt').value,
                    max_new_tokens: document.getElementById('tokens').value,
                    temperature: document.getElementById('temp').value,
                    top_k: document.getElementById('topk').value,
                    repetition_penalty: document.getElementById('rep').value
                })
            });
            const data = await res.json();
            out.innerText = data.output || data.error;
        } catch (e) {
            out.innerText = "Error: " + e;
        } finally {
            btn.disabled = false;
            btn.innerText = "🔥 Spit Bars";
        }
    }
</script>
</body>
</html>"""


def launch_builtin_server(port: int = 7860):
    """Launch built-in zero-dependency HTTP server with REST endpoint."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class RequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

        def do_POST(self):
            if self.path == "/api/generate":
                content_len = int(self.headers.get("Content-Length", 0))
                post_body = self.rfile.read(content_len)
                try:
                    payload = json.loads(post_body.decode("utf-8"))
                    prompt = payload.get("prompt", "")
                    tokens = int(payload.get("max_new_tokens", 80))
                    temp = float(payload.get("temperature", 0.8))
                    topk = int(payload.get("top_k", 40))
                    repp = float(payload.get("repetition_penalty", 1.15))

                    generated_text = run_generation(
                        prompt=prompt,
                        max_new_tokens=tokens,
                        temperature=temp,
                        top_k=topk,
                        repetition_penalty=repp,
                    )
                    res = json.dumps({"output": generated_text}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(res)
                except Exception as e:
                    err_res = json.dumps({"error": str(e)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(err_res)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            # Suppress noisy standard request log lines
            return

    server = HTTPServer(("0.0.0.0", port), RequestHandler)
    print(f"\n=======================================================")
    print(f"🚀 EmGPT Web Demo is LIVE at: http://localhost:{port}")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")


def launch_gradio():
    """Launch Gradio UI for environments where Gradio is fully supported."""
    import gradio as gr

    demo = gr.Interface(
        fn=run_generation,
        inputs=[
            gr.Textbox(label="Prompt", value="Look, if you had one shot", lines=3),
            gr.Slider(10, 200, value=80, step=5, label="Max New Tokens"),
            gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature"),
            gr.Slider(0, 100, value=40, step=5, label="Top-K Filter"),
            gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P Filter"),
            gr.Slider(1.0, 2.0, value=1.15, step=0.05, label="Repetition Penalty"),
        ],
        outputs=gr.Textbox(label="Generated Output", lines=12),
        title="🎤 EmGPT: ~12.7M Parameter Language Model",
        description="Autoregressive language model built from scratch in PyTorch on Eminem lyrics.",
    )
    demo.launch(share=False)


def main():
    parser = argparse.ArgumentParser(description="EmGPT Web App")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--force-local", action="store_true", help="Force built-in HTTP server")
    args = parser.parse_args()

    if not args.force_local:
        try:
            launch_gradio()
            return
        except Exception as e:
            print(f"[Notice] Gradio initialization bypassed ({e}). Falling back to built-in web server...")

    launch_builtin_server(port=args.port)


if __name__ == "__main__":
    main()
