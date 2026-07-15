#!/usr/bin/env python3
"""Tiny local progress popout for long-running Biostuff jobs."""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HTML = """<!doctype html><meta name="viewport" content="width=device-width, initial-scale=1"><title>Biostuff live run</title><style>
body{font:15px -apple-system,BlinkMacSystemFont,sans-serif;background:#101418;color:#f4f7f9;margin:0;padding:28px}main{max-width:760px;margin:auto;background:#182027;border:1px solid #34414a;border-radius:18px;padding:26px;box-shadow:0 18px 60px #0008}h1{margin:0 0 6px;font-size:24px}.sub,.muted{color:#9eabb3}.big{font-size:46px;font-weight:700;margin:18px 0 10px}progress{width:100%;height:22px;accent-color:#62d2a2}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:24px}.card{background:#202b33;border-radius:12px;padding:14px}dt{color:#9eabb3;font-size:11px;text-transform:uppercase}dd{margin:5px 0 0;font-size:18px;overflow-wrap:anywhere}.events{margin-top:24px;background:#11191f;border-radius:12px;padding:14px;max-height:240px;overflow:auto;line-height:1.65}.ok{color:#62d2a2}.warn{color:#ffd166}@media(max-width:600px){.grid{grid-template-columns:1fr 1fr}}
</style><main><h1 id="title">Biostuff</h1><div class="sub">Live local model run</div><div id="pct" class="big">Waiting...</div><progress id="bar" max="1" value="0"></progress><div class="grid"><div class="card"><dt>Rows complete</dt><dd id="done">-</dd></div><div class="card"><dt>Total rows</dt><dd id="total">-</dd></div><div class="card"><dt>Stage</dt><dd id="stage">-</dd></div><div class="card"><dt>Current assay</dt><dd id="assay">-</dd></div><div class="card"><dt>Current operation</dt><dd id="operation">-</dd></div><div class="card"><dt>Assay progress</dt><dd id="assayProgress">-</dd></div></div><section class="events"><strong>Recent events</strong><div id="events" class="muted">Waiting for run...</div></section><p id="updated" class="muted">Waiting for progress file...</p></main><script>
async function tick(){try{let p=await(await fetch('/progress?x='+Date.now())).json();title.textContent=p.title||'Biostuff';let f=p.fraction||0;pct.textContent=(f*100).toFixed(2)+'%';bar.value=f;done.textContent=(p.completed||0).toLocaleString();total.textContent=(p.total||0).toLocaleString();stage.textContent=p.stage||'-';assay.textContent=p.current_assay||'-';operation.textContent=p.operation||'-';assayProgress.textContent=(p.current_assay_completed||0)+' / '+(p.current_assay_total||0);events.innerHTML=(p.events||[]).slice().reverse().map(x=>'<div>'+x.replaceAll('&','&amp;').replaceAll('<','&lt;')+'</div>').join('')||'Waiting for run...';updated.textContent='Updated '+new Date().toLocaleTimeString()}catch(e){updated.textContent='Waiting for progress file...'}}tick();setInterval(tick,1000);
</script>"""


class Handler(BaseHTTPRequestHandler):
    progress_path: Path

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/progress"):
            try:
                payload = json.loads(self.progress_path.read_text())
                payload["file_mtime"] = self.progress_path.stat().st_mtime
            except (FileNotFoundError, json.JSONDecodeError):
                payload = {"completed": 0, "total": 0, "fraction": 0}
            body = json.dumps(payload).encode()
            content_type = "application/json"
        else:
            body = HTML.encode()
            content_type = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", default="results/head_b_progress.json")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    Handler.progress_path = Path(args.progress).expanduser().resolve()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(url, flush=True)
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
