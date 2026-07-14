#!/usr/bin/env python3
"""Tiny local progress popout for long-running Biostuff jobs."""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HTML = """<!doctype html><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Biostuff progress</title><style>
body{font:16px -apple-system,BlinkMacSystemFont,sans-serif;background:#101418;color:#f4f7f9;margin:0;padding:28px}
main{max-width:620px;margin:auto;background:#182027;border:1px solid #34414a;border-radius:18px;padding:26px;box-shadow:0 18px 60px #0008}
h1{margin:0 0 22px;font-size:22px}.big{font-size:46px;font-weight:700;margin:10px 0}progress{width:100%;height:22px;accent-color:#62d2a2}
dl{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:24px}dt{color:#9eabb3;font-size:12px;text-transform:uppercase}dd{margin:3px 0 0;font-size:18px}.muted{color:#9eabb3;font-size:13px;margin-top:24px}
</style><main><h1>Biostuff scoring</h1><div id="pct" class="big">Loading...</div><progress id="bar" max="1" value="0"></progress>
<dl><div><dt>Completed</dt><dd id="done">-</dd></div><div><dt>Total</dt><dd id="total">-</dd></div><div><dt>Elapsed observed</dt><dd id="elapsed">-</dd></div></dl><p id="updated" class="muted">Waiting for progress file...</p></main>
<script>
const fmt=s=>{s=Math.max(0,Math.round(s));let h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;return(h?h+'h ':'')+(m?m+'m ':'')+x+'s'};let samples=[];
async function tick(){try{let p=await(await fetch('/progress?x='+Date.now())).json(),now=Date.now()/1000;if(!samples.length||samples[samples.length-1].done!==p.completed)samples.push({done:p.completed,time:now});samples=samples.slice(-6);let first=samples[0],elapsed=now-first.time;pct.textContent=((p.fraction||0)*100).toFixed(2)+'%';bar.value=p.fraction||0;done.textContent=p.completed.toLocaleString();total.textContent=p.total.toLocaleString();document.querySelector('#elapsed').textContent=fmt(elapsed);updated.textContent='Updated '+new Date(now*1000).toLocaleTimeString()}catch(e){updated.textContent='Waiting for progress file...'}}tick();setInterval(tick,2000);
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
    parser.add_argument("--progress", default="results/proteingym_train_progress.json")
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
