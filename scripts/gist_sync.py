"""Sync data/live/* to a secret GitHub Gist every N seconds.

The always-on local daemon writes fresh 7B summaries / prices / news to
data/live/*. This sidecar pushes them to a secret gist so the Streamlit Cloud
app (which has no daemon) can read the freshest data via the gist raw URLs
(see webapp/live.py: GIST_RAW). Uses the authenticated `gh` CLI — no token in
code.

Run:  nohup .venv/bin/python -m scripts.gist_sync > data/live/gist_sync.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import subprocess
import time

GIST_ID = os.environ.get("GIST_ID", "0c5ef7ebcd9d275798653fef36424b4b")
INTERVAL = int(os.environ.get("GIST_SYNC_SEC", "60"))
BASE = "data/live"
FILES = ["signal.json", "summary.json", "prices.json", "daily_report.json", "news.jsonl"]


def push() -> tuple[int, str]:
    files = {}
    for f in FILES:
        p = os.path.join(BASE, f)
        if os.path.exists(p):
            try:
                files[f] = {"content": open(p, encoding="utf-8").read()}
            except Exception as e:  # noqa: BLE001
                return 1, f"read {f}: {e}"
    if not files:
        return 1, "no files"
    payload = json.dumps({"files": files})
    r = subprocess.run(
        ["gh", "api", "--method", "PATCH", f"/gists/{GIST_ID}", "--input", "-"],
        input=payload, text=True, capture_output=True)
    return r.returncode, (r.stderr or "").strip()[:200]


def main() -> None:
    print(f"[gist-sync] gist={GIST_ID} every {INTERVAL}s files={FILES}", flush=True)
    while True:
        rc, err = push()
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {'ok' if rc == 0 else 'ERR ' + err}", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
