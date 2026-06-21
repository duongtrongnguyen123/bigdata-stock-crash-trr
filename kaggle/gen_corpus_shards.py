"""Generate the full-corpus 2016-2023 stock backtest as date shards.

Splits the 2012 trading days into N contiguous windows and emits, for each
window, a BASE (USE_RAG=0) and a RAG (USE_RAG=1) shard kernel — so the whole
2016-2023 backtest fans out across many Kaggle accounts (2 notebooks each) and
finishes in one ~20-min wave instead of a ~5 h single run.

Each shard is the self-contained stock_standalone.py with a tiny env prelude
(TARGET_MODES/USE_RAG/TRR_START/TRR_END) injected right after the __future__
import. Every shard attaches the SAME dataset (full corpus stocknews.csv +
prices), so the RAG analogue bank spans the whole history (causal lookback)
while each shard only PREDICTS its window.

Outputs:
  kaggle/sd_corpus/            shared dataset (prices/*.csv + stocknews.csv)
  kaggle/cshards/cb{j}.py      base shard j      (j = 0..N-1)
  kaggle/cshards/cr{j}.py      rag  shard j
  kaggle/cshards/manifest.json [{tag,config,start,end,days}, ...]

Run: python kaggle/gen_corpus_shards.py [N=21]
"""
import json
import os
import shutil
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICES_SRC = os.path.join(ROOT, "data/ohlcv")   # full 2016-2023 OHLCV
NEWS_SRC = os.path.join(ROOT, "data/stockdata/stocknews_corpus.csv")
STANDALONE = os.path.join(ROOT, "kaggle/stock_standalone.py")
DS_DIR = os.path.join(ROOT, "kaggle/sd_corpus")
SH_DIR = os.path.join(ROOT, "kaggle/cshards")
TICKERS = ["AAPL", "AMZN", "GOOGL", "NVDA", "TSLA", "NFLX"]
FUTURE = "from __future__ import annotations\n"


def _norm_prices(src_csv: str) -> pd.DataFrame:
    """Read an OHLCV file and return a normalized date,close frame."""
    df = pd.read_csv(src_csv)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[["date", "close"]]


def windows(n: int):
    days = _norm_prices(os.path.join(PRICES_SRC, "AAPL.csv"))["date"].tolist()
    days = [d for d in days if "2016-01-01" <= d <= "2023-12-31"]
    size = -(-len(days) // n)            # ceil
    out = []
    for j in range(n):
        chunk = days[j * size:(j + 1) * size]
        if chunk:
            out.append((j, chunk[0], chunk[-1], len(chunk)))
    return out


def build_dataset():
    # FLAT layout (no subfolders): Kaggle `--dir-mode zip` skips subdirectories,
    # so price files sit beside stocknews.csv at the dataset root. The kernel
    # globs /kaggle/input/**/AAPL.csv, which matches the flat layout.
    os.makedirs(DS_DIR, exist_ok=True)
    for t in TICKERS:
        _norm_prices(os.path.join(PRICES_SRC, f"{t}.csv")).to_csv(
            os.path.join(DS_DIR, f"{t}.csv"), index=False)
    shutil.copy(NEWS_SRC, os.path.join(DS_DIR, "stocknews.csv"))
    sz = os.path.getsize(os.path.join(DS_DIR, "stocknews.csv")) / 1e6
    print(f"[gen] dataset -> {DS_DIR}  (stocknews {sz:.1f} MB + 6 price files)")


def _prelude(use_rag: int, start: str, end: str) -> str:
    return (
        "import os as _os\n"
        '_os.environ["TARGET_MODES"] = "crash"\n'
        f'_os.environ["USE_RAG"] = "{use_rag}"\n'
        f'_os.environ["TRR_START"] = "{start}"\n'
        f'_os.environ["TRR_END"] = "{end}"\n'
    )


def build_shards(n: int):
    src = open(STANDALONE).read()
    if FUTURE not in src:
        raise SystemExit("standalone missing __future__ import — cannot inject prelude")
    os.makedirs(SH_DIR, exist_ok=True)
    manifest = []
    for j, start, end, ndays in windows(n):
        for cfg, rag in (("cb", 0), ("cr", 1)):
            body = src.replace(FUTURE, FUTURE + _prelude(rag, start, end), 1)
            tag = f"{cfg}{j}"
            open(os.path.join(SH_DIR, f"{tag}.py"), "w").write(body)
            manifest.append({"tag": tag, "config": "base" if rag == 0 else "rag",
                             "shard": j, "start": start, "end": end, "days": ndays})
    json.dump(manifest, open(os.path.join(SH_DIR, "manifest.json"), "w"), indent=2)
    nb = sum(1 for m in manifest if m["config"] == "base")
    print(f"[gen] {len(manifest)} shards ({nb} base + {nb} rag) over {nb} windows "
          f"-> {SH_DIR}")
    print(f"[gen] window size ~{manifest[0]['days']} days; "
          f"span {manifest[0]['start']}..{manifest[-1]['end']}")
    return manifest


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    build_dataset()
    build_shards(n)
