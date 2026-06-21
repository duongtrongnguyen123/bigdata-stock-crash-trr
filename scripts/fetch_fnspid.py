"""Stream-filter the 23 GB FNSPID news CSV (read from stdin) — never storing the
full file. Two modes via env:

  HEADLINE mode (default): keep title+date+symbol for a ticker set
    FNSPID_TICKERS=AAPL,AMZN,...  FNSPID_OUT=data/fnspid/stocknews.csv

  CORPUS mode: keep the full Article BODY (+ summary) for ALL tickers from
  MIN_YEAR — builds a ~10 GB 2016-2023 inference corpus that RAG draws from:
    FNSPID_KEEP_BODY=1  FNSPID_ALL_TICKERS=1  FNSPID_MIN_YEAR=2016
    FNSPID_OUT=data/fnspid_corpus/news.csv

Usage:
    curl -sL "<FNSPID nasdaq_exteral_data.csv URL>" | python -m scripts.fetch_fnspid
"""
from __future__ import annotations

import os
import sys

import pandas as pd

TICKERS = set(os.environ.get(
    "FNSPID_TICKERS", "AAPL,AMZN,GOOGL,NVDA,TSLA,NFLX").split(","))
MIN_YEAR = int(os.environ.get("FNSPID_MIN_YEAR", "2016"))
OUT = os.environ.get("FNSPID_OUT", "data/fnspid/stocknews.csv")
KEEP_BODY = os.environ.get("FNSPID_KEEP_BODY", "0") == "1"
ALL_TICKERS = os.environ.get("FNSPID_ALL_TICKERS", "0") == "1"

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

cols = ["Date", "Article_title", "Stock_symbol"]
if KEEP_BODY:
    cols += ["Article", "Lsa_summary"]

kept = 0
chunks_seen = 0
first_write = True
reader = pd.read_csv(sys.stdin, usecols=cols, dtype=str, chunksize=100_000,
                     on_bad_lines="skip", engine="c")
for ch in reader:
    chunks_seen += 1
    if not ALL_TICKERS:
        ch = ch[ch["Stock_symbol"].isin(TICKERS)]
    if ch.empty:
        if chunks_seen % 20 == 0:
            print(f"  [{chunks_seen*100}k rows scanned] kept={kept}", flush=True)
        continue
    dt = pd.to_datetime(ch["Date"], errors="coerce", utc=True)
    ch = ch[dt.dt.year >= MIN_YEAR].copy()
    if ch.empty:
        continue
    ch["date"] = pd.to_datetime(ch["Date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    data = {"date": ch["date"], "title": ch["Article_title"].astype(str),
            "assets": ch["Stock_symbol"].astype(str), "source": "fnspid"}
    if KEEP_BODY:
        # prefer the LSA summary (compact) but keep the body too for RAG passages
        data["summary"] = ch["Lsa_summary"].astype(str)
        data["body"] = ch["Article"].astype(str)
    out = pd.DataFrame(data).dropna(subset=["date"])
    out.to_csv(OUT, mode="w" if first_write else "a", header=first_write, index=False)
    first_write = False
    kept += len(out)
    if chunks_seen % 20 == 0:
        sz = os.path.getsize(OUT) / 1e9 if os.path.exists(OUT) else 0
        print(f"  [{chunks_seen*100}k rows scanned] kept={kept}  out={sz:.2f} GB", flush=True)

print(f"DONE: scanned ~{chunks_seen*100}k rows, kept {kept} articles -> {OUT}", flush=True)
