"""Tests for the date-partitioned corpus index (trr.corpus).

Builds a tiny throwaway CSV → SQLite index in a temp dir (no dependency on the
12 GB FNSPID corpus), then checks day partitioning, RAG retrieval bounding, and
that the output feeds the pipeline.

Run: python tests/test_corpus.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from trr.corpus import CorpusIndex, build_index
from trr.pipeline import TRRPipeline
from trr.rag import CausalRAG
from trr.schema import NewsItem


def _tiny_corpus(path: str) -> None:
    """A 3-day corpus: day 2 is a crash day surrounded by calm/dup headlines."""
    rows = []
    # day 1 — calm, with near-duplicates to exercise dedup
    for i in range(5):
        rows.append(("2022-01-01", f"Quiet market mild gains session {i}",
                     "stocks drift higher on light volume", "AAPL", "fnspid"))
    # day 2 — crash signal mixed into a large noisy pool
    rows.append(("2022-01-02", "Major bank collapse triggers selloff and panic",
                 "contagion fears spread as markets plunge", "AAPL", "fnspid"))
    rows.append(("2022-01-02", "Fed warns recession; stocks tumble on crash fears",
                 "liquidation and default risk rise sharply", "NVDA", "fnspid"))
    for i in range(30):
        rows.append(("2022-01-02", f"Routine earnings preview note {i}",
                     "analyst maintains rating", "AAPL", "fnspid"))
    # day 3 — calm
    rows.append(("2022-01-03", "Markets stabilize after volatile week",
                 "buyers return on dip", "AAPL", "fnspid"))
    pd.DataFrame(rows, columns=["date", "title", "summary", "assets", "source"]
                 ).to_csv(path, index=False)


def _build_tmp_index():
    d = tempfile.mkdtemp(prefix="corpus_test_")
    csv = os.path.join(d, "news.csv")
    db = os.path.join(d, "news.db")
    _tiny_corpus(csv)
    n = build_index(csv, db, chunksize=10)   # tiny chunks to exercise chunking
    return db, n


def test_build_and_count():
    db, n = _build_tmp_index()
    assert n == 38, n
    idx = CorpusIndex(db)
    assert idx.count() == 38
    assert idx.available_dates() == ["2022-01-01", "2022-01-02", "2022-01-03"]
    st = idx.stats()
    assert st["per_year"] == {"2022": 38}, st
    idx.close()


def test_day_partition_returns_only_that_day():
    db, _ = _build_tmp_index()
    idx = CorpusIndex(db)
    pool = idx.day("2022-01-02")
    assert len(pool) == 32                       # 2 crash + 30 noise
    assert all(isinstance(it, NewsItem) for it in pool)
    assert all(it.timestamp.date() == date(2022, 1, 2) for it in pool)
    idx.close()


def test_retrieve_bounds_and_surfaces_crash_news():
    db, _ = _build_tmp_index()
    idx = CorpusIndex(db)
    sel = idx.retrieve("2022-01-02", k=5)
    assert len(sel) <= 5
    # The crash headlines must be retrieved ahead of the 30 routine notes.
    titles = " ".join(it.title.lower() for it in sel)
    assert "collapse" in titles or "crash" in titles or "tumble" in titles, titles
    idx.close()


def test_news_by_day_feeds_pipeline():
    db, _ = _build_tmp_index()
    idx = CorpusIndex(db)
    dates = [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)]
    by_day = idx.news_by_day(dates, k=8, portfolio=["AAPL", "NVDA"])
    assert set(by_day.keys()) == set(dates)
    assert all(len(v) <= 8 for v in by_day.values())
    # The bounded, corpus-sourced days run straight through the pipeline.
    df = TRRPipeline().run(by_day, dates)
    assert len(df) == 3
    # The crash day should not be the calmest of the three.
    assert df.loc[date(2022, 1, 2), "crash_prob"] >= df.loc[date(2022, 1, 3), "crash_prob"]
    idx.close()


def test_rag_lookback_bank_spans_full_history_under_sharding():
    """A date-sharded run (start restricts the predicted window) must still fit
    the RAG analogue bank on ALL days in news_by_day, so pre-window crash days
    remain retrievable as causal analogues."""
    days = [date(2022, 1, i) for i in range(1, 9)]
    by_day = {
        d: [NewsItem(id=f"{i}", timestamp=datetime(d.year, d.month, d.day),
                     title=("exchange hack panic selloff plunge" if i < 3
                            else "calm market mild gains"),
                     assets=["AAPL"])]
        for i, d in enumerate(days)
    }
    labels = {d: (1 if i < 3 else 0) for i, d in enumerate(days)}
    rag = CausalRAG(embargo=1, k=2, min_sim=0.0)
    pipe = TRRPipeline(portfolio=["AAPL"], batch=True, cross_batch=True,
                       rag=rag, rag_labels=labels)
    # Predict only the last 3 days; the bank must still cover all 8.
    out = pipe.run(by_day, start=date(2022, 1, 6))
    assert len(out) == 3
    assert len(rag._dates) == 8, rag._dates       # full history, not just window
    assert days[0] in rag._dates                  # a pre-window day is in the bank


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("[corpus] all tests passed")
