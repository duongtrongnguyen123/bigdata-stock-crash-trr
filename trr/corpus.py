"""Date-partitioned index over the large 2016-2023 FNSPID news corpus.

The corpus (``data/fnspid_corpus/news.csv``, ~12 GB, ~4.5M articles) is far too
large to hold in memory or scan per query. This module builds a *date-indexed
SQLite store* once, then serves any single day's news pool in O(log n) — so RAG
retrieval can draw per-day relevant passages from the full corpus without ever
loading it into RAM.

Flow at inference:
    corpus -> CorpusIndex.day(d)            # the day's FULL news pool (cheap)
           -> select_relevant(pool, query)  # bounded top-k crash-relevant slice
           -> TRRPipeline                    # LLM reasons over the bounded slice

The index stores the compact LSA ``summary`` as each item's body (a usable
passage) rather than the full article text, keeping the DB lean (~2-3 GB) and
queries fast; the full bodies remain in the CSV if ever needed.

Build once:
    python -m trr.corpus build      # data/fnspid_corpus/news.csv -> .../news.db
Then:
    from trr.corpus import CorpusIndex
    idx = CorpusIndex()
    by_day = idx.news_by_day(dates, k=40, portfolio=PORTFOLIO)   # pipeline-ready
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
import sys

import pandas as pd

from trr.schema import PORTFOLIO, NewsItem

DEFAULT_CSV = "data/fnspid_corpus/news.csv"
DEFAULT_DB = "data/fnspid_corpus/news.db"

# We index title + summary (compact passage) + tagged ticker; the full article
# body is intentionally left out of the index to keep it lean and fast.
_READ_COLS = ["date", "title", "summary", "assets", "source"]


def _clean(v) -> str:
    """Normalise a CSV cell to a stripped string ('' for NaN / 'nan')."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def build_index(csv_path: str = DEFAULT_CSV, db_path: str = DEFAULT_DB,
                chunksize: int = 100_000) -> int:
    """Stream the corpus CSV into a date-indexed SQLite store. Returns row count.

    Memory-bounded: reads the CSV in chunks (peak ~one chunk), so this never
    holds the 12 GB corpus in RAM. The date index is created AFTER the bulk load
    (far faster than maintaining it per-insert).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    # Build-time PRAGMAs: durability not needed for a rebuildable derived index.
    cur.execute("PRAGMA journal_mode=OFF")
    cur.execute("PRAGMA synchronous=OFF")
    cur.execute("CREATE TABLE news (date TEXT, title TEXT, body TEXT, "
                "assets TEXT, source TEXT)")

    total = 0
    reader = pd.read_csv(csv_path, usecols=_READ_COLS, dtype=str,
                         chunksize=chunksize, on_bad_lines="skip", engine="c")
    for ci, ch in enumerate(reader, 1):
        rows = []
        for date, title, summary, assets, source in zip(
                ch["date"], ch["title"], ch["summary"], ch["assets"], ch["source"]):
            d = _clean(date)[:10]            # YYYY-MM-DD
            t = _clean(title)
            if not d or not t:
                continue
            rows.append((d, t, _clean(summary), _clean(assets), _clean(source)))
        if rows:
            cur.executemany("INSERT INTO news VALUES (?,?,?,?,?)", rows)
            total += len(rows)
        if ci % 10 == 0:
            con.commit()
            print(f"  [{ci*chunksize//1000}k rows scanned] indexed={total}", flush=True)

    con.commit()
    print("  building date index ...", flush=True)
    cur.execute("CREATE INDEX idx_news_date ON news(date)")
    con.commit()
    con.close()
    print(f"DONE: indexed {total} articles -> {db_path}", flush=True)
    return total


class CorpusIndex:
    """Read-only date-partitioned view over the news corpus."""

    def __init__(self, db_path: str = DEFAULT_DB) -> None:
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"{db_path} not found — build it with `python -m trr.corpus build`")
        # Read-only connection; immutable=1 lets it open without a write lock.
        self.con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True,
                                   check_same_thread=False)

    @staticmethod
    def _as_str(d) -> str:
        if isinstance(d, (dt.date, dt.datetime)):
            return d.strftime("%Y-%m-%d")
        return str(d)[:10]

    def _rows_to_items(self, rows) -> list[NewsItem]:
        items = []
        for i, (date, title, body, assets, source) in enumerate(rows):
            try:
                ts = dt.datetime.strptime(date, "%Y-%m-%d")
            except (ValueError, TypeError):
                continue
            tickers = [a.strip().upper() for a in (assets or "").split(",") if a.strip()]
            items.append(NewsItem(id=f"{date}:{i}", timestamp=ts, title=title,
                                  body=body or "", source=source or "",
                                  assets=tickers))
        return items

    def day(self, d) -> list[NewsItem]:
        """The FULL news pool for a single day (unbounded)."""
        cur = self.con.execute(
            "SELECT date,title,body,assets,source FROM news WHERE date=?",
            (self._as_str(d),))
        return self._rows_to_items(cur.fetchall())

    def window(self, end, lookback_days: int) -> list[NewsItem]:
        """All news in [end-lookback, end] — for a multi-day live context."""
        end_d = self._as_str(end)
        start_d = (dt.date.fromisoformat(end_d)
                   - dt.timedelta(days=lookback_days)).isoformat()
        cur = self.con.execute(
            "SELECT date,title,body,assets,source FROM news "
            "WHERE date BETWEEN ? AND ? ORDER BY date", (start_d, end_d))
        return self._rows_to_items(cur.fetchall())

    def retrieve(self, d, query: str = None, k: int = 40,
                 portfolio=PORTFOLIO) -> list[NewsItem]:
        """RAG extraction: the k most crash/portfolio-relevant items for day `d`,
        retrieved from that day's full corpus pool via TF-IDF select_relevant."""
        from trr.select import crash_query, select_relevant
        pool = self.day(d)
        if not pool:
            return []
        q = query or crash_query(portfolio)
        return select_relevant(pool, q, k, portfolio)

    def news_by_day(self, dates, k: int = 40, portfolio=PORTFOLIO,
                    query: str = None) -> dict:
        """Pipeline-ready {date: [bounded relevant NewsItems]} drawn from the
        corpus. Each day is reduced to its k most relevant items, so the corpus
        scales to GBs while the LLM input stays O(days * k)."""
        from trr.select import crash_query
        q = query or crash_query(portfolio)
        out = {}
        for d in dates:
            key = d if isinstance(d, (dt.date, dt.datetime)) else dt.date.fromisoformat(str(d)[:10])
            if isinstance(key, dt.datetime):
                key = key.date()
            out[key] = self.retrieve(d, q, k, portfolio)
        return out

    def available_dates(self) -> list[str]:
        cur = self.con.execute("SELECT DISTINCT date FROM news ORDER BY date")
        return [r[0] for r in cur.fetchall()]

    def count(self) -> int:
        return self.con.execute("SELECT COUNT(*) FROM news").fetchone()[0]

    def stats(self) -> dict:
        cur = self.con.execute(
            "SELECT substr(date,1,4) y, COUNT(*) FROM news GROUP BY y ORDER BY y")
        per_year = {y: n for y, n in cur.fetchall()}
        return {"total": sum(per_year.values()), "per_year": per_year}

    def close(self) -> None:
        self.con.close()


def _main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        csv_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CSV
        db_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_DB
        build_index(csv_path, db_path)
        idx = CorpusIndex(db_path)
        st = idx.stats()
        print(f"[corpus] {st['total']} articles indexed; per-year: {st['per_year']}")
        return
    # No args: quick stats + a sample retrieval over an existing index.
    idx = CorpusIndex()
    st = idx.stats()
    print(f"[corpus] {st['total']} articles; per-year {st['per_year']}")
    dates = idx.available_dates()
    if dates:
        mid = dates[len(dates) // 2]
        pool = idx.day(mid)
        sel = idx.retrieve(mid, k=10)
        print(f"[corpus] {mid}: pool={len(pool)} -> retrieved {len(sel)} relevant")
        for it in sel[:3]:
            print(f"    {it.assets}  {it.title[:80]}")


if __name__ == "__main__":
    _main()
