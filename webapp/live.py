"""Live market monitor — pulls CURRENT prices + CURRENT news (yfinance) and runs
the TRR pipeline live to produce a real-time crash signal. Needs internet.

Pure functions (no Streamlit) so they're importable/testable headless. The
reasoning backend is MockLLM by default (no GPU/network) — the same 4-phase TRR
logic that the offline 32B uses, just a lighter reasoner for live use.
"""
from __future__ import annotations

from datetime import datetime, timezone

TICKERS = ["AAPL", "AMZN", "GOOGL", "NVDA", "TSLA", "NFLX"]


def fetch_live_headlines(tickers=TICKERS, max_per: int = 6):
    """Current headlines per ticker via yfinance -> list[NewsItem] (today)."""
    import yfinance as yf
    from trr.schema import NewsItem
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    items, seen = [], set()
    for t in tickers:
        try:
            news = getattr(yf.Ticker(t), "news", []) or []
        except Exception:  # noqa: BLE001
            news = []
        for i, it in enumerate(news[:max_per]):
            c = it.get("content", it) if isinstance(it, dict) else {}
            title = (c.get("title") or it.get("title") or "").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            items.append(NewsItem(id=f"{t}-{i}", timestamp=now, title=title,
                                  source="yfinance", assets=[t]))
    return items


def fetch_live_prices(tickers=TICKERS):
    """Latest close + 1-day return per ticker + equal-weight portfolio move."""
    import yfinance as yf
    rows, rets = {}, []
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period="5d")["Close"]
            last, prev = float(h.iloc[-1]), float(h.iloc[-2])
            r = last / prev - 1.0
            rows[t] = {"price": round(last, 2), "ret_1d": r}
            rets.append(r)
        except Exception:  # noqa: BLE001
            rows[t] = {"price": None, "ret_1d": None}
    port = sum(rets) / len(rets) if rets else 0.0
    return rows, port


def run_live(headlines):
    """Run one TRR step over the live headlines -> crash_prob, edges, rationale."""
    from trr.brainstorm import build_impact_graph
    from trr.attention import pagerank_prune
    from trr.llm import MockLLM
    from trr.reason import reason_crash
    llm = MockLLM()
    edges = llm.brainstorm_multi([headlines], TICKERS)[0] if headlines else []
    pruned = pagerank_prune(edges, TICKERS, top_k=30)
    prob, rationale = (reason_crash(pruned, llm, universe=TICKERS)
                       if pruned else (0.0, "no impacts extracted from live news"))
    return {
        "crash_prob": float(prob),
        "rationale": rationale,
        "n_news": len(headlines),
        "n_edges": len(pruned),
        "edges": [{"subject": e.subject, "object": e.object,
                   "polarity": e.polarity, "weight": round(e.weight, 2)}
                  for e in pruned[:20]],
        "asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def live_snapshot():
    """One call -> everything the live monitor needs."""
    heads = fetch_live_headlines()
    prices, port_move = fetch_live_prices()
    sig = run_live(heads)
    return {"signal": sig, "prices": prices, "portfolio_move": port_move,
            "headlines": [{"ticker": h.assets[0], "title": h.title} for h in heads]}


if __name__ == "__main__":
    snap = live_snapshot()
    s = snap["signal"]
    print(f"[live] asof {s['asof']}  crash_prob={s['crash_prob']:.2f}  "
          f"news={s['n_news']} edges={s['n_edges']}  port_move={snap['portfolio_move']:+.2%}")
    print(f"[live] rationale: {s['rationale'][:120]}")
    print(f"[live] {len(snap['headlines'])} live headlines, e.g.:")
    for h in snap["headlines"][:4]:
        print(f"    [{h['ticker']}] {h['title'][:70]}")
