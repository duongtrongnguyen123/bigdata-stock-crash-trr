# Results — Temporal Relational Reasoning of LLMs for Crypto Crash Prediction

Empirical study for the assignment *"Temporal Relational Reasoning of Large
Language Models for Stock Price Prediction"* (crypto adaptation of
[arXiv:2410.17266](https://arxiv.org/abs/2410.17266)).

**Task.** For each day, predict the probability that an equal-weight crypto
portfolio (BTC, ETH, SOL, BNB, AVAX, DOGE) **crashes** — drops > 8% over the next
3 days. Binary, imbalanced (~11% positive), scored by **AUROC**. The LLM reasons
**zero-shot / few-shot over news** (price is used only for labels and as an
optional ensemble signal).

**Pipeline.** Four phases per day: Brainstorm (news → directed impact graph) →
Memory (decay `R=exp(-t·λ)`) → Attention (PageRank prune) → Reasoning (LLM →
crash probability). Run on Kaggle RTX 6000 Pro (Blackwell, sm_120), batched
`transformers`, no internet.

---

## Headline results

| Setup | Model | Window | AUROC |
|---|---|---|---:|
| No few-shot (baseline) | Qwen2.5-14B | 2022–23 | 0.505 |
| **News reasoning + few-shot** | Qwen2.5-14B | 2022–23 | 0.560 |
| **News reasoning + few-shot** | **Qwen2.5-32B** | **2022–23** | **0.566** |
| News reasoning + few-shot | Qwen2.5-32B | **2024** (new regime) | **0.580** |
| News reasoning + few-shot | Qwen2.5-14B | 2024 | 0.376 ⚠️ |
| + price-momentum ensemble | 32B | 2022–23 | 0.576 |
| + Fear & Greed ensemble | 32B | 2022–23 | **0.653** |
| Social-post reasoning (Reddit) | 32B | 2022 | 0.475–0.489 ✗ |

Baselines: news-volume 0.458, price-momentum 0.550, base rate 0.107.

---

## What we learned

### 1. Few-shot prompting is the key lever
Zero-shot, the LLM anchors to a single probability for every day (crash-day mean
≈ non-crash mean ≈ 0.157) → **AUROC 0.505 (chance)**. Adding 3 worked exemplars
(no-crash / contained-stress / contagion) and telling it the ~13% base rate broke
the flatline → **0.566**. This was a far bigger gain than model size or
hyperparameters.

### 2. News reasoning generalizes across regimes — but only with a big model
The 32B model scores **0.566** on 2022–23 (bear market) and **0.580** on 2024
(ETF/halving bull run) — the signal holds out-of-regime. The 14B model is
comparable in-sample (0.560) but **collapses to 0.376 (below chance) on 2024**.
**Model scale buys robustness, not just in-sample accuracy.**

### 3. Memory/attention: slow decay + wide focus wins
Ablation (few-shot held fixed): `lam=0.6, top_k=30` beat `lam=0.9, top_k=15` for
both models (32B: 0.566 vs 0.538; 14B: 0.560 vs 0.545). Aggressive recency and
tight pruning hurt. Over-stuffing `max_items` past the input-token cap starved
the impact graph (edges dropped to ~1.5/day) and lowered AUROC.

### 4. Aggregate sentiment helps — but it's regime-dependent
The **Fear & Greed index** (crypto sentiment, partly social) is the single
strongest signal on the full 2022–23 window (fear level alone = 0.646; ensembled
with news reasoning = **0.653**). But on **2022 alone it falls to 0.488** — in a
relentless bear market, fear is constantly high and stops discriminating. The
0.653 lift is largely a **2023 effect**. Honest caveat: F&G is a *composite*
(volatility, momentum, social media, dominance), so part of its power is not
purely social.

### 5. Reasoning over social *posts* does NOT help
Feeding the top-15 engagement Reddit posts/day into the LLM (social-only 0.489,
news+social 0.475 on 2022) **underperformed news-only (0.524)**. Reddit titles
are noisy (memes, price chatter, shilling) and dilute the systemic-event signal
that news headlines carry. **Aggregate** social sentiment helps; social-post
*reasoning* does not.

### Best result
**News temporal-relational reasoning + Fear & Greed sentiment ensemble = AUROC
0.653** on 2022–23 — a real, honest signal well above price-only (0.55), lexicon
(0.46), and base-rate (0.50) baselines. The reasoning component generalizes to an
unseen 2024 regime (0.58) at 32B scale.

---

## Data

| Source | Coverage | Volume | Role |
|---|---|---|---|
| `oliviervha/crypto-news` | 2021-10 → 2023-12 | 30.5k headlines (~43/day) | main news corpus |
| `filipemunizz/bitcoin-news` | → 2024-10 | 5.8k headlines (2024) | 2024 regime test |
| `leukipp/reddit-crypto-data` | 2022 | 940k posts, 50 subreddits | social reasoning |
| Fear & Greed index (alternative.me) | 2018 → 2026 | daily | sentiment ensemble |
| eth-alpha 5-min OHLCV | 2022-01 → 2026-03 | 6 assets | crash labels |

**Evaluation sizes:** 2022–23 = 712 days / 76 crashes; 2024 = 284 days / 19
crashes; 2022-only = 363 days / 63 crashes.

## Honest limitations
- **Small positive counts** (19–76 crashes) → AUROC has real variance; only the
  large gaps (few-shot +0.06, sentiment +0.09) are clearly meaningful, not the
  0.566-vs-0.560 differences.
- **News cap**: the brainstorm uses ≤ 20–24 of ~43 headlines/day; raising the cap
  (with a larger input budget) is an unexplored lever.
- **No 2025**: no dated crypto-news headline corpus covering 2025 exists on
  Kaggle, so the out-of-sample test stops at Oct 2024.
- **Crash *timing* is intrinsically hard**: ~0.57 from news alone is a modest but
  genuine edge, consistent with the paper needing its full machinery.

## Reproduce
```bash
# Offline LLM runs (Kaggle RTX 6000 Pro): kaggle/trr_standalone.py + deploy_trr.sh
# Local pipeline + analysis:
make trr-labels        # crash labels (FTX/LUNA appear as worst drawdowns)
make trr-eval          # TRR vs baselines (MockLLM harness)
python -m pytest tests/test_trr.py
```
Ablation variants are in `kaggle/` (`exp1_14b.py`, `exp2_32b.py`, `exp3_14b.py`,
`exp2024_*.py`, `social_*_32b.py`); each kernel writes `eval_results.json` +
`trr_predictions.csv` + a timeline plot.
