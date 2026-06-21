"""Merge the corpus shard predictions into full 2016-2023 series and score.

Each shard wrote trr_predictions.csv (day, crash_prob, n_news, ..., label_true).
We concatenate all base shards (cb*) into one series and all RAG shards (cr*)
into another, then compute AUROC / PR-AUC over the WHOLE window — the headline
numbers for the full-corpus backtest, plus the news-volume baseline and the
base-vs-RAG delta.

Run: python kaggle/eval_corpus.py
"""
import glob
import json
import os

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kaggle/out_corpus")


def _merge(prefix: str) -> pd.DataFrame:
    frames = []
    for d in sorted(glob.glob(os.path.join(OUT, f"{prefix}*"))):
        f = os.path.join(d, "trr_predictions.csv")
        if os.path.exists(f):
            frames.append(pd.read_csv(f))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="day").sort_values("day").set_index("day")
    return df


def _score(df: pd.DataFrame, label="label_true", score="crash_prob") -> dict:
    if df.empty or label not in df or df[label].nunique() < 2:
        return {"n_days": int(len(df)), "n_pos": int(df.get(label, pd.Series()).sum()),
                "note": "insufficient data / single class"}
    y = df[label].to_numpy()
    out = {"n_days": int(len(df)), "n_pos": int(y.sum()),
           "base_rate": round(float(y.mean()), 4),
           "auroc": round(float(roc_auc_score(y, df[score])), 4),
           "pr_auc": round(float(average_precision_score(y, df[score])), 4)}
    if "n_news" in df and df["n_news"].nunique() > 1:
        out["news_volume_auroc"] = round(float(roc_auc_score(y, df["n_news"])), 4)
    return out


def main():
    base, rag = _merge("cb"), _merge("cr")
    res = {
        "window": {"start": str(base.index.min() if not base.empty else None),
                   "end": str(base.index.max() if not base.empty else None)},
        "shards_found": {"base": int(len(glob.glob(os.path.join(OUT, "cb*")))),
                         "rag": int(len(glob.glob(os.path.join(OUT, "cr*"))))},
        "base": _score(base),
        "rag": _score(rag),
    }
    if "auroc" in res["base"] and "auroc" in res["rag"]:
        res["rag_minus_base_auroc"] = round(res["rag"]["auroc"] - res["base"]["auroc"], 4)
    print(json.dumps(res, indent=2))
    json.dump(res, open(os.path.join(OUT, "corpus_eval.json"), "w"), indent=2)
    print(f"\n-> {os.path.join(OUT, 'corpus_eval.json')}")


if __name__ == "__main__":
    main()
