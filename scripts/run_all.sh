#!/usr/bin/env bash
# Reproduce the full TRR analysis stack from the committed prediction CSVs +
# prices. Does NOT require GPUs or Kaggle (those produced kaggle/out_*/). Runs
# the training, ablations, threshold, backtest, figures, model export, and tests.
#
# Usage: bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python

echo "==> [1/7] feature table";        $PY -m train.features   | tail -3
echo "==> [2/7] training (does training help?)"; $PY -m train.run | tail -6
echo "==> [3/7] ablations (fairness, calibration, P@K)"; $PY -m train.ablations | tail -6
echo "==> [4/7] threshold / confusion"; $PY -m train.threshold | tail -4
echo "==> [5/7] economic backtest";     $PY -m train.backtest  | tail -4
echo "==> [6/7] export model + figures"; $PY -m train.export | tail -1; $PY -m train.figures | tail -1
echo "==> [7/7] test suite";            $PY -m pytest tests/ serving/tests/ -q | tail -3
echo "==> done. Results in reports/ (RESULTS_TRR.md, *_results.md, figures/)."
