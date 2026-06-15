#!/usr/bin/env bash
export KAGGLE_API_TOKEN=$(awk 'NR==1{print $2}' /home/nduong/dev/bigdata/accounts.txt)
K=/tmp/bdval/bin/kaggle; cd /home/nduong/dev/bigdata
slug=crypto-trr-self-consistency-r1
while $K kernels status nguyenduongtrong/$slug 2>&1 | grep -qiE "RUNNING|QUEUED|PENDING"; do sleep 120; done
echo "=== self-consistency terminal: $($K kernels status nguyenduongtrong/$slug 2>&1 | grep -oE 'COMPLETE|ERROR') ==="
rm -rf kaggle/out_selfconsist; mkdir -p kaggle/out_selfconsist
$K kernels output nguyenduongtrong/$slug -p kaggle/out_selfconsist >/dev/null 2>&1
echo "BUILD: $(grep -o 'BUILD=[a-z0-9-]*' kaggle/out_selfconsist/*.log 2>/dev/null | head -1)"
/tmp/bdval/bin/python - <<'PY'
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from trr.labels import crash_labels
try:
    df=pd.read_csv('kaggle/out_selfconsist/trr_predictions.csv',index_col=0); df.index=pd.to_datetime(df.index).date
    lab=crash_labels(horizon=3); lab.index=pd.to_datetime(lab.index).date
    y=np.array([int(lab['crash'].get(d,0)) for d in df.index])
    print('SELF-CONSISTENCY (R1-distill-32B, K=3) 2022: AUROC=%.3f | days=%d crashes=%d | crash %.3f vs non %.3f | edges %.1f'%(
      roc_auc_score(y,df['crash_prob']),len(df),y.sum(),df[y==1].crash_prob.mean(),df[y==0].crash_prob.mean(),df['n_edges'].mean()))
    print('reference (Qwen-32B greedy, 2022 slice): 0.524')
except Exception as e: print('no preds:',e)
PY
