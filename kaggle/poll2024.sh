#!/usr/bin/env bash
export KAGGLE_API_TOKEN=$(awk 'NR==1{print $2}' accounts.txt)
K=/tmp/bdval/bin/kaggle
for m in 32b 14b; do
  while $K kernels status nguyenduongtrong/crypto-trr-2024-$m 2>&1 | grep -qiE "RUNNING|QUEUED|PENDING"; do sleep 60; done
done
echo "=== 2024 runs terminal ==="
for m in 32b 14b; do
  od=kaggle/out_2024_$m; rm -rf $od; mkdir -p $od
  $K kernels output nguyenduongtrong/crypto-trr-2024-$m -p $od >/dev/null 2>&1
  echo "### 2024-$m: $($K kernels status nguyenduongtrong/crypto-trr-2024-$m 2>&1 | grep -oE 'COMPLETE|ERROR')"
  /tmp/bdval/bin/python -c "
import json,pandas as pd
from trr.labels import crash_labels
try:
    m=json.load(open('$od/eval_results.json'))['metrics']['TRR']
    df=pd.read_csv('$od/trr_predictions.csv',index_col=0); df.index=pd.to_datetime(df.index).date
    lab=crash_labels(); lab.index=pd.to_datetime(lab.index).date
    df['c']=[int(lab['crash'].get(d,0)) for d in df.index]
    print('   AUROC=%.3f PR-AUC=%.3f | days=%d crashes=%d | crash %.3f vs non %.3f | edges %.1f'%(
      m['auroc'],m['pr_auc'],len(df),df.c.sum(),df[df.c==1].crash_prob.mean(),df[df.c==0].crash_prob.mean(),df['n_edges'].mean()))
except Exception as e: print('   (no metrics:',e,')')
" 2>/dev/null
done
