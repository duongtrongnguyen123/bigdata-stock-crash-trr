#!/usr/bin/env bash
A2=$(awk 'NR==2{print $2}' /home/nduong/dev/bigdata/accounts.txt)
A3=$(awk 'NR==3{print $2}' /home/nduong/dev/bigdata/accounts.txt)
A3U=$(awk 'NR==3{print $1}' /home/nduong/dev/bigdata/accounts.txt)
K=/tmp/bdval/bin/kaggle; cd /home/nduong/dev/bigdata
while KAGGLE_API_TOKEN="$A3" $K kernels status $A3U/crypto-trr-sc-qwen-k3 2>&1 | grep -qiE "RUNNING|QUEUED|PENDING"; do sleep 120; done
while KAGGLE_API_TOKEN="$A2" $K kernels status zhongzhing/crypto-trr-sc-qwen-k5 2>&1 | grep -qiE "RUNNING|QUEUED|PENDING"; do sleep 120; done
echo "=== K-sweep terminal (K=1 ref = 0.524) ==="
for spec in "k3:$A3:$A3U/crypto-trr-sc-qwen-k3" "k5:$A2:zhongzhing/crypto-trr-sc-qwen-k5"; do
  m=${spec%%:*}; rest=${spec#*:}; tok=${rest%%:*}; slug=${rest#*:}
  od=kaggle/out_$m; rm -rf $od; mkdir -p $od
  KAGGLE_API_TOKEN="$tok" $K kernels output $slug -p $od >/dev/null 2>&1
  /tmp/bdval/bin/python - "$m" "$od" <<'PY'
import sys,pandas as pd,numpy as np
from sklearn.metrics import roc_auc_score
from trr.labels import crash_labels
m,od=sys.argv[1],sys.argv[2]
try:
    df=pd.read_csv(f'{od}/trr_predictions.csv',index_col=0); df.index=pd.to_datetime(df.index).date
    lab=crash_labels(horizon=3); lab.index=pd.to_datetime(lab.index).date
    y=np.array([int(lab['crash'].get(d,0)) for d in df.index])
    print('  %s: AUROC=%.3f edges=%.1f crash %.3f vs non %.3f'%(m,roc_auc_score(y,df['crash_prob']),df['n_edges'].mean(),df[y==1].crash_prob.mean(),df[y==0].crash_prob.mean()))
except Exception as e: print('  %s: no preds (%s)'%(m,e))
PY
done
