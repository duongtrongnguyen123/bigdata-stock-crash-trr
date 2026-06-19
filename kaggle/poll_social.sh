#!/usr/bin/env bash
export KAGGLE_API_TOKEN=$(awk 'NR==1{print $2}' accounts.txt)
K=/tmp/bdval/bin/kaggle
for v in only combined; do
  while $K kernels status nguyenduongtrong/crypto-trr-social-$v 2>&1 | grep -qiE "RUNNING|QUEUED|PENDING"; do sleep 60; done
done
echo "=== social runs terminal ==="
/tmp/bdval/bin/python - <<'PY'
import json,pandas as pd,numpy as np,subprocess,os
from datetime import datetime,timezone
from sklearn.metrics import roc_auc_score
from trr.labels import crash_labels
K="/tmp/bdval/bin/kaggle"
fng=json.load(open('data/fng.json'))['data']
f=pd.Series({datetime.fromtimestamp(int(x['timestamp']),timezone.utc).date():int(x['value']) for x in fng})
L=crash_labels(horizon=3); L.index=pd.to_datetime(L.index).date
for v in ['only','combined']:
    od=f'kaggle/out_social_{v}'; os.system(f'rm -rf {od}; mkdir -p {od}')
    os.system(f'{K} kernels output nguyenduongtrong/crypto-trr-social-{v} -p {od} >/dev/null 2>&1')
    try:
        df=pd.read_csv(f'{od}/trr_predictions.csv',index_col=0); df.index=pd.to_datetime(df.index).date
        y=np.array([int(L['crash'].get(d,0)) for d in df.index])
        au=roc_auc_score(y,df['crash_prob'])
        ff=(100-f.reindex(df.index).ffill()).values
        ens=max(roc_auc_score(y, a*((df['crash_prob']-df['crash_prob'].min())/(df['crash_prob'].max()-df['crash_prob'].min()+1e-9)).values + (1-a)*((100-f.reindex(df.index).ffill())/100).values) for a in np.linspace(0,1,11))
        print('social-%s: AUROC=%.3f | days=%d crashes=%d | crash %.3f vs non %.3f | edges %.1f | +F&G ens=%.3f'%(
          v,au,len(df),y.sum(),df[y==1].crash_prob.mean(),df[y==0].crash_prob.mean(),df['n_edges'].mean(),ens))
    except Exception as e: print('social-%s: (no metrics: %s)'%(v,e))
print('baselines (2022): news-only=0.524  F&G=0.488')
PY
