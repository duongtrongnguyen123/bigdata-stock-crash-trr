#!/usr/bin/env bash
# Wait for both per-asset runs (14B + 32B) and compute per-asset AUROC.
export KAGGLE_API_TOKEN=$(awk 'NR==1{print $2}' /home/nduong/dev/bigdata/accounts.txt)
K=/tmp/bdval/bin/kaggle
cd /home/nduong/dev/bigdata
for slug in crypto-trr-per-asset-14b crypto-trr-per-asset; do
  while $K kernels status nguyenduongtrong/$slug 2>&1 | grep -qiE "RUNNING|QUEUED|PENDING"; do sleep 90; done
done
echo "=== both per-asset runs terminal ==="
for tag in "14b:crypto-trr-per-asset-14b" "32b:crypto-trr-per-asset"; do
  m=${tag%%:*}; slug=${tag##*:}
  od="kaggle/out_perasset_$m"; rm -rf "$od"; mkdir -p "$od"
  $K kernels output "nguyenduongtrong/$slug" -p "$od" >/dev/null 2>&1
  state=$($K kernels status "nguyenduongtrong/$slug" 2>&1 | grep -oE 'COMPLETE|ERROR')
  echo "##### PER-ASSET $m ($state)"
  /tmp/bdval/bin/python -m trr.analysis per-asset "$od/trr_predictions.csv" 2>/dev/null || echo "(no preds)"
done
