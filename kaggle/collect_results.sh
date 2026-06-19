#!/usr/bin/env bash
K=/tmp/bdval/bin/kaggle
A1=$(awk 'NR==1{print $2}' accounts.txt)
A2=$(awk 'NR==2{print $2}' accounts.txt)
# kernel|token|outdir
EXPS=(
  "nguyenduongtrong/crypto-trr-qwen|$A1|out_v4"
  "nguyenduongtrong/crypto-trr-exp1-14b|$A1|out_exp1"
  "zhongzhing/crypto-trr-exp2|$A2|out_exp2"
  "zhongzhing/crypto-trr-exp3|$A2|out_exp3"
)
running=1
while [ $running -eq 1 ]; do
  running=0
  for e in "${EXPS[@]}"; do
    IFS='|' read -r kid tok _ <<< "$e"
    st=$(KAGGLE_API_TOKEN="$tok" $K kernels status "$kid" 2>&1 | tail -1)
    echo "$kid -> $st" | grep -oE "crypto-trr.*(RUNNING|QUEUED|COMPLETE|ERROR|PENDING)" 
    echo "$st" | grep -qiE "RUNNING|QUEUED|PENDING" && running=1
  done
  [ $running -eq 1 ] && sleep 90
done
echo "=== ALL TERMINAL — collecting ==="
for e in "${EXPS[@]}"; do
  IFS='|' read -r kid tok od <<< "$e"
  rm -rf "kaggle/$od"; mkdir -p "kaggle/$od"
  KAGGLE_API_TOKEN="$tok" $K kernels output "$kid" -p "kaggle/$od" >/dev/null 2>&1
  echo "### $kid"
  cat "kaggle/$od/eval_results.json" 2>/dev/null || echo "(no metrics)"
  echo
done
