#!/usr/bin/env bash
# Idempotent SERIAL launcher for the 5 FNSPID shards (2021H2-2023, the bear market).
# One account per shard; skips shards already RUNNING/QUEUED/COMPLETE.
# MUST be the only process touching ~/.kaggle while it runs.
set -u
cd /home/nduong/dev/bigdata
KG=.venv/bin/kaggle
ACC=accounts.txt

# tag:lineno (valid spare accounts 18-22)
MAP="f1:18 f2:19 f3:20 f4:21 f5:22"

setauth(){ local n=$1 u k; u=$(awk -v x=$n 'NR==x{print $1}' $ACC); k=$(awk -v x=$n 'NR==x{print $2}' $ACC)
  printf '%s' "$k" > ~/.kaggle/access_token
  printf '{"username":"%s","key":"%s"}\n' "$u" "$k" > ~/.kaggle/kaggle.json; echo "$u"; }

for m in $MAP; do
  tag=${m%:*}; n=${m#*:}; u=$(setauth $n)
  if $KG datasets list --mine 2>&1 | head -1 | grep -qiE "Save the token|401|403"; then
    echo "[$tag/$u] TOKEN INVALID — SKIP"; continue; fi
  st=$($KG kernels status "$u/fnspid-$tag" 2>&1 | tail -1)
  case "$st" in *RUNNING*|*QUEUED*|*COMPLETE*) echo "[$tag/$u] already $st — SKIP"; continue;; esac
  dd="kaggle/fds_$tag"; rm -rf "$dd"; cp -r kaggle/build_fnspid "$dd"
  printf '{"title":"fnspid %s","id":"%s/fnspid-bundle","licenses":[{"name":"other"}],"isPrivate":true}\n' "$tag" "$u" > "$dd/dataset-metadata.json"
  if ! $KG datasets files "$u/fnspid-bundle" 2>/dev/null | grep -q stocknews; then
    $KG datasets create -p "$dd" --dir-mode zip >/dev/null 2>&1
    for w in $(seq 1 16); do sleep 7; $KG datasets files "$u/fnspid-bundle" 2>/dev/null | grep -q stocknews && break; done
  fi
  pd="kaggle/pf_$tag"; rm -rf "$pd"; mkdir -p "$pd"; cp kaggle/shard_${tag}.py "$pd/shard_${tag}.py"
  cat > "$pd/kernel-metadata.json" <<JSON
{"id":"$u/fnspid-$tag","title":"fnspid-$tag","code_file":"shard_${tag}.py","language":"python","kernel_type":"script","is_private":true,"enable_gpu":true,"enable_tpu":false,"enable_internet":false,"machine_shape":"NvidiaRtxPro6000","competition_sources":["nvidia-nemotron-model-reasoning-challenge"],"dataset_sources":["$u/fnspid-bundle"],"kernel_sources":[],"model_sources":["qwen-lm/qwen2.5/transformers/32b-instruct/1"]}
JSON
  echo "[$tag/$u] $($KG kernels push -p "$pd" 2>&1 | tail -1)"
done
cp /tmp/acct1_token ~/.kaggle/access_token; cp /tmp/acct1_kaggle.json ~/.kaggle/kaggle.json
echo "DONE restored: $($KG config view 2>&1 | grep -i username | awk '{print $NF}')"
