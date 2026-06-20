#!/usr/bin/env bash
# Poll the 4 parallel stock-TRR kernels across 4 accounts (lambda sweep).
# Swaps ~/.kaggle/access_token per account each check; restores account 1 at rest.
# When ALL are terminal, pulls each kernel's output into kaggle/out_<tag>/ and exits.
set -u
cd /home/nduong/dev/bigdata
KG=.venv/bin/kaggle
ACC=accounts.txt

# lineno : slug : tag : lambda
JOBS=(
  "1:nguyenduongtrong/stock-trr-32b-rtx-6000-pro:main:0.6"
  "2:zhongzhing/stock-trr-lam03:03:0.3"
  "3:hduong/stock-trr-lam10:10:1.0"
  "4:truongdinhduc06/stock-trr-lam02:02:0.2"
)

setauth() { local n=$1 u k; u=$(awk -v x=$n 'NR==x{print $1}' $ACC); k=$(awk -v x=$n 'NR==x{print $2}' $ACC)
  printf '%s' "$k" > ~/.kaggle/access_token
  printf '{"username":"%s","key":"%s"}\n' "$u" "$k" > ~/.kaggle/kaggle.json; }
restore1() { cp /tmp/acct1_token ~/.kaggle/access_token; cp /tmp/acct1_kaggle.json ~/.kaggle/kaggle.json; }

for cycle in $(seq 1 70); do
  running=0
  for j in "${JOBS[@]}"; do
    IFS=: read -r n slug tag lam <<< "$j"
    setauth "$n"
    st=$($KG kernels status "$slug" 2>&1 | tail -1)
    echo "[cyc $cycle][lam $lam][$tag] $st"
    case "$st" in *RUNNING*|*QUEUED*) running=$((running+1));; esac
  done
  echo "--- cycle $cycle: $running still running ---"
  if [ "$running" -eq 0 ]; then
    echo "=== ALL DONE — fetching outputs ==="
    for j in "${JOBS[@]}"; do
      IFS=: read -r n slug tag lam <<< "$j"
      setauth "$n"
      out="kaggle/out_${tag}"; rm -rf "$out"; mkdir -p "$out"
      $KG kernels output "$slug" -p "$out" 2>&1 | tail -1
      echo "[$tag lam=$lam] $(ls $out 2>/dev/null | tr '\n' ' ')"
    done
    restore1
    echo "=== FETCH COMPLETE ==="
    break
  fi
  restore1
  sleep 150
done
