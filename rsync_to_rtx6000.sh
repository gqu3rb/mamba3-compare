#!/bin/bash
# Run on H200, from ~/GithubMamba3Train/mamba3_compare_nano4
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

REMOTE=ZhangHengHao@140.116.246.191
REMOTE_DIR=/home/ZhangHengHao/GithubMamba3Train/mamba3_compare_rtx6000
ARTIFACTS=(shared_data.pt results_nano.pt)   # the only things that bypass git

# --- 1. code: H200 -> GitHub -------------------------------------------
if ! git diff --quiet HEAD; then
    echo "ERROR: uncommitted changes in $(pwd) -- commit them first:" >&2
    git status --short >&2
    exit 1
fi
git pull --ff-only origin main      # pick up anything pushed from RTX6000
git push origin main

# --- 2. artifacts: H200 -> RTX6000, explicit list, never a wildcard ----
for f in "${ARTIFACTS[@]}"; do
    [[ -f $f ]] || { echo "ERROR: missing artifact $f" >&2; exit 1; }
done
rsync -avh --partial --inplace "${ARTIFACTS[@]}" "$REMOTE:$REMOTE_DIR/"

# --- 3. code: GitHub -> RTX6000 ----------------------------------------
ssh "$REMOTE" "set -e
    cd $REMOTE_DIR
    git fetch origin
    git pull --ff-only origin main"
