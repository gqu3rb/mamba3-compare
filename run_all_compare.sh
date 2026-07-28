#!/bin/bash
# Enable strict error handling for the local H200 script
set -e

RTX6000_HOST="Zhanghenghao@140.116.246.191"
LOG_FILE="$(pwd)/rtx6000_execution_$(date +%Y%m%d_%H%M%S).log"

echo "========================================"
echo "[1/2] Executing tasks on H200"
echo "========================================"

cd ~/GithubMamba3Train/RTX6000_mamba3
# Check git command and terminate immediately on failure
git checkout nano || { echo "Error: git checkout nano failed on H200. Aborting."; exit 1; }
cd ..

echo ">> Submitting sbatch job and waiting for completion..."
sbatch --wait ~/GithubMamba3Train/mamba3_compare_nano4/mamba3_run_ref.sh

echo ">> sbatch job finished. Syncing files to RTX6000..."
cd ~/GithubMamba3Train/mamba3_compare_nano4

# Disable 'set -e' temporarily for the if-condition to catch the rsync failure properly
set +e
bash rsync_to_rtx6000.sh
RSYNC_EXIT_CODE=$?
set -e

if [ $RSYNC_EXIT_CODE -ne 0 ]; then
    echo ">> rsync_to_rtx6000.sh failed."
    exit 1
fi

echo "========================================"
echo "[2/2] Executing tasks on RTX6000"
echo "========================================"
echo ">> Streaming RTX6000 output below (also saving to $LOG_FILE) :"

# Execute commands on RTX6000 via SSH.
# 1. 'set -e' is passed into the Here-Doc so the remote session also terminates on errors.
# 2. 'git checkout' is explicitly checked on the remote side.
# 3. Output is piped to 'tee' on the H200 side, displaying it on your screen and saving it locally.
ssh "$RTX6000_HOST" << 'EOF' | tee "$LOG_FILE"
set -e

cd ~
source setMamba3env.sh

cd ~/GithubMamba3Train/mamba
git checkout rtx6000-adapt || { echo "Error: git checkout rtx6000-adapt failed on RTX6000. Aborting."; exit 1; }

cd ~/GithubMamba3Train/mamba3_compare_rtx6000
source mamba3_run_rtx6000.sh
EOF

echo "========================================"
echo "Compare completed successfully."
echo "========================================"
