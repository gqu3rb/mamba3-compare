#!/bin/bash
set -e

REMOTE=ZhangHengHao@140.116.246.191
REMOTE_DIR=/home/ZhangHengHao/GithubMamba3Train/mamba3_compare_rtx6000

# Refresh the git bundle so RTX6000's repo can pull the latest .py/.sh source
git bundle create ../mamba3_compare_nano4.bundle main

# Send the bundle to the same path the `origin` remote already points at
# (set up earlier via: git remote add origin ../mamba3_compare_nano4.bundle)
scp ../mamba3_compare_nano4.bundle "$REMOTE":~/GithubMamba3Train/

# Copy generated artifacts (shared_data.pt, results_*.pt, etc.) as before
scp -r ./* "$REMOTE":"$REMOTE_DIR"/

# Update the remote git working tree's tracked .py/.sh files
ssh "$REMOTE" "cd $REMOTE_DIR && git pull origin main"