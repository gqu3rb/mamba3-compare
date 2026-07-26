#!/bin/bash
#SBATCH --account=MST115278
#SBATCH --job-name=mamba3_test
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=/home/m314510193/GithubMamba3Train/mamba3_compare_nano4.log

set -euo pipefail

module load miniconda3/26.1.1
module load cuda/13.0
conda activate mamba3
export PATH="$CONDA_PREFIX/bin:$PATH"

# This env installs mamba_ssm as a *copy* (plain `pip install .` -- site-packages/mamba_ssm
# is a real directory, and direct_url.json says "dir_info": {}), so edits to
# $MAMBA_REPO/mamba_ssm/modules/mamba3.py are invisible to `import mamba_ssm`.
# PYTHONPATH is inserted into sys.path ahead of site-packages, making the git tree win.
export MAMBA_REPO=/home/m314510193/GithubMamba3Train/RTX6000_mamba3
export PYTHONPATH="$MAMBA_REPO${PYTHONPATH:+:$PYTHONPATH}"

# Fail loudly rather than silently measuring the stale installed copy.
python - "$MAMBA_REPO" <<'EOF'
import sys, mamba_ssm
expected = sys.argv[1] + "/mamba_ssm/__init__.py"
print("mamba_ssm ->", mamba_ssm.__file__, flush=True)
if mamba_ssm.__file__ != expected:
    sys.exit(f"ERROR: PYTHONPATH shim did not take effect (expected {expected})")
EOF

python /home/m314510193/GithubMamba3Train/mamba3_compare_nano4/gen_shared_data.py
python /home/m314510193/GithubMamba3Train/mamba3_compare_nano4/run_side.py --tag nano
