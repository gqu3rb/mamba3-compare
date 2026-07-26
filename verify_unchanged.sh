#!/bin/bash
#SBATCH --account=MST115278
#SBATCH --job-name=mamba3_verify
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=/home/m314510193/GithubMamba3Train/mamba3_verify_unchanged.log
#
# Regression check: re-run one side and confirm a mamba_ssm change altered no number.
# H200 / `nano` side only (the RTX6000 box is not on SLURM and needs no PYTHONPATH shim).
#
# Usage:
#     cp results_nano.pt my_baseline.pt        # BEFORE making the change
#     ...edit mamba_ssm...                    # do NOT re-run gen_shared_data.py
#     sbatch --wait ./verify_unchanged.sh nano my_baseline.pt
#
# Both runs must use the SAME shared_data.pt; regenerating it changes the weights, so
# every downstream tensor would legitimately differ and the check becomes meaningless.
# That is why this script never calls gen_shared_data.py.
#
# stdout is not watchable from the login node, so each step also writes a file on the
# shared storage:
#   shim_check.log        -- which mamba_ssm actually got imported
#   verify_unchanged.log  -- the per-key comparison result
set -euo pipefail

usage() { echo "usage: sbatch --wait ./verify_unchanged.sh <tag> <baseline.pt> [tol]" >&2; exit 2; }
TAG=${1:-}   ; [[ -n $TAG ]]      || usage
BASELINE=${2:-}; [[ -n $BASELINE ]] || usage
TOL=${3:-0.0}

module load miniconda3/26.1.1
module load cuda/13.0
conda activate mamba3
export PATH="$CONDA_PREFIX/bin:$PATH"

export MAMBA_REPO=/home/m314510193/GithubMamba3Train/RTX6000_mamba3
export PYTHONPATH="$MAMBA_REPO${PYTHONPATH:+:$PYTHONPATH}"

HARNESS=/home/m314510193/GithubMamba3Train/mamba3_compare_nano4
cd "$HARNESS"

[[ -f $BASELINE ]] || { echo "ERROR: baseline not found: $BASELINE" >&2; exit 1; }

# Fail loudly rather than silently measuring the stale site-packages copy.
python - "$MAMBA_REPO" "$HARNESS/shim_check.log" <<'EOF'
import sys, mamba_ssm
from mamba_ssm.utils import tap

repo, log_path = sys.argv[1], sys.argv[2]
expected = repo + "/mamba_ssm/__init__.py"
with open(log_path, "w") as f:
    f.write(f"mamba_ssm -> {mamba_ssm.__file__}\n")
    f.write(f"expected  -> {expected}\n")
    f.write(f"tap.IMPL  -> {tap.IMPL}\n")
    f.write(f"shim_ok   -> {mamba_ssm.__file__ == expected}\n")
if mamba_ssm.__file__ != expected:
    sys.exit(f"ERROR: PYTHONPATH shim did not take effect (expected {expected})")
EOF

python run_side.py --tag "$TAG"
python verify_unchanged.py --baseline "$BASELINE" --new "results_${TAG}.pt" --tol "$TOL"
