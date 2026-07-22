#!/bin/bash
#SBATCH --account=MST115278
#SBATCH --job-name=mamba3_test
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=/home/m314510193/GithubMamba3Train/mamba3_compare_nano4.log

module load miniconda3/26.1.1
module load cuda/13.0
conda activate mamba3
export PATH="$CONDA_PREFIX/bin:$PATH"

python /home/m314510193/GithubMamba3Train/mamba3_compare_nano4/gen_shared_data.py
python /home/m314510193/GithubMamba3Train/mamba3_compare_nano4/run_side.py --tag nano
