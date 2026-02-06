#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=JEPATargeLength
#SBATCH --exclude=gcn131
#SBATCH --time=10:00:00
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --array=0

cd ~/phd/wavjepa
HYDRA_FULL_ERROR=1

module load 2023
module load Anaconda3/2023.07-2
source activate sjape


python3 denoise.py trainer.batch_size=32 trainer.num_gpus=2 trainer.alpha=0.0