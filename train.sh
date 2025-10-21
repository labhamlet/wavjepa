#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=JEPATargeLength
#SBATCH --exclude=gcn131
#SBATCH --time=66:00:00
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --array=0

cd ~/phd/GRAM-JEPA
HYDRA_FULL_ERROR=1

module load 2023
module load Anaconda3/2023.07-2
source activate sjape


python3 train.py data=audioset data.clean_data_ratio=1.0 extractor=ConvChannelFeatureExtractor masker.channel_based_masking=True trainer.batch_size=16