#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=Classify
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=slurm_output_%A_%a.out

cd $HOME/phd/listen-eval-kit

module load 2023
module load Anaconda3/2023.07-2
source activate listen-eval


python3 -m pip install heareval hearbaseline