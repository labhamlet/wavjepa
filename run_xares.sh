#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=08:00:00
#SBATCH --output=xares/slurm_output_%A_%a.out
#SBATCH --array=0-2

cd /projects/0/prjs1261/xares_gyuksel3/xares


tasks=(
asvspoof_task.py
voxlingua33_task.py
voxceleb1_task.py
)


module load 2023
module load Anaconda3/2023.07-2
source activate xares

task=${tasks[$SLURM_ARRAY_TASK_ID]}

python -m xares.run --max-jobs 1 example/wavjepa/wavjepa_encoder.py src/tasks/$task
