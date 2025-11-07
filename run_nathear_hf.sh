#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn24
#SBATCH --time=00:05:00
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --array=0



cd ~/phd/gitpull/wavjepa
module load 2023
module load Anaconda3/2023.07-2
source activate sjepa-eval
cd listen-eval-kit


task_name=esc50-v2.0.0-full
tasks_dir=/projects/0/prjs1338/tasks_noisy_ambisonics

embeddings_dir=/projects/0/prjs1338/NatEmbeddingsWavJEPAhf
score_dir=nathear_wavjepa_hf

model_name=hear_configs.WavJEPA_huggingface

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir
python3 -m heareval.predictions.runner $embeddings_dir/$model_name/$task_name

rm -r -d -f $embeddings_dir/$model_name/$task_name