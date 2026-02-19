#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=08:00:00
#SBATCH --output=xares/slurm_output_%A_%a.out
#SBATCH --array=0-19

cd /projects/0/prjs1261/xares_gyuksel3/xares

tasks=(
asvspoof_task.py
clotho_task.py 
crema_d_task.py
desed_task.py
esc50_task.py
fluentspeechcommands_kws_task.py
freemusicarchive_genre_task.py
fsd50k_task.py
fsdkaggle2018_task.py
gtzan_task.py
libricount_task.py
librispeech_male_female_task.py
nsynth_instument_task.py
ravdess_task.py
speechcommandsv1_task.py
urbansound8k_task.py
vocalimitations_task.py
vocalsound_task.py
voxceleb1_task.py
voxlingua33_task.py
)


module load 2023
module load Anaconda3/2023.07-2
source activate xares

task=${tasks[$SLURM_ARRAY_TASK_ID]}

python -m xares.run --max-jobs 1 example/wavjepa_ls/wavjepa_encoder.py src/tasks/$task
