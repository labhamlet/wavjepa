#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=00:05:00
#SBATCH --output=hear/slurm_output_%A_%a.out
#SBATCH --array=0

cd ~/phd/wavjepa
module load 2023
module load Anaconda3/2023.07-2
source activate sjepa-eval
cd listen-eval-kit

task_dirs=(
# /projects/0/prjs1338/tasks
# /projects/0/prjs1338/tasks
# /projects/0/prjs1261/tasks
# /projects/0/prjs1338/tasks
# /projects/0/prjs1338/tasks
# /projects/0/prjs1338/tasks
# /projects/0/prjs1338/tasks
# /projects/0/prjs1338/tasks
# /projects/0/prjs1338/tasks
# /projects/0/prjs1338/tasks
# /projects/0/prjs1338/tasks
)

task_names=(
# beijing_opera-v1.0-hear2021-full
# dcase2016_task2-hear2021-full
# fsd50k-v1.0-full
# esc50-v2.0.0-full
# libricount-v1.0.0-hear2021-full
# speech_commands-v0.0.2-5h
# mridangam_stroke-v1.5-full
# mridangam_tonic-v1.5-full
# tfds_crema_d-1.0.0-full
# nsynth_pitch-v2.2.3-5h
# vox_lingua_top10-hear2021-full
)

ratios=(
0.25
0.5
0.75
1.0
)

task_name=vox_lingua_top10-hear2021-full
task_dir=/projects/0/prjs1338/tasks
ratio=${ratios[$SLURM_ARRAY_TASK_ID]}
embeddings_dir="/projects/prjs1261/JepaEmbeddingsm"
score_dir="hear_wavjepa_libri_m"

model_name="hear_configs.WavJEPA"
sr=16000
model_size=base

weights=/gpfs/work5/0/prjs1261/saved_models_jepa_reproduce/SR=16000/LibriRatio=0.0/BatchSize=32/NrSamples=8/NrGPUs=2/ModelSize=base/LR=0.0004/Masking=time-inverse-masker/TargetProb=0.25/TargetLen=10/ContextLen=10/TopK=8/step=25000.ckpt
# weights=/projects/0/prjs1261/wavjepa_base_final/step=$ratio.ckpt
model_options="{\"sr\": \"$sr\", \"model\": \"$model_size\"}"

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir "$task_dir" --task "$task_name" --embeddings-dir "$embeddings_dir" --model "$weights" --model-options "$model_options"
python3 -m heareval.predictions.runner "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name"

mkdir -p "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name"

mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/test.predicted-scores.json" "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"
mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/"*predictions.pkl "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"
mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/"*embeddings.npy "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"

rm -rf "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name"