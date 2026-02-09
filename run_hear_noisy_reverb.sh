#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=02:00:00
#SBATCH --output=hear_noisy_reverb/slurm_output_%A_%a.out
#SBATCH --array=0-5

cd ~/phd/wavjepa
module load 2023
module load Anaconda3/2023.07-2
source activate sjepa-eval
cd listen-eval-kit


# Too big to do it in one go
# speech_commands-v0.0.2-5h
# Use fast grid
# dcase2016_task2-hear2021-full



#Already done:


# Define arrays
task_dirs=(
    libricount-v1.0.0-hear2021-full
)

task_names=(
    -5
    0
    5
    10
    15
    20
)

num_task_names=${#task_names[@]}   # 6
num_task_dirs=${#task_dirs[@]}     # 5

# Calculate indices
task_name_idx=$((SLURM_ARRAY_TASK_ID % num_task_names))
task_dir_idx=$((SLURM_ARRAY_TASK_ID / num_task_names % num_task_dirs))

task_name=${task_names[$task_name_idx]}
task_dir=${task_dirs[$task_dir_idx]}

tasks_dir="/projects/prjs1338/create_noisy_reverb_hear/outputs/$task_dir"

embeddings_dir="/projects/prjs1338/JepaEmbeddingsDenoised/$task_dir"
score_dir="noisy_reverb_hear_wavjepa_robust/$task_dir"

model_name="hear_configs.WavJEPA"
sr=16000
model_size=base

weights=/gpfs/work4/0/prjs1338/saved_models_jepa_denoised_l2/InChannels=1/WithNoise=True/WithRIR=True/SNRl=-5/SNRh=5/CleanRatio=0.0/SR=16000/alpha=0.0/BatchSize=32/NrSamples=8/NrGPUs=2/ModelSize=base/LR=0.0001/Masking=time-inverse-masker/TargetProb=0.25/TargetLen=10/ContextLen=10/TopK=8/step=100000.ckpt
# weights=/gpfs/work5/0/prjs1261/wavjepa_base_final/step=375000.ckpt

model_options="{\"sr\": \"$sr\", \"model\": \"$model_size\"}"

echo "Running job $SLURM_ARRAY_TASK_ID: task_dir=$task_dir, task_name=$task_name"

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir "$tasks_dir" --task "$task_name" --embeddings-dir "$embeddings_dir" --model "$weights" --model-options "$model_options"
python3 -m heareval.predictions.runner "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name"

mkdir -p "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name"

mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/test.predicted-scores.json" "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"
mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/"*predictions.pkl "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"
mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/"*embeddings.npy "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"

rm -rf "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name"