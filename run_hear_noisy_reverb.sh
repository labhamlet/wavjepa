#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=00:10:00
#SBATCH --output=hear_noisy_reverb/slurm_output_%A_%a.out
#SBATCH --array=0-47

cd ~/phd/wavjepa
module load 2023
module load Anaconda3/2023.07-2
source activate sjepa-eval
cd listen-eval-kit

# Define arrays
task_dirs=(
    # esc50-v2.0.0-full
    tfds_crema_d-1.0.0-full
    beijing_opera-v1.0-hear2021-full
)

task_names=(
    -5
    0
    5
    10
    15
    20
)

alphas=(
    0.8
    0.6
    0.4
    0.2
    # 0.0
)

# Calculate total combinations: 2 task_dirs × 6 task_names × 4 alphas = 48 jobs
# Map SLURM_ARRAY_TASK_ID to the three dimensions
num_alphas=${#alphas[@]}           # 5
num_task_names=${#task_names[@]}   # 6
num_task_dirs=${#task_dirs[@]}     # 2

# Calculate indices
alpha_idx=$((SLURM_ARRAY_TASK_ID % num_alphas))
task_name_idx=$(((SLURM_ARRAY_TASK_ID / num_alphas) % num_task_names))
task_dir_idx=$((SLURM_ARRAY_TASK_ID / (num_alphas * num_task_names)))

# Get actual values
alpha=${alphas[$alpha_idx]}
task_name=${task_names[$task_name_idx]}
task_dir=${task_dirs[$task_dir_idx]}

tasks_dir=/projects/prjs1338/create_noisy_reverb_hear/outputs/$task_dir

embeddings_dir="/projects/prjs1338/JepaDenoisedEmbeddingsNoisyReverb$alpha"
score_dir="noisy_reverb_hear_wavjepa_denoised_$alpha"

model_name="hear_configs.WavJEPA"
sr=16000
model_size=base

weights=/gpfs/work4/0/prjs1338/saved_models_jepa_denoised_l2/InChannels=1/WithNoise=True/WithRIR=True/SNRl=-5/SNRh=5/CleanRatio=0.0/SR=16000/alpha=$alpha/BatchSize=32/NrSamples=8/NrGPUs=2/ModelSize=base/LR=0.0001/Masking=time-inverse-masker/TargetProb=0.25/TargetLen=10/ContextLen=10/TopK=8/step=25000.ckpt

model_options="{\"sr\": \"$sr\", \"model\": \"$model_size\"}"

echo "Running job $SLURM_ARRAY_TASK_ID: task_dir=$task_dir, task_name=$task_name, alpha=$alpha"

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir "$tasks_dir" --task "$task_name" --embeddings-dir "$embeddings_dir" --model "$weights" --model-options "$model_options"
python3 -m heareval.predictions.runner "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name"

mkdir -p "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name"

mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/test.predicted-scores.json" "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"
mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/"*predictions.pkl "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"
mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/"*embeddings.npy "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"

rm -rf "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name"