#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=02:00:00
#SBATCH --output=steps/slurm_output_%A_%a.out
#SBATCH --array=0-29

cd ~/phd/wavjepa
module load 2023
module load Anaconda3/2023.07-2
source activate sjepa-eval
cd listen-eval-kit

steps=(
25000
50000
75000
100000
125000
150000
175000
200000
225000
250000
275000
300000
325000
350000
375000
)

task_names=(
"esc50-v2.0.0-full"
"esc50-v2.0.0-full"
)

tasks_dirs=(
"/projects/0/prjs1338/tasks"
"/projects/0/prjs1338/tasks_noisy_ambisonics"
)

# Calculate which step and which task based on array index
# 15 steps × 2 tasks = 30 total jobs (0-29)
num_steps=${#steps[@]}
step_index=$((SLURM_ARRAY_TASK_ID % num_steps))
task_index=$((SLURM_ARRAY_TASK_ID / num_steps))

step=${steps[$step_index]}
task_name=${task_names[$task_index]}
tasks_dir=${tasks_dirs[$task_index]}

# Construct grids with step included
if [ $task_index -eq 0 ]; then
    embeddings_dir="/projects/0/prjs1338/JepaEmbeddings${step}"
else
    embeddings_dir="/projects/0/prjs1338/JepaEmbeddingsNoisy${step}"
fi

score_dir="hear_wavjepa"

model_name="hear_configs.WavJEPA"
sr=16000
model_size=base

weights="/gpfs/work5/0/prjs1261/wavjepa_base_final/step=${step}.ckpt"

model_options="{\"sr\": \"$sr\", \"model\": \"$model_size\"}"

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir "$tasks_dir" --task "$task_name" --embeddings-dir "$embeddings_dir" --model "$weights" --model-options "$model_options"
python3 -m heareval.predictions.runner "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name"

mkdir -p "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name"

mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/test.predicted-scores.json" "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"
mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/"*predictions.pkl "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"
mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/"*embeddings.npy "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"

rm -rf "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name"