#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=05:00:00
#SBATCH --output=hear_real_world/slurm_output_%A_%a.out
#SBATCH --array=0-1


cd ~/phd/wavjepa
module load 2023
module load Anaconda3/2023.07-2
source activate sjepa-eval
cd listen-eval-kit

task_names=(
"tau2021-v1.0.0-full"
"starss23-v1.0.0-full"
)

tasks_dir="/projects/prjs1338/tasks_real_world"

embeddings_dir="/projects/prjs1338/JepaDenoisedEmbeddingsRealWorld"
score_dir="real_word_wavjepa_denoised"

model_name="hear_configs.WavJEPA"
sr=16000
model_size=base

task_name=${task_names[$SLURM_ARRAY_TASK_ID]}


weights=/gpfs/work4/0/prjs1338/saved_models_jepa_denoised_l2/InChannels=1/WithNoise=True/WithRIR=True/SNRl=-5/SNRh=5/CleanRatio=0.0/SR=16000/alpha=0.0/BatchSize=32/NrSamples=8/NrGPUs=2/ModelSize=base/LR=0.0001/Masking=time-inverse-masker/TargetProb=0.25/TargetLen=10/ContextLen=10/TopK=8/step=95000.ckpt

model_options="{\"sr\": \"$sr\", \"model\": \"$model_size\"}"


python3 -m heareval.embeddings.runner "$model_name" --tasks-dir "$tasks_dir" --task "$task_name" --embeddings-dir "$embeddings_dir" --model "$weights" --model-options "$model_options"
python3 -m heareval.predictions.runner "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name" --grid fast

# mkdir -p "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name"

# mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/test.predicted-scores.json" "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"
# mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/"*predictions.pkl "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"
# mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/"*embeddings.npy "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"

# rm -rf "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name"