#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=02:00:00
#SBATCH --output=steps/slurm_output_%A_%a.out
#SBATCH --array=0

cd ~/phd/wavjepa
module load 2023
module load Anaconda3/2023.07-2
source activate sjepa-eval
cd listen-eval-kit


# task_names=(
# esc50-v2.0.0-full
# )


task_names=(
0
)



task_name=${task_names[$SLURM_ARRAY_TASK_ID]}
tasks_dir=/projects/prjs1338/create_nat_hear/outputs/esc50-v2.0.0-full
embeddings_dir="/projects/prjs1338/JepaEmbeddings"
score_dir="hear_wavjepa"

model_name="hear_configs.WavJEPA"
sr=16000
model_size=base

# This is clean ratio 0.1
# weights="/projects/prjs1338/saved_models_jepa_real_noise_snr/InChannels=1/WithNoise=True/WithRIR=True/SNRl=5/SNRh=40/CleanRatio=0.1/SR=16000/BatchSize=32/NrSamples=8/NrGPUs=2/ModelSize=base/LR=0.0004/Masking=time-inverse-masker/TargetProb=0.25/TargetLen=10/ContextLen=10/TopK=8/step=175000.ckpt"

#This is clean ratio 0.0 
# weights=/projects/prjs1338/saved_models_jepa_real_new/InChannels=1/WithNoise=True/WithRIR=True/SNRl=5/SNRh=40/CleanRatio=0.0/SR=16000/BatchSize=32/NrSamples=8/NrGPUs=2/ModelSize=base/LR=0.0004/Masking=time-inverse-masker/TargetProb=0.25/TargetLen=10/ContextLen=10/TopK=8/step=150000.ckpt

#This is with the SNR curic
# weights=/projects/prjs1338/saved_models_jepa_real_noise_snr/InChannels=1/WithNoise=True/WithRIR=True/SNRl=0/SNRh=0/CleanRatio=0.0/SR=16000/BatchSize=32/NrSamples=8/NrGPUs=2/ModelSize=base/LR=0.0004/Masking=time-inverse-masker/TargetProb=0.25/TargetLen=10/ContextLen=10/TopK=8/step=175000.ckpt

#This is with aug curic 
weights=/projects/prjs1338/saved_models_jepa_real_aug_curic/InChannels=1/WithNoise=True/WithRIR=True/SNRl=5/SNRh=40/CleanRatio=0.0/SR=16000/BatchSize=32/NrSamples=8/NrGPUs=2/ModelSize=base/LR=0.0004/Masking=time-inverse-masker/TargetProb=0.25/TargetLen=10/ContextLen=10/TopK=8/step=175000.ckpt

# #This is the clean trained baseline
weights=/gpfs/work5/0/prjs1261/wavjepa_base_final/step=175000.ckpt
model_options="{\"sr\": \"$sr\", \"model\": \"$model_size\"}"

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir "$tasks_dir" --task "$task_name" --embeddings-dir "$embeddings_dir" --model "$weights" --model-options "$model_options"
python3 -m heareval.predictions.runner "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name"

mkdir -p "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name"

mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/test.predicted-scores.json" "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"
mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/"*predictions.pkl "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"
mv "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/"*embeddings.npy "/projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name/"

rm -rf "$embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name"