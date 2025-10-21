#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=00:30:00
#SBATCH --output=steps/slurm_output_%A_%a.out
#SBATCH --array=0

cd ~/phd/GRAM-JEPA
module load 2023
module load Anaconda3/2023.07-2
source activate sjepa-eval
cd listen-eval-kit

task_names=(sc_anechoic)

embeddings_dir=/projects/0/prjs1338/JepaEmbedding
score_dir=local_natjepa
tasks_dir=/projects/0/prjs1261/tasks_spatial

model_name=hear_configs.SJEPA_nat
sr=16000
model_size=base
weights=/gpfs/work4/0/prjs1338/saved_models_jepa_naturalistic_mix/InChannels=2/WithNoise=True/WithRIR=True/CleanRatio=0.0/Extractor=spatial-conv-channel-extractor/ShareWeights=False/SR=16000/BatchSize=16/NrSamples=8/NrGPUs=2/ModelSize=base/Masking=time-inverse-masker/MaskProb=0.065/MaskLen=None/Cluster=None/TopK=8/step=370000.ckpt
task_name=${task_names[$SLURM_ARRAY_TASK_ID]}

model_options="{\"sr\": \"$sr\", \"model\": \"$model_size\"}"

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir --model "$weights" --model-options "$model_options"
python3 -m heareval.predictions.runner $embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name --localization cartesian-regression

mkdir -p /projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name

mv $embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/test.predicted-scores.json  /projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name
mv $embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/*predictions.pkl /projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name
mv $embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/*embeddings.npy /projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name

rm -r -d -f $embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name