#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=02:00:00
#SBATCH --output=steps/slurm_output_%A_%a.out
#SBATCH --array=0-1

cd ~/phd/wavjepa
module load 2023
module load Anaconda3/2023.07-2
source activate sjepa-eval
cd listen-eval-kit


grids=(default
default
)
task_names=(
esc50-v2.0.0-full
vox_lingua_top10-hear2021-full)

tasks_dirs=(
/projects/0/prjs1338/tasks
/projects/0/prjs1338/tasks

)

task_index=$((SLURM_ARRAY_TASK_ID))

task_name=${task_names[$task_index]}
grid=${grids[$task_index]}
tasks_dir=${tasks_dirs[$task_index]}

embeddings_dir=/projects/0/prjs1338/JepaEmbeddings
score_dir=hear_wavjepa

model_name=hear_configs.WavJEPA
sr=16000
model_size=base

weights=/gpfs/work4/0/prjs1338/saved_models_jepa_mixed/InChannels=1/WithNoise=False/WithRIR=False/CleanRatio=1.0/Extractor=conv-extractor/ShareWeights=False/SR=16000/BatchSize=32/NrSamples=8/NrGPUs=2/ModelSize=base/LR=0.0004/Masking=time-inverse-masker/TargetProb=0.25/TargetLen=5/ContextLen=10/TopK=8/step=110000.ckpt

model_options="{\"sr\": \"$sr\", \"model\": \"$model_size\"}"

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task $task_name --embeddings-dir $embeddings_dir --model "$weights" --model-options "$model_options"
python3 -m heareval.predictions.runner $embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name --grid $grid

mkdir -p /projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name

mv $embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/test.predicted-scores.json  /projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name
mv $embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/*predictions.pkl /projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name
mv $embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/*embeddings.npy /projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name

rm -r -d -f $embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name