#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=02:00:00
#SBATCH --output=evalhear/slurm_output_%A_%a.out
#SBATCH --array=0-10

cd ~/phd/GRAM-JEPA
module load 2023
module load Anaconda3/2023.07-2
source activate sjepa-eval
cd listen-eval-kit


grids=(default
fast
default
default
default
default
default
default
default
default
default
)

task_names=(
fsd50k-v1.0-full
dcase2016_task2-hear2021-full
beijing_opera-v1.0-hear2021-full
esc50-v2.0.0-full
libricount-v1.0.0-hear2021-full
speech_commands-v0.0.2-5h
mridangam_stroke-v1.5-full
mridangam_tonic-v1.5-full
tfds_crema_d-1.0.0-full
nsynth_pitch-v2.2.3-5h
vox_lingua_top10-hear2021-full
)
tasks_dirs=(
/projects/0/prjs1261/tasks
/projects/0/prjs1261/tasks
/projects/0/prjs1338/tasks
/projects/0/prjs1338/tasks
/projects/0/prjs1338/tasks
/projects/0/prjs1338/tasks
/projects/0/prjs1338/tasks
/projects/0/prjs1338/tasks
/projects/0/prjs1338/tasks
/projects/0/prjs1338/tasks
/projects/0/prjs1338/tasks
)

# ratios=(1.0 0.25 0.5 0.75)
# target_idx=$SLURM_ARRAY_TASK_ID
# task_index=0

ratio=1.0
task_name=${task_names[$task_index]}
grid=${grids[$task_index]}
tasks_dir=${tasks_dirs[$task_index]}

embeddings_dir=/projects/0/prjs1338/JepaEmbeddings$ratio
score_dir=hear_wavjepa_step_$ratio

model_name=hear_configs.SJEPA_nat
sr=16000
model_size=base
weights=/gpfs/work4/0/prjs1338/saved_models_jepa_naturalistic_mix/InChannels=2/WithNoise=True/WithRIR=True/CleanRatio=$ratio/Extractor=spatial-conv-channel-extractor/ShareWeights=False/SR=16000/BatchSize=16/NrSamples=8/NrGPUs=2/ModelSize=base/LR=0.0004/Masking=time-inverse-masker/TargetProb=0.25/TargetLen=10/ContextLen=10/TopK=8/step=375000.ckpt

model_options="{\"sr\": \"$sr\", \"model\": \"$model_size\"}"

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task $task_name --embeddings-dir $embeddings_dir --model "$weights" --model-options "$model_options"
python3 -m heareval.predictions.runner $embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name --grid $grid

mkdir -p /projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name

mv $embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/test.predicted-scores.json  /projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name
mv $embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/*predictions.pkl /projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name
mv $embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name/*embeddings.npy /projects/0/prjs1338/$score_dir/$model_name-sr=$sr-model=$model_size/$task_name

rm -r -d -f $embeddings_dir/$model_name-sr=$sr-model=$model_size/$task_name