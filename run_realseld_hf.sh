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


SLURM_ARRAY_TASK_ID=0

task_names=(tau2018-ov1-v1.0.0-full
tau2018-ov2-v1.0.0-full
tau2018-ov3-v1.0.0-full
tau2019-v1.0.0-full
tau2020-v1.0.0-full
tau2021-v1.0.0-full
starss23-v1.0.0-full)

tasks_dirs=(
/projects/0/prjs1338/realsed
/projects/0/prjs1338/realsed
/projects/0/prjs1338/realsed
/projects/0/prjs1338/realsed
/projects/0/prjs1338/realsed
/projects/0/prjs1338/realsed
/projects/0/prjs1338/realsed
)

task_index=$((SLURM_ARRAY_TASK_ID))

task_name=${task_names[$task_index]}
tasks_dir=${tasks_dirs[$task_index]}

embeddings_dir=/projects/0/prjs1338/JepaEmbeddings1
score_dir=realsed_wavjepa

model_name=hear_configs.WavJEPA_huggingface
model_size=base


model_options="{\"model\": \"$model_size\"}"

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task $task_name --embeddings-dir $embeddings_dir --model-options "$model_options"
python3 -m heareval.predictions.runner $embeddings_dir/$model_name-model=$model_size/$task_name --grid fast

mkdir -p /projects/0/prjs1338/$score_dir/$model_name-$model_name-model=$model_size/$task_name

mv $embeddings_dir/$model_name-model=$model_size/$task_name/test.predicted-scores.json  /projects/0/prjs1338/$score_dir/$model_name-model=$model_size/$task_name
mv $embeddings_dir/$model_name-model=$model_size/$task_name/*predictions.pkl /projects/0/prjs1338/$score_dir/$model_name-model=$model_size/$task_name
mv $embeddings_dir/$model_name-model=$model_size/$task_name/*embeddings.npy /projects/0/prjs1338/$score_dir/$model_name-model=$model_size/$task_name

rm -r -d -f $embeddings_dir/$model_name-model=$model_size/$task_name