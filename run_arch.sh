#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=05:00:00
#SBATCH --output=arch_eval_out/slurm_output_%A_%a.out
#SBATCH --array=8-11

cd ~/phd/wavjepa/ARCH
module load 2023
module load Anaconda3/2023.07-2
source activate sjepa-eval

tasks=(esc50 us8k fsd50k vivae fma_small magna_tag_a_tune irmas medleydb ravdess audio_mnist slurp emovo)

task_name=${tasks[$SLURM_ARRAY_TASK_ID]}

weights=/gpfs/work4/0/prjs1338/saved_models_jepa_denoised/Data=LibriSpeech/Extractor=wavjepa/InSeconds=2.01/BatchSize=32/NrSamples=8/NrGPUs=2/LR=0.0001/Alpha=0.0/step=5000.ckpt

python3 evaluate_wavjepa_model.py --weights $weights --device cuda --max_epochs 200 --verbose --tsv_logging_file results/wavjepa_time_full.tsv --n_iters 1 --data_config_file configs/datasets_config.json --enabled_datasets $task_name --precompute_embeddings