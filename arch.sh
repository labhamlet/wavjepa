#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=05:00:00
#SBATCH --output=arch_eval_out/slurm_output_%A_%a.out
#SBATCH --array=0-11

cd ~/phd/wavjepa/ARCH
module load 2023
module load Anaconda3/2023.07-2
source activate sjepa-eval

SLURM_ARRAY_TASK_ID=3
tasks=(esc50 us8k fsd50k vivae fma_small magna_tag_a_tune irmas medleydb ravdess audio_mnist slurp emovo)

task_idx=$((SLURM_ARRAY_TASK_ID))
task_name=${tasks[$task_idx]}


weights=/gpfs/work4/0/prjs1338/saved_models_jepa_denoised_l2/InChannels=1/WithNoise=True/WithRIR=True/SNRl=-5/SNRh=5/CleanRatio=0.0/SR=16000/alpha=0.0/BatchSize=32/NrSamples=8/NrGPUs=2/ModelSize=base/LR=0.0001/Masking=time-inverse-masker/TargetProb=0.25/TargetLen=10/ContextLen=10/TopK=8/step=25000.ckpt

python3 evaluate_wavjepa_model.py --weights $weights --device cuda --max_epochs 200 --verbose --tsv_logging_file results/wavjepa_time_full.tsv --n_iters 1 --data_config_file configs/datasets_config.json --enabled_datasets $task_name --precompute_embeddings