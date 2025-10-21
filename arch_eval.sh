#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=03:00:00
#SBATCH --output=arch_eval_out/slurm_output_%A_%a.out
#SBATCH --array=0-83

cd ~/phd/GRAM-JEPA/ARCH
module load 2023
module load Anaconda3/2023.07-2
source activate sjepa-eval

steps=(50000 100000 150000 200000 250000 300000 350000)
tasks=(esc50 us8k fsd50k vivae fma_small magna_tag_a_tune irmas medleydb ravdess audio_mnist slurp emovo)

step_idx=$((SLURM_ARRAY_TASK_ID % 7))
step=${steps[$step_idx]}

task_idx=$((SLURM_ARRAY_TASK_ID / 7))
task_name=${tasks[$task_idx]}


weights=/gpfs/work5/0/prjs1261/saved_models_jepa_scale/InChannels=1/WithNoise=False/WithRIR=False/AugmentProb=1/Extractor=spatial-conv-extractor-removed-last/SR=16000/BatchSize=32/NrSamples=8/ModelSize=base/Masking=time-inverse-masker/MaskProb=0.065/MaskLen=None/Cluster=None/TopK=8/step=$step.ckpt

python3 evaluate_wavjepa_model.py --weights $weights --device cuda --max_epochs 200 --verbose --tsv_logging_file results/wavjepa_time_full.tsv --n_iters 1 --data_config_file configs/datasets_config.json --enabled_datasets $task_name --precompute_embeddings