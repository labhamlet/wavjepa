#!/bin/bash
# WavJEPA pre-training on the 16 kHz AudioSet shards, one H100, packed context + torch.compile,
# one crop per clip (no in-batch crop sampling). Checkpoints every CKPT_EVERY steps (for HEAR)
# plus last.ckpt; resume with EXTRA="ckpt_path=<...>/last.ckpt".
#
#   CROP_SECONDS=2.01 BATCH_SIZE=256 sbatch --job-name=wj2s  train_audioset.sbatch
#   CROP_SECONDS=5    BATCH_SIZE=256 sbatch --job-name=wj5s  train_audioset.sbatch
#   CROP_SECONDS=2.01 BATCH_SIZE=256 EXTRACTOR=wav2vec2 sbatch --job-name=wj2sw2v train_audioset.sbatch
#   Env: EXTRACTOR CROP_SECONDS BATCH_SIZE SAMPLES (1) STEPS (375000) CKPT_EVERY (10000) WORKERS (16) SAVE_DIR EXTRA
#
#SBATCH --job-name=wavjepa
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00
#SBATCH --output=benchmarks/logs/train_%x_%j.out

set -o pipefail
cd "${WAVJEPA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load 2023
module load Anaconda3/2023.07-2
source activate sjape
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${TMPDIR:-/tmp}/inductor_$USER}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CROP_SECONDS=${CROP_SECONDS:-2.01} 
BATCH_SIZE=${BATCH_SIZE:-256}
SAMPLES=${SAMPLES:-1}
STEPS=${STEPS:-375000}
CKPT_EVERY=${CKPT_EVERY:-10000}
WORKERS=${WORKERS:-16}
SAVE_DIR=${SAVE_DIR:-/scratch-shared/$USER/wavjepa_runs}
EXTRA=${EXTRA:-}
EXTRACTOR=${EXTRACTOR:-wavjepa}   # wavjepa (10 ms tokens) or wav2vec2 (extra (512,2,2) layer, 20 ms tokens)
DATA_DIRS=/scratch-shared/gyuksel2/audioset_16k/unbal_train{000..869}.tar

mkdir -p "$SAVE_DIR"
echo "node $(hostname), $(date), job $SLURM_JOB_ID ($SLURM_JOB_NAME)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "config: extractor $EXTRACTOR, ${CROP_SECONDS}s crops, batch $BATCH_SIZE clips x $SAMPLES, steps $STEPS, ckpt every $CKPT_EVERY, workers $WORKERS, save_dir $SAVE_DIR, extra '$EXTRA'"

# shellcheck disable=SC2086
python train.py masker=AudioSet data=audioset_16k "extractor=$EXTRACTOR" \
    "data.data_dirs=\"$DATA_DIRS\"" "data.samples_per_audio=$SAMPLES" "data.process_seconds=$CROP_SECONDS" \
    "data.num_workers=$WORKERS" data.prefetch_factor=2 \
    "trainer.batch_size=$BATCH_SIZE" trainer.num_gpus=1 "trainer.steps=$STEPS" \
    "trainer.checkpoint_every_n_steps=$CKPT_EVERY" trainer.use_packing=True trainer.compile_modules=True \
    "save_dir=$SAVE_DIR" $EXTRA
echo "train.py exit=$? $(date)"
