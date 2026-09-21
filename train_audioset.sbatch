#!/bin/bash
# WavJEPA pre-training on the 16 kHz AudioSet shards, one H100, packed context + torch.compile,
# one crop per clip (no in-batch crop sampling). Checkpoints every CKPT_EVERY steps (for HEAR)
# plus last.ckpt; resume with EXTRA="ckpt_path=<...>/last.ckpt".
#
#   CROP_SECONDS=2.01 BATCH_SIZE=256 sbatch --job-name=wj2s  train_audioset.sbatch
#   CROP_SECONDS=5    BATCH_SIZE=256 sbatch --job-name=wj5s  train_audioset.sbatch
#   CROP_SECONDS=2.01 BATCH_SIZE=256 EXTRACTOR=wav2vec2 sbatch --job-name=wj2sw2v train_audioset.sbatch
#   Env: EXTRACTOR CROP_SECONDS BATCH_SIZE SAMPLES (1) STEPS (375000) CKPT_EVERY (10000) WORKERS (16) SAVE_DIR EXTRA
#        STAGE_LOCAL=1 (copy the shards to the node NVMe first; add --constraint=scratch-node)
#   E2 (50 Hz, same seconds as the paper model, batch 512, one H100, fresh):
#   STAGE_LOCAL=1 EXTRACTOR=wav2vec2 CROP_SECONDS=2.01 BATCH_SIZE=512 EXTRA="masker.target_length=5 masker.context_mask_length=5" \
#     sbatch --export=ALL --constraint=scratch-node --job-name=e2w2v2 train_audioset.sbatch
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
NUM_GPUS=${NUM_GPUS:-1}   # for N>1 also pass: sbatch --gpus=N --cpus-per-task=$((16*N)) (DDP; effective batch = N x BATCH_SIZE)
SAVE_DIR=${SAVE_DIR:-/scratch-shared/$USER/wavjepa_runs}
EXTRA=${EXTRA:-}
EXTRACTOR=${EXTRACTOR:-wavjepa}   # wavjepa (10 ms tokens) or wav2vec2 (extra (512,2,2) layer, 20 ms tokens)
SHARDS=/scratch-shared/gyuksel2/audioset_16k
DATA_DIRS="$SHARDS/unbal_train{000..869}.tar"
STAGE_LOCAL=${STAGE_LOCAL:-0}   # 1 = copy the shards to the per-job NVMe first (submit with --constraint=scratch-node)

mkdir -p "$SAVE_DIR"
echo "node $(hostname), $(date), job $SLURM_JOB_ID ($SLURM_JOB_NAME)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---- optional: stage the 16 kHz shards (365 GB) to the per-job NVMe volume, as benchmarks/mock_train.sbatch does
JOBVOL=/scratch-node/$USER.$SLURM_JOB_ID
if [ "$STAGE_LOCAL" = 1 ] && [ -d "$JOBVOL" ]; then
    LOCAL=$JOBVOL/audioset_16k; mkdir -p "$LOCAL"
    trap 'echo "removing $LOCAL"; rm -rf "$LOCAL"' EXIT
    t0=$(date +%s)
    seq -f "$SHARDS/unbal_train%03g.tar" 0 869 | xargs -P 16 -I{} cp {} "$LOCAL"/
    echo "staged $(ls "$LOCAL" | wc -l) shards, $(du -sh "$LOCAL" | cut -f1), in $(( $(date +%s) - t0 )) s"
    df -h "$JOBVOL" | tail -1
    DATA_DIRS="$LOCAL/unbal_train{000..869}.tar"
elif [ "$STAGE_LOCAL" = 1 ]; then
    echo "STAGE_LOCAL=1 but no /scratch-node volume on $(hostname): reading from scratch-shared"
fi
echo "config: extractor $EXTRACTOR, ${CROP_SECONDS}s crops, batch $BATCH_SIZE clips x $SAMPLES, steps $STEPS, gpus $NUM_GPUS, ckpt every $CKPT_EVERY, workers $WORKERS, save_dir $SAVE_DIR, extra '$EXTRA'"

# shellcheck disable=SC2086
python train.py masker=AudioSet data=audioset_16k "extractor=$EXTRACTOR" \
    "data.data_dirs=\"$DATA_DIRS\"" "data.samples_per_audio=$SAMPLES" "data.process_seconds=$CROP_SECONDS" \
    "data.num_workers=$WORKERS" data.prefetch_factor=2 \
    "trainer.batch_size=$BATCH_SIZE" "trainer.num_gpus=$NUM_GPUS" "trainer.steps=$STEPS" \
    "trainer.checkpoint_every_n_steps=$CKPT_EVERY" trainer.use_packing=True trainer.compile_modules=True \
    "save_dir=$SAVE_DIR" $EXTRA
echo "train.py exit=$? $(date)"
