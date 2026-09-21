#!/bin/bash
# HEAR evaluation of the revision runs (latest common checkpoint) on one task.
#   sbatch run_hear_revision.sbatch                       # ESC-50, step 130000, all three models
#   STEP=160000 TASK=esc50-v2.0.0-full sbatch --array=0 run_hear_revision.sbatch
# Array index -> model: 0 = 2.01 s wavjepa, 1 = 4.02 s wav2vec2 extractor, 2 = 5 s wavjepa, 3 = the PAPER checkpoint (375k, ignores STEP).
#
#SBATCH --job-name=hearrev
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
# 2026-09-09: user's standing instruction — submit these evaluations on gpu_a100 (the defaults above), not gpu_h100.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --time=02:00:00
#SBATCH --output=benchmarks/logs/hear_%x_%A_%a.out

set -o pipefail
REPO=${WAVJEPA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO"
module load 2023
module load Anaconda3/2023.07-2
source activate sjepa-eval

STEP=${STEP:-130000}
# E1 ablation runs (fresh A / B / C, batch 256, 4 h jobs, checkpoints every 5k)
RUNS_E1="/scratch-shared/gyuksel2/wavjepa_runs_e1/saved_models_jepa_new_masking/Data=AudioSet16k/Extractor=wavjepa/InSeconds=2.01/BatchSize=256/NrSamples=1/NrGPUs=1/LR=0.0004/TargetProb=0.25/TargetLen=10/ContextProb=0.65/ContextLen=10/MinContextBlock=1/ContextRatio=0.1/Packed=True"
TASK=${TASK:-esc50-v2.0.0-full}
TASKS_DIR=${TASKS_DIR:-/scratch-shared/gyuksel2/hear_tasks}   # 16 kHz task copies (scripts/make_hear_task_16k.py); /projects/0/prjs1261/tasks holds 32 kHz only
GRID=${GRID:-default}
RUNS=/scratch-shared/gyuksel2/wavjepa_runs/saved_models_jepa_new_masking/Data=AudioSet16k
TAIL="NrSamples=1/NrGPUs=1/LR=0.0004/TargetProb=0.25/TargetLen=10/ContextProb=0.65/ContextLen=10/MinContextBlock=1/ContextRatio=0.1/Packed=True"

case "$SLURM_ARRAY_TASK_ID" in
  0) RUN=wj2s;    MODEL=hear_configs.WavJEPA;      W="$RUNS/Extractor=wavjepa/InSeconds=2.01/BatchSize=256/$TAIL/step=$STEP.ckpt";;
  1) RUN=wj4sw2v; MODEL=hear_configs.WavJEPA_w2v2; W="$RUNS/Extractor=wav2vec2/InSeconds=4.02/BatchSize=256/$TAIL/step=$STEP.ckpt";;
  2) RUN=wj5s;    MODEL=hear_configs.WavJEPA_5s;   W="$RUNS/Extractor=wavjepa/InSeconds=5/BatchSize=128/$TAIL/step=$STEP.ckpt";;
  3) RUN=paper375k; MODEL=hear_configs.WavJEPA; W="/projects/0/prjs1261/saved_models_jepa_reproduce/SR=16000/LibriRatio=0.0/BatchSize=32/NrSamples=8/NrGPUs=2/ModelSize=base/LR=0.0004/Masking=time-inverse-masker/TargetProb=0.25/TargetLen=10/ContextLen=10/TopK=8/step=375000.ckpt"; STEP=375000;;
  21) RUN=paper_b512; MODEL=hear_configs.WavJEPA; W="/projects/0/prjs1261/saved_models_jepa_reproduce/SR=16000/LibriRatio=0.0/BatchSize=32/NrSamples=8/NrGPUs=2/ModelSize=base/LR=0.0004/Masking=time-inverse-masker/TargetProb=0.25/TargetLen=10/ContextLen=10/TopK=8/step=$STEP.ckpt";;   # the paper's batch-512 run at any of its 25k checkpoints (STEP)
  # BEATs_iter3, whole clip in one pass (= non-streaming) through the streaming wrapper with W = C = inf;
  # frames = one per 160 ms time patch (mean of its 8 band tokens), scene = mean over frames.
  4) RUN=beats; MODEL=hear_configs.BEATs_stream; W="/projects/0/prjs1338/models/BEATs/BEATs_iter3.pt"; STEP=iter3; export STREAM_W=inf STREAM_C=inf MODEL_PATH="$W";;
  5) RUN=eat;   MODEL=hear_configs.EAT_stream;   W="/projects/0/prjs1338/models/EAT/EAT-base_epoch30_pretrain"; STEP=epoch30; export STREAM_W=${STREAM_W:-10} STREAM_C=${STREAM_C:-10} EAT_MODEL_DIR="$W";;  # 10 s chunks concatenated
  6) RUN=mwmae_official; MODEL=hear_configs.MWMAE_stream; W="/projects/0/prjs1338/models/MWMAE/mwmae_base_200_4x16_8x128_default_bfloat16_run1"; STEP=198400; export STREAM_W=${STREAM_W:-inf} STREAM_C=${STREAM_C:-inf} MW_MAE_MODEL_DIR="$W" MWMAE_FRAMING=${MWMAE_FRAMING:-all} MWMAE_NORM=clip MWMAE_BAND_POOL=concat XLA_PYTHON_CLIENT_PREALLOCATE=false; source activate mwmae-eval;;  # the authors' HEAR wrapper exactly: whole clip, clip norm, every column, 3840-d (verified identical)
  7) RUN=mwmae_mean; MODEL=hear_configs.MWMAE_stream; W="/projects/0/prjs1338/models/MWMAE/mwmae_base_200_4x16_8x128_default_bfloat16_run1"; STEP=198400; export STREAM_W=${STREAM_W:-inf} STREAM_C=${STREAM_C:-inf} MW_MAE_MODEL_DIR="$W" MWMAE_FRAMING=${MWMAE_FRAMING:-all} MWMAE_NORM=clip MWMAE_BAND_POOL=mean XLA_PYTHON_CLIENT_PREALLOCATE=false; source activate mwmae-eval;;  # as 6 with the 5 band tokens averaged (768-d)
  8) RUN=audiomae; MODEL=hear_configs.AudioMAE_stream; W="/projects/0/prjs1338/models/AudioMAE/pretrained.pth"; STEP=as2m; export STREAM_W=${STREAM_W:-10.24} STREAM_C=${STREAM_C:-10.24} AUDIOMAE_MODEL_PATH="$W";;  # 10.24 s units concatenated, 8 band tokens averaged
  9) RUN=ssast;    MODEL=hear_configs.SSAST_stream;    W="/projects/0/prjs1338/models/SSAST/SSAST-Base-Patch-400.pth"; STEP=patch400; export STREAM_W=${STREAM_W:-10.24} STREAM_C=${STREAM_C:-10.24} SSAST_MODEL_PATH="$W"; source activate /projects/0/prjs1338/envs/ssast-eval;;  # 10.24 s units (the pre-training input length)
  10) RUN=atst_clipenc; MODEL=hear_configs.ATST_stream; W="/projects/0/prjs1338/models/ATST/base.ckpt"; STEP=base; export STREAM_W=${STREAM_W:-6} STREAM_C=${STREAM_C:-6} ATST_MODEL_PATH="$W" ATST_SCENE=${ATST_SCENE:-clip};;  # the authors' CLIP encoder (12-block CLS + mean, 18432-d); ATST_SCENE=mean = mean of the 40 ms tokens (run name atst_stepbase, 86.9)
  11) RUN=ssast_awsomenorm; MODEL=hear_configs.SSAST_stream; W="/projects/0/prjs1338/models/SSAST/SSAST-Base-Patch-400.pth"; STEP=patch400; export SSAST_NORM=awsome STREAM_W=${STREAM_W:-10.24} STREAM_C=${STREAM_C:-10.24} SSAST_MODEL_PATH="$W"; source activate /projects/0/prjs1338/envs/ssast-eval;;  # audit: the benchmark wrapper's (x-mean)/std**2 normalisation, everything else the authors'
  # AudioSet-pretrained HuBERT / wav2vec2 (ALM/*-audioset), whole clip in one pass as in the awsome wrapper; WAVJEPA_LAYERS=1,4,8,12 = layer probe
  23) RUN=hubert_as_base;  MODEL=hear_configs.HF_AS; W="/projects/0/prjs1338/models/HF/hubert-base-audioset";   STEP=as;;
  24) RUN=w2v2_as_base;    MODEL=hear_configs.HF_AS; W="/projects/0/prjs1338/models/HF/wav2vec2-base-audioset"; STEP=as;;
  25) RUN=hubert_as_large; MODEL=hear_configs.HF_AS; W="/projects/0/prjs1338/models/HF/hubert-large-audioset";  STEP=as;;
  26) RUN=w2v2_as_large;   MODEL=hear_configs.HF_AS; W="/projects/0/prjs1338/models/HF/wav2vec2-large-audioset"; STEP=as;;
  # E6 ablation runs: the run dir and score name come from the environment (E6_DIR = full checkpoint dir, E6_RUN = name)
  27) RUN="${E6_RUN:?E6_RUN unset}"; MODEL=hear_configs.WavJEPA; W="${E6_DIR:?E6_DIR unset}/step=$STEP.ckpt";;
  12) RUN=audiomae_awsomenorm; MODEL=hear_configs.AudioMAE_stream; W="/projects/0/prjs1338/models/AudioMAE/pretrained.pth"; STEP=as2m; export AUDIOMAE_NORM=awsome STREAM_W=${STREAM_W:-10.24} STREAM_C=${STREAM_C:-10.24} AUDIOMAE_MODEL_PATH="$W";;  # audit: as above for AudioMAE
  # E1 objective ablation (batch 256, 2.01 s, 100 Hz; /scratch-shared/gyuksel2/wavjepa_runs_e1): STEP selects the checkpoint
  13) RUN=e1A_jepa;   MODEL=hear_configs.WavJEPA; W="$RUNS_E1/step=$STEP.ckpt";;
  14) RUN=e1B_latent; MODEL=hear_configs.WavJEPA; W="$RUNS_E1/Objective=latent/step=$STEP.ckpt";;
  15) RUN=e1C_bestrq; MODEL=hear_configs.WavJEPA; W="$RUNS_E1/Objective=bestrq/step=$STEP.ckpt";;
  17) RUN=e1D_d2v2;   MODEL=hear_configs.WavJEPA; W="$RUNS_E1/Objective=d2v2/D2v2Dec=tf12/step=$STEP.ckpt";;   # arm A's predictor as the decoder, one joint pass (256 clips x 4 masks = 1024 views/step)
  22) RUN=e1D_d2v2_b64; MODEL=hear_configs.WavJEPA; W="${RUNS_E1/BatchSize=256/BatchSize=64}/Objective=d2v2/D2v2Dec=tf12/step=$STEP.ckpt";;   # as 17 at 64 clips x 4 masks = 256 views/step, matched to A's 256 (job e1D64)
  20) RUN=e1D_d2v2_conv20x7; MODEL=hear_configs.WavJEPA; W="$RUNS_E1/Objective=d2v2/D2v2Dec=20x7/step=$STEP.ckpt";;   # data2vec 2.0's conv decoder, deepened to reach context
  19) RUN=e1D_d2v2_dec4x7; MODEL=hear_configs.WavJEPA; W="$RUNS_E1/Objective=d2v2/D2v2Dec=4x7/step=$STEP.ckpt";;   # their 4 x 7 decoder, range-limited under our masking (32 % unreachable)
  # arm P: the paper's pipeline (legacy loader, 32 clips x 8 in-batch crops) on one GPU (train_paper_pipeline.sbatch)
  18) RUN=e1P_paperpipe; MODEL=hear_configs.WavJEPA; W="/scratch-shared/gyuksel2/wavjepa_runs_e1/saved_models_jepa_new_masking/Data=AudioSet/Extractor=wavjepa/InSeconds=2.01/BatchSize=32/NrSamples=8/NrGPUs=1/LR=0.0004/TargetProb=0.25/TargetLen=10/ContextProb=0.65/ContextLen=10/MinContextBlock=1/ContextRatio=0.1/Packed=True/step=$STEP.ckpt";;
  # E2: 50 Hz front end, 2.01 s crops, batch 512, 100 ms mask blocks (5 tokens); STEP selects the checkpoint
  16) RUN=e2w2v2_b512; MODEL=hear_configs.WavJEPA_w2v2_2s; W="$RUNS/Extractor=wav2vec2/InSeconds=2.01/BatchSize=512/NrSamples=1/NrGPUs=1/LR=0.0004/TargetProb=0.25/TargetLen=5/ContextProb=0.65/ContextLen=5/MinContextBlock=1/ContextRatio=0.1/Packed=True/step=$STEP.ckpt";;
  *) echo "bad array index"; exit 1;;
esac
# WAVJEPA_LAYERS=1,4,8,12 concatenates those encoder blocks instead of using the final layer (layer-wise probing);
# the run name records it so the scores never collide with the default single-layer ones.
# GOTCHA: the value contains commas, so it must reach the job through the environment (`WAVJEPA_LAYERS=1,4,8,12 sbatch --export=ALL ...`);
# inside `--export=ALL,WAVJEPA_LAYERS=1,4,8,12` sbatch splits on the commas and the job sees WAVJEPA_LAYERS=1 (run name ..._L1).
if [ -n "${WAVJEPA_LAYERS:-}" ]; then export WAVJEPA_LAYERS; RUN="${RUN}_L$(echo "$WAVJEPA_LAYERS" | tr ',' '-')"; fi
# WAVJEPA_LAYER_POOL=mean averages the selected layers (768-d) instead of concatenating them (len x 768); run name gets "m"
if [ "${WAVJEPA_LAYER_POOL:-concat}" = mean ]; then export WAVJEPA_LAYER_POOL; RUN="${RUN}m"; fi
# HEAR_LAYER_SUM=<L>: SUPERB-style learnable softmax-weighted sum over the L concatenated layers, learned inside the probe
# (heareval LayerWeightedSum). Probe stage only: the EMBEDDINGS are the plain L-layer concat (shared with any other probe of
# them), while the probe dir and the score dir get a "_ws" tag so these scores never mix with the plain-concat ones.
WS_TAG=""; if [ -n "${HEAR_LAYER_SUM:-}" ]; then export HEAR_LAYER_SUM; WS_TAG="_ws"; fi
# HEAR_LAYER_SUM_LR_MULT=<m>: lr multiplier for the layer-weight logits (SUPERB); recorded in the tag (_ws100) so runs
# with different multipliers never share a score dir. Unset = the heareval default.
if [ -n "${HEAR_LAYER_SUM:-}" ] && [ -n "${HEAR_LAYER_SUM_LR_MULT:-}" ]; then export HEAR_LAYER_SUM_LR_MULT; WS_TAG="_ws${HEAR_LAYER_SUM_LR_MULT}"; fi
RUN="${RUN}${RUN_SUFFIX:-}"   # RUN_SUFFIX=_a100 etc.: submit the same evaluation twice (different partitions) without the two jobs sharing embedding/score dirs
test -e "$W" || { echo "checkpoint not found: $W" >&2; exit 1; }

# Embedding root. Default: scratch-shared (kept across jobs: STAGE=extract / probe share them). EMB_ROOT=local writes them
# to the job's NVMe volume /scratch-node/<user>.<job> (submit with --constraint=scratch-node; wiped when the job ends):
# for STAGE=both jobs the embeddings are only needed inside the job, and the eval kit writes 2 small files per clip
# (46k for SC-5h), which on the shared GPFS under load cost ~140-250 ms each (an SC-5h job took > 1 h) versus
# milliseconds on the NVMe. NOTE: $TMPDIR (/scratch-local/<user>.<job>) is NOT local on Snellius -- it is a per-node
# directory on the shared scratch1 GPFS (1.8 PB "free"), so it is just as slow; only /scratch-node is a real local disk.
# Any other value = a directory (e.g. a /projects path).
EMB_ROOT=${EMB_ROOT:-/scratch-shared/$USER/hear_embeddings}
if [ "$EMB_ROOT" = local ]; then
    [ "$STAGE" = both ] || [ -z "${STAGE:-}" ] || { echo "EMB_ROOT=local only makes sense with STAGE=both (embeddings vanish with the job)" >&2; exit 1; }
    JOBVOL=/scratch-node/$USER.$SLURM_JOB_ID
    [ -d "$JOBVOL" ] || { echo "EMB_ROOT=local needs the job's NVMe volume $JOBVOL: submit with --constraint=scratch-node (node $(hostname) has none)" >&2; exit 1; }
    EMB_ROOT="$JOBVOL/hear_embeddings"
    mkdir -p "$EMB_ROOT" && touch "$EMB_ROOT/.w" 2>/dev/null || { echo "no writable node-local scratch at $EMB_ROOT" >&2; exit 1; }
    echo "embeddings on node-local NVMe: $EMB_ROOT ($(df -h "$EMB_ROOT" | tail -1 | awk '{print $2" volume, "$4" free"}'))"
    # Slurm removes the per-job $TMPDIR when the job ends; delete explicitly too (also on failure / scancel), so nothing
    # is left behind and only the scores (copied to /projects below) survive.
    trap 'rm -rf "$EMB_ROOT"; echo "removed node-local embeddings"' EXIT

fi
EMB=$EMB_ROOT/${RUN}_step${STEP}
GRID_TAG=""; [ "$GRID" != "default" ] && GRID_TAG="_grid${GRID}"   # keep the default-grid results separate
# E7 probe seeds, in two stages so the seeds can run in PARALLEL (extraction is seed-independent):
#   STAGE=extract  -> embeddings only, one job per (model, task)          [GPU-bound]
#   STAGE=probe    -> probe only, one job per (model, task, HEAR_SEED)    [reuses the cached embeddings]
#   STAGE=both     -> the original single-job behaviour (default)
# Seed 42 is the kit's default and owns the untagged score dir; other seeds get "_seed<s>".
STAGE=${STAGE:-both}
SEED=${HEAR_SEED:-42}; export HEAR_SEED=$SEED
SEED_TAG=""; [ "$SEED" != 42 ] && SEED_TAG="_seed${SEED}"
SCORES=/projects/0/prjs1338/hear_scores_revision/${RUN}${WS_TAG}_step${STEP}${GRID_TAG}${SEED_TAG}/$TASK
echo "node $(hostname), $(date): run=$RUN model=$MODEL step=$STEP task=$TASK grid=$GRID seed=$SEED stage=$STAGE"
echo "weights: $W"
mkdir -p "$EMB"

cd "$REPO/hear-eval-kit"     # task_embeddings.py appends ../../wavjepa (= the repo root) to sys.path
TASKDIR="$EMB/$MODEL/$TASK"
if [ "$STAGE" != probe ]; then
    python3 -m heareval.embeddings.runner "$MODEL" --tasks-dir "$TASKS_DIR" --task "$TASK" --embeddings-dir "$EMB" --model "$W" || { echo "embeddings failed"; exit 1; }
fi
if [ "$STAGE" = extract ]; then echo "EXTRACTED $RUN step $STEP $TASK -> $TASKDIR"; echo "finished $(date)"; exit 0; fi
test -e "$TASKDIR/.done.embeddings" || { echo "no embeddings in $TASKDIR (run STAGE=extract first)" >&2; exit 1; }

# The eval kit writes every probe output (scores, predictions, CSV logs) INTO the task directory, so concurrent seeds on
# the same embeddings would clobber each other. Every probe job therefore works in its OWN directory of symlinks to the
# shared embedding files: no re-extraction, no races, seeds run in parallel.
PROBE_DIR="$EMB/probe_seed${SEED}${GRID_TAG}${WS_TAG}/$MODEL/$TASK"; mkdir -p "$PROBE_DIR" "$SCORES"
for f in "$TASKDIR"/* "$TASKDIR"/.done.embeddings; do
    [ -e "$f" ] || continue; b=$(basename "$f")
    case "$b" in test.predicted-*|test.predictions.json|prediction-done.json|*.predictions.pkl|layer_weights.json|logs|lightning_logs) continue;; esac
    ln -sfn "$f" "$PROBE_DIR/$b"
done
rm -f "$PROBE_DIR/prediction-done.json"   # the kit skips the probe when this marker is present
source activate sjepa-eval   # the probe always runs in the eval-kit env
python3 -m heareval.predictions.runner "$PROBE_DIR" --grid "$GRID" || { echo "predictions failed"; exit 1; }

cp "$PROBE_DIR/test.predicted-scores.json" "$SCORES/" && echo "scores -> $SCORES/test.predicted-scores.json"
[ -f "$PROBE_DIR/layer_weights.json" ] && cp "$PROBE_DIR/layer_weights.json" "$SCORES/"   # learned layer weights (HEAR_LAYER_SUM)
python3 -c "import json; d=json.load(open('$SCORES/test.predicted-scores.json')); a=d.get('aggregated_scores', d.get('test', d)); print('RESULT run=$RUN$WS_TAG step=$STEP task=$TASK seed=$SEED grid=$GRID score=%.4f' % a.get('test_score_mean', a.get('test_score', float('nan'))))"
# Embeddings are KEPT (the eval kit skips extraction when <emb>/<model>/<task>/.done.embeddings exists),
# so re-running another probe grid on the same model/task only runs the prediction stage. CLEAN_EMB=1 deletes them.
[ "${CLEAN_EMB:-0}" = 1 ] && rm -rf "$EMB/$MODEL/$TASK"
echo "finished $(date)"
