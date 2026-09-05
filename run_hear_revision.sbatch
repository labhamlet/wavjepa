#!/bin/bash
# HEAR evaluation of the revision runs (latest common checkpoint) on one task.
#   sbatch run_hear_revision.sbatch                       # ESC-50, step 130000, all three models
#   STEP=160000 TASK=esc50-v2.0.0-full sbatch --array=0 run_hear_revision.sbatch
# Array index -> model: 0 = 2.01 s wavjepa, 1 = 4.02 s wav2vec2 extractor, 2 = 5 s wavjepa.
#
#SBATCH --job-name=hearrev
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
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
TASK=${TASK:-esc50-v2.0.0-full}
TASKS_DIR=${TASKS_DIR:-/projects/0/prjs1261/tasks}
GRID=${GRID:-default}
RUNS=/scratch-shared/gyuksel2/wavjepa_runs/saved_models_jepa_new_masking/Data=AudioSet16k
TAIL="NrSamples=1/NrGPUs=1/LR=0.0004/TargetProb=0.25/TargetLen=10/ContextProb=0.65/ContextLen=10/MinContextBlock=1/ContextRatio=0.1/Packed=True"

case "$SLURM_ARRAY_TASK_ID" in
  0) RUN=wj2s;    MODEL=hear_configs.WavJEPA;      W="$RUNS/Extractor=wavjepa/InSeconds=2.01/BatchSize=256/$TAIL/step=$STEP.ckpt";;
  1) RUN=wj4sw2v; MODEL=hear_configs.WavJEPA_w2v2; W="$RUNS/Extractor=wav2vec2/InSeconds=4.02/BatchSize=256/$TAIL/step=$STEP.ckpt";;
  2) RUN=wj5s;    MODEL=hear_configs.WavJEPA_5s;   W="$RUNS/Extractor=wavjepa/InSeconds=5/BatchSize=128/$TAIL/step=$STEP.ckpt";;
  *) echo "bad array index"; exit 1;;
esac
test -f "$W" || { echo "checkpoint not found: $W" >&2; exit 1; }

EMB=/scratch-shared/$USER/hear_embeddings/${RUN}_step${STEP}
SCORES=/projects/0/prjs1338/hear_scores_revision/${RUN}_step${STEP}/$TASK
echo "node $(hostname), $(date): run=$RUN model=$MODEL step=$STEP task=$TASK grid=$GRID"
echo "weights: $W"
mkdir -p "$EMB" "$SCORES"

cd "$REPO/hear-eval-kit"     # task_embeddings.py appends ../../wavjepa (= the repo root) to sys.path
python3 -m heareval.embeddings.runner "$MODEL" --tasks-dir "$TASKS_DIR" --task "$TASK" --embeddings-dir "$EMB" --model "$W" || { echo "embeddings failed"; exit 1; }
python3 -m heareval.predictions.runner "$EMB/$MODEL/$TASK" --grid "$GRID" || { echo "predictions failed"; exit 1; }

cp "$EMB/$MODEL/$TASK/test.predicted-scores.json" "$SCORES/" && echo "scores -> $SCORES/test.predicted-scores.json"
python3 -c "import json,sys; d=json.load(open('$SCORES/test.predicted-scores.json')); print('RESULT $RUN step $STEP $TASK:', json.dumps({k:v for k,v in d.items() if 'test' in k or 'aggregated' in k or 'score' in k}, indent=1)[:1500])"
rm -rf "$EMB/$MODEL/$TASK"
echo "finished $(date)"
