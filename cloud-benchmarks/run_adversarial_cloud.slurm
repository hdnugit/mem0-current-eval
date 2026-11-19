#!/bin/bash
# LOCOMO adversarial questions (category 5) for a finished Mem0 Cloud run.
# Reuses the memories that run_locomo_cloud.slurm already ingested (no re-ingest):
#   step 1  answer_adversarial.py     search Mem0 Cloud (top 50) + answer, suite's prompt and model
#   step 2  ../evaluation/eval_adversarial.py   four-label adversarial judge (gpt-4o-mini)
#
# Submit from this folder, with the SAME project name as the main run:
#   sbatch run_adversarial_cloud.slurm                      # project locomo-cloud
#   sbatch run_adversarial_cloud.slurm smoke --conversations 0 --max-questions 5
# Arguments after the project name go to answer_adversarial.py. Rerunning continues
# where step 1 stopped.
#
# Output in results/locomo/predicted_<project-name>/:
#   adversarial_answers_top50.json   answers
#   adversarial_metrics_top50.json   judged answers; summary table is in the Slurm log
#SBATCH --job-name=locomo-cloud-adv
#SBATCH --account=punim1942
#SBATCH --partition=sapphire
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/locomo-cloud-adv-%j.out

set -euo pipefail
HERE="$SLURM_SUBMIT_DIR"
BENCH=/data/gpfs/projects/punim1942/mem0/evaluation
JUDGE_DIR="$HERE/../evaluation"
PROJECT="${1:-locomo-cloud}"
shift || true
RUN_DIR="$HERE/results/locomo/predicted_$PROJECT"
ANSWERS="$RUN_DIR/adversarial_answers_top50.json"
METRICS="$RUN_DIR/adversarial_metrics_top50.json"

[ -d "$RUN_DIR" ] || { echo "No main run at $RUN_DIR (run run_locomo_cloud.slurm $PROJECT first)"; exit 1; }
echo "Host: $(hostname)   Project: $PROJECT   Started: $(date)"

cd "$BENCH"
PYTHONPATH="$BENCH" .venv/bin/python "$HERE/answer_adversarial.py" \
    --run-dir "$RUN_DIR" --output-file "$ANSWERS" --top-k 50 "$@"

cd "$JUDGE_DIR"
../.venv/bin/python eval_adversarial.py \
    --input_file "$ANSWERS" --dataset "$BENCH/datasets/locomo/locomo10.json" --output_file "$METRICS"
echo "Finished: $(date)"
