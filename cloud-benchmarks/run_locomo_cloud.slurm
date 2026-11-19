#!/bin/bash
# Run mem0's own LOCOMO benchmark (mem0/evaluation = mem0ai/memory-benchmarks)
# against the Mem0 Cloud platform, on a compute node.
#
# Keys are read from /data/gpfs/projects/punim1942/mem0/evaluation/.env
# (MEM0_API_KEY and OPENAI_API_KEY).
#
# Submit from this folder:
#   sbatch run_locomo_cloud.slurm                       # full run: open-domain (category 3) only, top 50 memories; see README for why
#   sbatch run_locomo_cloud.slurm smoke --conversations 0 --max-questions 20
#   sbatch run_locomo_cloud.slurm <project-name> --resume   # continue a stopped run
# First argument = project name (default: locomo-cloud); the rest is passed to
# benchmarks.locomo.run unchanged.
#
# Output: slurm_logs/locomo-cloud-<jobid>.out, results in results/locomo/predicted_<project-name>/
#SBATCH --job-name=locomo-cloud
#SBATCH --account=punim1942
#SBATCH --partition=sapphire
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/locomo-cloud-%j.out

set -euo pipefail
HERE="$SLURM_SUBMIT_DIR"
BENCH=/data/gpfs/projects/punim1942/mem0/evaluation
PY311=/apps/easybuild-2022/easybuild/software/Core/Anaconda3/2024.02-1/bin/python3.11
PROJECT="${1:-locomo-cloud}"
shift || true

cd "$BENCH"
if [ ! -f .env ]; then
    echo "Missing $BENCH/.env (needs MEM0_API_KEY and OPENAI_API_KEY)"; exit 1
fi

# Build the venv once (same Python as mem0-current-eval/.venv, works on compute nodes).
if [ ! -x .venv/bin/python ]; then
    "$PY311" -m venv .venv
    .venv/bin/pip install --quiet -r requirements.txt
fi

# The suite (commit 4b61c5d) posts adds to /v3/memories/, which Mem0 Cloud now treats as
# "list memories" (400 "Filters are required"); the add endpoint is /v3/memories/add/.
if git apply --check "$HERE/mem0_client_add_endpoint.patch" 2>/dev/null; then
    git apply "$HERE/mem0_client_add_endpoint.patch"
fi
grep -q "/v3/memories/add/" benchmarks/common/mem0_client.py || { echo "add-endpoint patch not applied"; exit 1; }

echo "Host: $(hostname)   memory-benchmarks commit: $(git rev-parse --short HEAD)   Started: $(date)"
.venv/bin/python -m benchmarks.locomo.run \
    --project-name "$PROJECT" \
    --backend cloud \
    --output-dir "$HERE/results/locomo" \
    --top-k 50 --top-k-cutoffs 50 \
    --categories 3 \
    "$@"
echo "Finished: $(date)"
