#!/bin/bash
# Soft abstain variant: re-answer all questions from the stored retrievals with the
# soft abstain prompt, then score Table 1 (10 judge runs) and the adversarial questions.
# Submit from this folder:   sbatch mem0_abstain_soft.slurm
# Output goes to slurm_logs/mem0-abstain-soft-<jobid>.out
#SBATCH --job-name=mem0-abstain-soft
#SBATCH --account=punim1942
#SBATCH --partition=sapphire
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/mem0-abstain-soft-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "Host: $(hostname)   Commit: $(git rev-parse --short HEAD)   Started: $(date)"
make mem0-abstain-soft-all
echo "Finished: $(date)"
