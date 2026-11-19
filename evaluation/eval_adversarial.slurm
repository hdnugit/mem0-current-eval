#!/bin/bash
# Run the adversarial (category 5) judge on a compute node, not the login node.
# Submit from this folder:   sbatch eval_adversarial.slurm
# Output goes to slurm_logs/eval-adversarial-<jobid>.out
#SBATCH --job-name=eval-adversarial
#SBATCH --account=punim1942
#SBATCH --partition=sapphire
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/eval-adversarial-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "Host: $(hostname)   Commit: $(git rev-parse --short HEAD)   Started: $(date)"
make eval-adversarial HAND=adversarial_hand_labels.json
echo "Finished: $(date)"
