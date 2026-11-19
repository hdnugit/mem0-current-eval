# Mem0 Cloud run of mem0's own LOCOMO benchmark

Runs the benchmark suite shipped in the current mem0 repo (`../../mem0/evaluation`, the
`mem0ai/memory-benchmarks` submodule) against the **Mem0 Cloud platform**, as in that suite's
README ("Option A: Mem0 Cloud"). This is separate from `../evaluation/`, which reproduces the
paper's Table 1 with the open-source Memory class.

Decisions (2026-11-19):
- LOCOMO only (no LongMemEval/BEAM, no other baselines), categories 1-4 as in Table 1.
- Top 50 memories only (`--top-k 50 --top-k-cutoffs 50`); the suite's default is 200 with cutoffs 10/20/50/200.
- Answerer and judge are the suite's code default, `gpt-5` (its README says `gpt-4o`).
- The suite reports the LLM-judge score (J) only, not F1/B1.

Keys: `MEM0_API_KEY` and `OPENAI_API_KEY` in `../../mem0/evaluation/.env` (gitignored, not in this repo).
The run writes the LOCOMO conversations into that Mem0 account as memories.

## Main run (categories 1-4)

```bash
sbatch run_locomo_cloud.slurm smoke --conversations 0 --max-questions 20   # quick test
sbatch run_locomo_cloud.slurm                                              # full run, project "locomo-cloud"
sbatch run_locomo_cloud.slurm locomo-cloud --resume                        # continue a stopped run
```

Results: `results/locomo/predicted_<project-name>/`; log: `slurm_logs/locomo-cloud-<jobid>.out`.

## Adversarial questions (category 5), after the main run

The suite cannot run category 5 (no gold answer; it reads `qa["answer"]`). `answer_adversarial.py`
reuses the memories of the main run (user ids from `_ingestion_<conv>.json`, so no re-ingest),
searches and answers the 446 questions with the suite's search call, answer prompt and model, and
`../evaluation/eval_adversarial.py` judges them (DECLINED/REATTRIBUTED correct, TRAP/OTHER wrong).

```bash
sbatch run_adversarial_cloud.slurm            # same project name as the main run
```

Needs the main run's results folder and the memories still present in the Mem0 account.
Output: `adversarial_answers_top50.json` and `adversarial_metrics_top50.json` in the run's results folder.

Status 2026-11-19: scripts written, not yet run.
