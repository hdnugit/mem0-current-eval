# Mem0 Cloud run of mem0's own LOCOMO benchmark

Runs the benchmark suite shipped in the current mem0 repo (`../../mem0/evaluation`, the
`mem0ai/memory-benchmarks` submodule) against the **Mem0 Cloud platform**, as in that suite's
README ("Option A: Mem0 Cloud"). This is separate from `../evaluation/`, which reproduces the
paper's Table 1 with the open-source Memory class.

Decisions (2026-11-19):
- LOCOMO only (no LongMemEval/BEAM, no other baselines).
- Questions: open-domain (category 3, 96 questions) plus all 446 adversarial (category 5) = 542 searches.
  Reason: the Mem0 free plan allows 1,000 retrieval requests/month (10,000 adds), and all of
  categories 1-4 would need 1,540 (multi-hop 282, temporal 321, open-domain 96, single-hop 841).
  All 10 conversations are still ingested (about 5,900 adds). Other categories can be added later
  with `sbatch run_locomo_cloud.slurm locomo-cloud --resume --categories 1,3` once quota allows.
- Top 50 memories only (`--top-k 50 --top-k-cutoffs 50`); the suite's default is 200 with cutoffs 10/20/50/200.
- Answerer and judge are the suite's code default, `gpt-5` (its README says `gpt-4o`).
- The suite reports the LLM-judge score (J) only, not F1/B1.

Keys: `MEM0_API_KEY` and `OPENAI_API_KEY` in `../../mem0/evaluation/.env` (gitignored, not in this repo).
The run writes the LOCOMO conversations into that Mem0 account as memories.

## Main run (category 3)

```bash
sbatch run_locomo_cloud.slurm smoke --conversations 0 --max-questions 20   # quick test
sbatch run_locomo_cloud.slurm                                              # category 3, project "locomo-cloud"
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

## Fix needed for Mem0 Cloud (2026-11-19)

The suite (commit 4b61c5d, May 2026) posts adds to `/v3/memories/`. Mem0 Cloud now treats that as
"list memories" and returns 400 "Filters are required"; the add endpoint is `/v3/memories/add/`
(as in `mem0/client/main.py`). `mem0_client_add_endpoint.patch` fixes the one line and
`run_locomo_cloud.slurm` applies it to `../../mem0/evaluation` before running. Search and event
polling endpoints still work unchanged.

The suite marks a chunk as done even when its add failed, so a run with add failures must not be
resumed: delete its `predicted_<project-name>/` folder and start again. After a run, check the log
for `Ingestion failed` (should be 0).

## Result: open-domain (category 3), top 50, Mem0 Cloud (job 35393456, 2026-11-19)

J = 66.7% (64/96), gpt-5 answerer and judge, single judge run. Ingestion: all 10 conversations,
0 failed adds; every question retrieved 50 memories; 0 failed searches. Job time 2 h 04 min.
For reference, the OSS run in `../evaluation` scores J 53.33 on the same 96 questions, but with a
different pipeline, top 30, and gpt-4o-mini answerer/judge, so the two are not directly comparable.
Files: `results/locomo/locomo_results_20261119_171820.json` (summary) and
`results/locomo/predicted_locomo-cloud/` (per question, plus `_ingestion_<conv>.json` user ids).


## Result: adversarial (category 5), top 50, Mem0 Cloud (job 35397155, 2026-11-19)

Correct 50.4% (225/446): DECLINED 212, REATTRIBUTED 13, TRAP 120, OTHER 101. Answers by gpt-5 with
the suite's answer prompt; labels by `../evaluation/eval_adversarial.py` (gpt-4o-mini judge, no
hand labels for these answers). 0 failed searches, every question retrieved 50 memories.
For reference, the OSS base run in `../evaluation` scores 37.7% with its own pipeline and prompt.
Caveats: 3 answers came back empty from gpt-5 (judged OTHER, REATTRIBUTED, OTHER; the
REATTRIBUTED one is a judge error). The judge was validated on the OSS run's short answers; these
answers are longer (median 108 characters), so its agreement with hand labels here is untested.
Files: `adversarial_answers_top50.json`, `adversarial_metrics_top50.json` in
`results/locomo/predicted_locomo-cloud/`.
Mem0 retrieval requests used by this project so far: 96 + 446 + 1 test = 543 of the 1,000/month.
Status 2026-11-19: first job (35393360) failed on every add for the reason above and was cancelled;
its output was deleted. Job 35393380 (categories 1-4) ingested about 10% and was stopped cleanly
(SIGINT, no failures, no searches) to switch to category 3 only. Job 35393456 resumed ingestion from
its checkpoints and finished cleanly. Adversarial step ran as job 35397155 (first run of answer_adversarial.py, no errors).

### Hand labels for the adversarial answers (2026-11-20)

`adversarial_hand_labels_cloud.json` has a hand label for each of the 446 answers above:
DECLINED 219, REATTRIBUTED 30, TRAP 100, OTHER 97, so 249 correct (55.8%). The judge agrees on
correct/wrong for 90.6% of answers and on all four labels for 89.2%. It marks 33 hand-correct
answers wrong (23 give the fact but say in the same answer that it belongs to the other person,
which the judge labels TRAP) and 9 hand-wrong answers correct (two of them empty answers).
Rules used: an answer that gives the fact and explicitly assigns it to the other person is
REATTRIBUTED; an answer that explicitly says a detail of the question is wrong ("it wasn't a
cake, it was ice cream") is DECLINED; an answer that silently swaps the detail, or only names the
other person's thing, is TRAP. By question type: fact of the other speaker 60.5% correct (332),
same speaker with a changed detail 42.0% (112). Retrieved memories were not saved by this run, so
whether the trap memory was retrieved cannot be checked.

```bash
cd ../evaluation && python eval_adversarial.py --report_only \
  --output_file ../cloud-benchmarks/results/locomo/predicted_locomo-cloud/adversarial_metrics_top50.json \
  --hand_labels ../cloud-benchmarks/adversarial_hand_labels_cloud.json
```
