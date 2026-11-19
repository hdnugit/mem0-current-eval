# Use the eval venv's Python even if the venv isn't activated (override: make PYTHON=...)
PYTHON ?= $(abspath ../.venv/bin/python)


# Run the experiments
# --- Mem0 (OSS backend, Table 1 "Mem0" row) ---
MEM0_RESULTS = results/mem0_results_top_30_filter_False_graph_False.json
MEM0_METRICS = results/mem0_evaluation_metrics.json
MEM0_ADVERSARIAL = results/mem0_adversarial_metrics.json

run-mem0-add:
	$(PYTHON) run_experiments.py --technique_type mem0 --method add

run-mem0-add-fresh:   # wipe saved progress and re-ingest everything
	$(PYTHON) run_experiments.py --technique_type mem0 --method add --fresh

run-mem0-search:
	mkdir -p results
	$(PYTHON) run_experiments.py --technique_type mem0 --method search --output_folder results/ --top_k 30

eval-mem0:
	$(PYTHON) evals.py --input_file $(MEM0_RESULTS) --output_file $(MEM0_METRICS) --num_runs 10

scores-mem0:
	$(PYTHON) generate_scores.py --input_file $(MEM0_METRICS)

# LOCOMO category 5 (adversarial); separate judge, not part of the paper's Table 1.
# Add HAND=adversarial_hand_labels.json to compare the judge with hand labels.
eval-adversarial:
	$(PYTHON) eval_adversarial.py --input_file $(MEM0_RESULTS) --output_file $(MEM0_ADVERSARIAL) $(if $(HAND),--hand_labels $(HAND))

mem0-all: run-mem0-add run-mem0-search eval-mem0 scores-mem0

# --- Abstain baseline (not in the paper): same retrievals as the Mem0 run above,
# answered with ANSWER_PROMPT_ABSTAIN. Needs $(MEM0_RESULTS); the memory store is not used.
MEM0_ABSTAIN_RESULTS = results/mem0_abstain_results_top_30_filter_False_graph_False.json
MEM0_ABSTAIN_METRICS = results/mem0_abstain_evaluation_metrics.json
MEM0_ABSTAIN_ADVERSARIAL = results/mem0_abstain_adversarial_metrics.json

run-mem0-abstain:      # resumable; rerun the same command after an interruption
	$(PYTHON) run_abstain_baseline.py --input_file $(MEM0_RESULTS) --output_file $(MEM0_ABSTAIN_RESULTS)

eval-mem0-abstain:
	$(PYTHON) evals.py --input_file $(MEM0_ABSTAIN_RESULTS) --output_file $(MEM0_ABSTAIN_METRICS) --num_runs 10

scores-mem0-abstain:
	$(PYTHON) generate_scores.py --input_file $(MEM0_ABSTAIN_METRICS)

eval-adversarial-abstain:
	$(PYTHON) eval_adversarial.py --input_file $(MEM0_ABSTAIN_RESULTS) --output_file $(MEM0_ABSTAIN_ADVERSARIAL)

stats-mem0-abstain:
	$(PYTHON) run_abstain_baseline.py --output_file $(MEM0_ABSTAIN_RESULTS) --stats_only

mem0-abstain-all: run-mem0-abstain eval-mem0-abstain scores-mem0-abstain eval-adversarial-abstain

run-rag:
	$(PYTHON) run_experiments.py --technique_type rag --chunk_size 500 --num_chunks 1 --output_folder results/

run-full-context:
	$(PYTHON) run_experiments.py --technique_type rag --chunk_size -1 --num_chunks 1 --output_folder results/

run-langmem:
	$(PYTHON) run_experiments.py --technique_type langmem --output_folder results/

run-zep-add:
	$(PYTHON) run_experiments.py --technique_type zep --method add --output_folder results/

run-zep-search:
	$(PYTHON) run_experiments.py --technique_type zep --method search --output_folder results/

run-openai:
	$(PYTHON) run_experiments.py --technique_type openai --output_folder results/
