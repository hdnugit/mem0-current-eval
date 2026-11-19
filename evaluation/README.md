# Mem0: Building Production‑Ready AI Agents with Scalable Long‑Term Memory

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2504.19413)
[![Website](https://img.shields.io/badge/Website-Project-blue)](https://mem0.ai/research)

This repository contains the code and dataset for our paper: **Mem0: Building Production‑Ready AI Agents with Scalable Long‑Term Memory**.

## ▶️ Reproducing Table 1 (Mem0 row) with open-source Mem0

This copy of the paper's evaluation code has been adapted to run against the
**open-source** `Memory` class (local Qdrant + SQLite) instead of the hosted
Mem0 platform. Graph memory (Mem0g) is not wired up.

```bash
cd mem0-paper-eval
source .venv/bin/activate          # Python 3.12 venv with mem0 installed editable from ../mem0
cd evaluation
cp .env.example .env               # add OPENAI_API_KEY

make run-mem0-add      # ingest all 10 LOCOMO conversations (slow: many LLM calls)
make run-mem0-search   # top-30 retrieval + gpt-4o-mini answers -> results/
make eval-mem0         # F1, BLEU-1 and 10 LLM-judge runs per question
make scores-mem0       # Table 1 layout: F1 / B1 / J (mean ± std over runs)
```

Resuming: `make run-mem0-add` saves progress after every batch in
`mem0_store/add_progress.json`, stops the whole run on a persistent error
(e.g. out of credit), and picks up where it stopped when rerun. Errors are
logged to `mem0_store/add.log`. `make run-mem0-add-fresh` starts over.

What differs from the paper's run:
- Backend is OSS Mem0 from this checkout, not the 2025 Mem0 platform. The paper's
  custom instructions are passed via `custom_instructions` (see `src/memzero/oss.py`).
- Search uses current OSS defaults (semantic + BM25 + entity boost), `top_k=30`.
- Expect numbers to differ from Table 1; they are not directly comparable.

Session dates (observation-date fix): OSS `Memory.add()` cannot pass a
timestamp to mem0's extraction prompt, which then resolves "yesterday",
"last week" etc. against *today's* date. `src/memzero/oss.py` patches the
prompt builder so each LOCOMO session's date is used as the Observation Date
(and Current Date); `add.py` wraps every `add()` in `observation_date(...)`,
and an `add()` without one raises. The first run (before this fix; its store
and results have been deleted) had ~23% of memories dated 2026 and temporal
J of 26.5. The add step
refuses to resume into a store without the `OBSERVATION_DATE_FIX` marker.
The store now defaults to `MEM0_STORE_DIR=mem0_store_obsdate` in `.env`.

## 📋 Overview

This project evaluates Mem0 and compares it with different memory and retrieval techniques for AI systems:

1. **Established LOCOMO Benchmarks**: We evaluate against five established approaches from the literature: LoCoMo, ReadAgent, MemoryBank, MemGPT, and A-Mem.
2. **Open-Source Memory Solutions**: We test promising open-source memory architectures including LangMem, which provides flexible memory management capabilities.
3. **RAG Systems**: We implement Retrieval-Augmented Generation with various configurations, testing different chunk sizes and retrieval counts to optimize performance.
4. **Full-Context Processing**: We examine the effectiveness of passing the entire conversation history within the context window of the LLM as a baseline approach.
5. **Proprietary Memory Systems**: We evaluate OpenAI's built-in memory feature available in their ChatGPT interface to compare against commercial solutions.
6. **Third-Party Memory Providers**: We incorporate Zep, a specialized memory management platform designed for AI agents, to assess the performance of dedicated memory infrastructure.

We test these techniques on the LOCOMO dataset, which contains conversational data with various question types to evaluate memory recall and understanding.

## 🔍 Dataset

The LOCOMO dataset used in our experiments can be downloaded from our Google Drive repository:

[Download LOCOMO Dataset](https://drive.google.com/drive/folders/1L-cTjTm0ohMsitsHg4dijSPJtqNflwX-?usp=drive_link)

The dataset contains conversational data specifically designed to test memory recall and understanding across various question types and complexity levels.

Place the dataset files in the `dataset/` directory:
- `locomo10.json`: Original dataset
- `locomo10_rag.json`: Dataset formatted for RAG experiments

## 📁 Project Structure

```
.
├── src/                  # Source code for different memory techniques
│   ├── mem0/             # Implementation of the Mem0 technique
│   ├── openai/           # Implementation of the OpenAI memory
│   ├── zep/              # Implementation of the Zep memory
│   ├── rag.py            # Implementation of the RAG technique
│   └── langmem.py        # Implementation of the Language-based memory
├── metrics/              # Code for evaluation metrics
├── results/              # Results of experiments
├── dataset/              # Dataset files
├── evals.py              # Evaluation script
├── run_experiments.py    # Script to run experiments
├── generate_scores.py    # Script to generate scores from results
└── prompts.py            # Prompts used for the models
```

## 🚀 Getting Started

### Prerequisites

Create a `.env` file with your API keys and configurations. The following keys are required:

```
# OpenAI API key for GPT models and embeddings
OPENAI_API_KEY="your-openai-api-key"

# Mem0 API keys (for Mem0 and Mem0+ techniques)
MEM0_API_KEY="your-mem0-api-key"
MEM0_PROJECT_ID="your-mem0-project-id"
MEM0_ORGANIZATION_ID="your-mem0-organization-id"

# Model configuration
MODEL="gpt-4o-mini"  # or your preferred model
EMBEDDING_MODEL="text-embedding-3-small"  # or your preferred embedding model
ZEP_API_KEY="api-key-from-zep"
```

### Running Experiments

You can run experiments using the provided Makefile commands:

#### Memory Techniques

```bash
# Run Mem0 experiments
make run-mem0-add         # Add memories using Mem0
make run-mem0-search      # Search memories using Mem0

# Run Mem0+ experiments (with graph-based search)
make run-mem0-plus-add    # Add memories using Mem0+
make run-mem0-plus-search # Search memories using Mem0+

# Run RAG experiments
make run-rag              # Run RAG with chunk size 500
make run-full-context     # Run RAG with full context

# Run LangMem experiments
make run-langmem          # Run LangMem

# Run Zep experiments
make run-zep-add          # Add memories using Zep
make run-zep-search       # Search memories using Zep

# Run OpenAI experiments
make run-openai           # Run OpenAI experiments
```

Alternatively, you can run experiments directly with custom parameters:

```bash
python run_experiments.py --technique_type [mem0|rag|langmem] [additional parameters]
```

#### Command-line Parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--technique_type` | Memory technique to use (mem0, rag, langmem) | mem0 |
| `--method` | Method to use (add, search) | add |
| `--chunk_size` | Chunk size for processing | 1000 |
| `--top_k` | Number of top memories to retrieve | 30 |
| `--filter_memories` | Whether to filter memories | False |
| `--is_graph` | Whether to use graph-based search | False |
| `--num_chunks` | Number of chunks to process for RAG | 1 |

### 📊 Evaluation

To evaluate results, run:

```bash
python evals.py --input_file [path_to_results] --output_file [output_path]
```

This script:
1. Processes each question-answer pair
2. Calculates BLEU and F1 scores automatically
3. Uses an LLM judge to evaluate answer correctness
4. Saves the combined results to the output file

### 🎭 Adversarial questions (LOCOMO category 5)

Not part of the paper: the paper and `evals.py` skip category 5 because those questions
have no gold answer. Each one asks about something that is not true of the person named
(usually a fact about the other speaker); the dataset's `adversarial_answer` is the trap.
`eval_adversarial.py` judges only these 446 answers with a separate, strict judge
(`evaluate_adversarial_judge` in `metrics/llm_judge.py`), which sees the question, the trap
answer, the evidence turn with its real speaker, and the generated answer.

| Label | Meaning | Scored as |
|---|---|---|
| `DECLINED` | says it is not mentioned, or rejects the premise | correct |
| `REATTRIBUTED` | gives the fact but assigns it to the right person | correct |
| `TRAP` | gives the trap answer (or a paraphrase) for the person asked about | wrong |
| `OTHER` | accepts the premise and gives a different answer | wrong |

The judge does not pick the label itself. It answers three yes/no checks (does the answer
state the trap, does it explicitly refuse or deny the premise, does it move the fact to
another person) and `metrics/adversarial_rules.py` derives the label. A refusal counts
only if the words the judge quotes for it are really in the answer and contain a refusal
word.

```bash
sbatch eval_adversarial.slurm                            # on Spartan; log in slurm_logs/
make eval-adversarial HAND=adversarial_hand_labels.json  # or directly
```

This writes `results/mem0_adversarial_metrics.json` and, with `HAND=`, compares the judge
with `adversarial_hand_labels.json` (hand labels for the 446 answers of the committed run:
39.0% correct; a short answer that only names the other person's thing counts as TRAP,
and so does an answer that gives the trap fact with the altered detail corrected, such as
"a necklace from grandma" when asked about grandpa's gift) and writes the disagreements next to it. `python eval_adversarial.py
--report_only --hand_labels adversarial_hand_labels.json` re-derives the labels from the
stored checks without calling the judge. The Table 1 metrics file is not changed.

### 📈 Generating Scores

Generate final scores with:

```bash
python generate_scores.py
```

This script:
1. Loads the evaluation metrics data
2. Calculates mean scores for each category (BLEU, F1, LLM)
3. Reports the number of questions per category
4. Calculates overall mean scores across all categories

Example output:
```
Mean Scores Per Category:
         bleu_score  f1_score  llm_score  count
category                                       
1           0.xxxx    0.xxxx     0.xxxx     xx
2           0.xxxx    0.xxxx     0.xxxx     xx
3           0.xxxx    0.xxxx     0.xxxx     xx

Overall Mean Scores:
bleu_score    0.xxxx
f1_score      0.xxxx
llm_score     0.xxxx
```

## 📏 Evaluation Metrics

We use several metrics to evaluate the performance of different memory techniques:

1. **BLEU Score**: Measures the similarity between the model's response and the ground truth
2. **F1 Score**: Measures the harmonic mean of precision and recall
3. **LLM Score**: A binary score (0 or 1) determined by an LLM judge evaluating the correctness of responses
4. **Token Consumption**: Number of tokens required to generate final answer.
5. **Latency**: Time required during search and to generate response.

## 📚 Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@article{mem0,
  title={Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory},
  author={Chhikara, Prateek and Khant, Dev and Aryan, Saket and Singh, Taranjeet and Yadav, Deshraj},
  journal={arXiv preprint arXiv:2504.19413},
  year={2025}
}
```

## 📄 License

[MIT License](LICENSE)

## 👥 Contributors

- [Prateek Chhikara](https://github.com/prateekchhikara)
- [Dev Khant](https://github.com/Dev-Khant)
- [Saket Aryan](https://github.com/whysosaket)
- [Taranjeet Singh](https://github.com/taranjeet)
- [Deshraj Yadav](https://github.com/deshraj)

