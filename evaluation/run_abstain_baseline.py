"""Abstain baseline: Mem0 with an answer prompt that may decline.

Re-answers every LOCOMO question from the memories that the normal Mem0 run
already retrieved (stored per question in its results file), using
ANSWER_PROMPT_ABSTAIN (or, with --variant soft, ANSWER_PROMPT_ABSTAIN_SOFT) instead of
ANSWER_PROMPT. Retrieval is therefore
identical to the base run; only the answer prompt differs. The memory store is
not needed and no memories are added or searched.

The output has the same format as the search step's results file, so evals.py,
generate_scores.py and eval_adversarial.py work on it unchanged.

Usage:
    python run_abstain_baseline.py \
        --input_file results/mem0_results_top_30_filter_False_graph_False.json \
        --output_file results/mem0_abstain_results_top_30_filter_False_graph_False.json
    python run_abstain_baseline.py --output_file <file> --stats_only   # abstention counts only
"""

import argparse
import concurrent.futures
import json
import os
import time
from collections import defaultdict

CATEGORY_NAMES = {1: "Multi-Hop", 2: "Temporal", 3: "Open Domain", 4: "Single Hop", 5: "Adversarial"}


def render_prompt(item, speaker_1, speaker_2, variant="strict"):
    """Same rendering as src/memzero/search.py, with one of the abstain prompts."""
    from jinja2 import Template
    from prompts import ABSTAIN_PROMPTS

    memories_1 = [f"{m['timestamp']}: {m['memory']}" for m in item["speaker_1_memories"]]
    memories_2 = [f"{m['timestamp']}: {m['memory']}" for m in item["speaker_2_memories"]]
    return Template(ABSTAIN_PROMPTS[variant]).render(
        speaker_1_user_id=speaker_1,
        speaker_2_user_id=speaker_2,
        speaker_1_memories=json.dumps(memories_1, indent=4),
        speaker_2_memories=json.dumps(memories_2, indent=4),
        speaker_1_graph_memories=json.dumps(None, indent=4),
        speaker_2_graph_memories=json.dumps(None, indent=4),
        question=item["question"],
    )


def answer_one(client, model, item, speaker_1, speaker_2, max_retries, variant="strict"):
    prompt = render_prompt(item, speaker_1, speaker_2, variant)
    for attempt in range(max_retries):
        try:
            t1 = time.time()
            response = client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": prompt}], temperature=0.0
            )
            return dict(item, response=response.choices[0].message.content, response_time=time.time() - t1)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2 ** (attempt + 1), 60))


def is_abstention(response):
    from prompts import ABSTAIN_PHRASE

    return ABSTAIN_PHRASE.lower() in str(response).lower()


def print_stats(results):
    """How often the fixed abstain phrase was used, per category."""
    counts, abstained = defaultdict(int), defaultdict(int)
    for items in results.values():
        for item in items:
            category = int(item["category"])
            counts[category] += 1
            abstained[category] += is_abstention(item["response"])
    print(f"\n{'Category':<14}{'Questions':>10}{'Abstained':>11}{'Share':>8}")
    for category in sorted(counts):
        share = 100 * abstained[category] / counts[category]
        print(f"{CATEGORY_NAMES.get(category, category):<14}{counts[category]:>10}{abstained[category]:>11}{share:>7.1f}%")
    answerable = [c for c in counts if c != 5]
    n, a = sum(counts[c] for c in answerable), sum(abstained[c] for c in answerable)
    if n:
        print(f"{'Answerable':<14}{n:>10}{a:>11}{100 * a / n:>7.1f}%   (categories 1-4: abstaining here is a wrong answer)")


def main():
    parser = argparse.ArgumentParser(description="Mem0 abstain baseline: re-answer from stored retrievals")
    parser.add_argument("--input_file", default="results/mem0_results_top_30_filter_False_graph_False.json")
    parser.add_argument("--output_file", default=None, help="Default: results/mem0_abstain[_soft]_results_top_30_...json")
    parser.add_argument(
        "--variant", choices=("strict", "soft"), default="strict",
        help="strict: check the premise and decline when unsupported; soft: answer by default, "
        "decline only for a different person or an unrelated topic",
    )
    parser.add_argument("--dataset", default="dataset/locomo10.json")
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions per conversation (smoke test)")
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing output file and start over")
    parser.add_argument("--stats_only", action="store_true", help="Print abstention counts for --output_file and exit")
    args = parser.parse_args()
    if args.output_file is None:
        name = "mem0_abstain" if args.variant == "strict" else f"mem0_abstain_{args.variant}"
        args.output_file = f"results/{name}_results_top_30_filter_False_graph_False.json"

    if args.stats_only:
        with open(args.output_file) as f:
            print_stats(json.load(f))
        return

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI()
    model = os.getenv("MODEL", "gpt-4o-mini")

    with open(args.input_file) as f:
        base = json.load(f)
    with open(args.dataset) as f:
        dataset = json.load(f)

    # Resume: conversations already fully answered in the output file are kept.
    results = {}
    if os.path.exists(args.output_file) and not args.fresh:
        with open(args.output_file) as f:
            results = json.load(f)

    print(f"Abstain baseline ({args.variant} prompt): answering with {model} from retrievals in {args.input_file}")
    for conv, items in base.items():
        if args.limit:
            items = items[: args.limit]
        if len(results.get(conv, [])) == len(items):
            print(f"conversation {conv}: already done ({len(items)} questions)")
            continue
        conversation = dataset[int(conv)]["conversation"]
        speaker_1, speaker_2 = conversation["speaker_a"], conversation["speaker_b"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            results[conv] = list(
                executor.map(
                    lambda item: answer_one(client, model, item, speaker_1, speaker_2, args.max_retries, args.variant),
                    items,
                )
            )
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"conversation {conv}: {len(items)} questions answered")

    print(f"Results saved to {args.output_file}")
    print_stats(results)


if __name__ == "__main__":
    main()
