"""LLM judge for LOCOMO adversarial questions (category 5).

evals.py skips category 5 because those questions have no gold answer. This
script judges only them, with a separate four-label judge
(metrics/llm_judge.py: evaluate_adversarial_judge):

    DECLINED, REATTRIBUTED  -> correct
    TRAP, OTHER             -> wrong

It reads the search step's results file, looks up each question's evidence
turn(s) in the dataset, and writes one record per question. The paper's
metrics file and scores are not touched; this category is not in the paper.

Usage:
    python eval_adversarial.py --input_file results/mem0_results_top_30_filter_False_graph_False.json \
        --output_file results/mem0_adversarial_metrics.json
    # check the judge against hand labels:
    python eval_adversarial.py ... --hand_labels adversarial_hand_labels.json
    # re-derive labels from the stored checks and recompute the summary, without calling the judge:
    python eval_adversarial.py --output_file results/mem0_adversarial_metrics.json --report_only \
        --hand_labels adversarial_hand_labels.json
"""

import argparse
import concurrent.futures
import json
import time
from collections import Counter, defaultdict

from metrics.adversarial_rules import ADVERSARIAL_CORRECT as CORRECT
from metrics.adversarial_rules import ADVERSARIAL_LABELS as LABELS
from metrics.adversarial_rules import label_from_checks


def evidence_index(dataset):
    """{conversation index: {dia_id: "Speaker: text"}}"""
    index = {}
    for i, item in enumerate(dataset):
        turns = {}
        for value in item["conversation"].values():
            if isinstance(value, list):
                for turn in value:
                    turns[turn["dia_id"]] = f"{turn['speaker']}: {turn.get('text', '')}"
        index[i] = turns
    return index


def collect_questions(results, dataset):
    turns = evidence_index(dataset)
    questions = []
    for conv, items in results.items():
        for item in items:
            if str(item["category"]) != "5":
                continue
            evidence_ids = item.get("evidence") or []
            questions.append(
                {
                    "conversation": str(conv),
                    "question": str(item["question"]),
                    "adversarial_answer": str(item.get("adversarial_answer", "")),
                    "response": str(item["response"]),
                    "evidence_ids": evidence_ids,
                    "evidence": [turns[int(conv)][e] for e in evidence_ids if e in turns.get(int(conv), {})],
                }
            )
    return questions


def judge_one(q, model, max_retries):
    from metrics.llm_judge import evaluate_adversarial_judge  # needs OPENAI_API_KEY

    for attempt in range(max_retries):
        try:
            label, reasoning, checks = evaluate_adversarial_judge(
                q["question"], q["adversarial_answer"], q["response"], q["evidence"], model=model
            )
            return dict(q, label=label, correct=int(label in CORRECT), reasoning=reasoning, checks=checks)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2 ** (attempt + 1), 30))


def summarise(records):
    n = len(records)
    counts = Counter(r["label"] for r in records)
    print(f"\nAdversarial questions (category 5): {n}")
    print(f"{'Label':<14}{'Count':>7}{'Share':>9}")
    for label in LABELS:
        print(f"{label:<14}{counts[label]:>7}{100 * counts[label] / n:>8.1f}%")
    correct = sum(counts[label] for label in CORRECT)
    print(f"{'Correct':<14}{correct:>7}{100 * correct / n:>8.1f}%   (DECLINED + REATTRIBUTED)")


def cohen_kappa(pairs, labels):
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[label] * right[label] for label in labels) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def compare(records, hand_labels, disagreements_file):
    """Agreement between the judge and hand labels.

    A hand label applies only if it was given for the same answer text, so
    labels from a different run are ignored rather than silently reused.
    """
    hand = {(str(h["conversation"]), h["question"]): h for h in hand_labels}
    pairs, disagreements, stale = [], [], 0
    for r in records:
        h = hand.get((r["conversation"], r["question"]))
        if h is None:
            continue
        if h.get("response") is not None and h["response"].strip() != r["response"].strip():
            stale += 1
            continue
        pairs.append((h["label"], r["label"]))
        if h["label"] != r["label"]:
            disagreements.append(
                {
                    "conversation": r["conversation"],
                    "question": r["question"],
                    "adversarial_answer": r["adversarial_answer"],
                    "evidence": r["evidence"],
                    "response": r["response"],
                    "hand_label": h["label"],
                    "judge_label": r["label"],
                    "judge_reasoning": r.get("reasoning", ""),
                    "judge_checks": r.get("checks", {}),
                }
            )
    if not pairs:
        print("\nNo hand labels matched these answers (different run?).")
        return
    n = len(pairs)
    exact = sum(a == b for a, b in pairs)
    binary = [(a in CORRECT, b in CORRECT) for a, b in pairs]
    binary_agree = sum(a == b for a, b in binary)
    print(f"\nJudge vs hand labels: {n} answers compared" + (f" ({stale} skipped: answer text differs)" if stale else ""))
    print(f"  four-label agreement    {100 * exact / n:5.1f}%   Cohen's kappa {cohen_kappa(pairs, LABELS):.2f}")
    print(f"  correct/wrong agreement {100 * binary_agree / n:5.1f}%   Cohen's kappa {cohen_kappa(binary, (True, False)):.2f}")
    hand_acc = sum(a for a, _ in binary) / n
    judge_acc = sum(b for _, b in binary) / n
    print(f"  accuracy by hand labels {100 * hand_acc:5.1f}%   by judge {100 * judge_acc:5.1f}%")
    matrix = Counter(pairs)
    print("\n  rows = hand label, columns = judge label")
    print("  " + f"{'':<14}" + "".join(f"{label[:8]:>10}" for label in LABELS))
    for a in LABELS:
        print("  " + f"{a:<14}" + "".join(f"{matrix[(a, b)]:>10}" for b in LABELS))
    if disagreements_file:
        with open(disagreements_file, "w") as f:
            json.dump(disagreements, f, indent=2, ensure_ascii=False)
        print(f"\n  {len(disagreements)} disagreements written to {disagreements_file}")


def main():
    parser = argparse.ArgumentParser(description="LLM judge for LOCOMO adversarial questions (category 5)")
    parser.add_argument("--input_file", default="results/mem0_results_top_30_filter_False_graph_False.json")
    parser.add_argument("--dataset", default="dataset/locomo10.json")
    parser.add_argument("--output_file", default="results/mem0_adversarial_metrics.json")
    parser.add_argument("--model", default="gpt-4o-mini", help="Judge model")
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="Judge only the first N questions (smoke test)")
    parser.add_argument("--hand_labels", default=None, help="JSON list of {conversation, question, response, label}")
    parser.add_argument("--disagreements_file", default=None, help="Where to write judge/hand disagreements")
    parser.add_argument("--report_only", action="store_true", help="Reuse --output_file: re-derive labels from stored checks, no judge calls")
    args = parser.parse_args()

    if args.report_only:
        # Re-derive every label from the stored checks with the current rules
        # (no judge calls), and save the file if any label changed.
        with open(args.output_file) as f:
            by_conversation = json.load(f)
        records, changed = [], 0
        for items in by_conversation.values():
            for r in items:
                if "checks" in r:
                    label = label_from_checks(r["response"], r["checks"])
                    changed += label != r["label"]
                    r["label"], r["correct"] = label, int(label in CORRECT)
                records.append(r)
        if changed:
            with open(args.output_file, "w") as f:
                json.dump(by_conversation, f, indent=4, ensure_ascii=False)
            print(f"{changed} labels re-derived from stored checks; {args.output_file} updated")
    else:
        with open(args.input_file) as f:
            results = json.load(f)
        with open(args.dataset) as f:
            dataset = json.load(f)
        questions = collect_questions(results, dataset)
        if args.limit:
            questions = questions[: args.limit]
        print(f"Judging {len(questions)} adversarial answers with {args.model}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            records = list(executor.map(lambda q: judge_one(q, args.model, args.max_retries), questions))
        by_conversation = defaultdict(list)
        for r in records:
            by_conversation[r["conversation"]].append(r)
        with open(args.output_file, "w") as f:
            json.dump(by_conversation, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {args.output_file}")

    summarise(records)
    if args.hand_labels:
        with open(args.hand_labels) as f:
            hand_labels = json.load(f)
        default_out = args.output_file.replace(".json", "_disagreements.json")
        compare(records, hand_labels, args.disagreements_file or default_out)


if __name__ == "__main__":
    main()
