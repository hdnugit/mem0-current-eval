"""Summarise evals.py output in the layout of Table 1 of the Mem0 paper.

F1 and BLEU-1 are deterministic and reported as means (x100).
J (LLM-as-a-Judge) is reported as mean +/- std across the independent judge
runs stored in each item's "llm_scores" (see `evals.py --num_runs`): accuracy
is computed per run, then averaged across runs.
"""

import argparse
import json

import numpy as np
import pandas as pd

# LOCOMO category ids -> Table 1 column names (category 5, adversarial, is excluded).
CATEGORY_NAMES = {1: "Multi-Hop", 2: "Temporal", 3: "Open Domain", 4: "Single Hop"}
TABLE_ORDER = ["Single Hop", "Multi-Hop", "Open Domain", "Temporal"]


def j_stats(group):
    runs = np.array(group["llm_scores"].tolist(), dtype=float)  # (questions, runs)
    per_run = runs.mean(axis=0) * 100
    std = per_run.std(ddof=1) if len(per_run) > 1 else 0.0
    return per_run.mean(), std, len(per_run)


def summarise(df):
    f1 = df["f1_score"].mean() * 100
    b1 = df["bleu_score"].mean() * 100
    j_mean, j_std, n_runs = j_stats(df)
    return {"F1": f1, "B1": b1, "J": j_mean, "J_std": j_std, "runs": n_runs, "count": len(df)}


def main():
    parser = argparse.ArgumentParser(description="Generate Table 1 style scores")
    parser.add_argument("--input_file", default="evaluation_metrics.json", help="Output of evals.py")
    args = parser.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)

    df = pd.DataFrame([item for items in data.values() for item in items])
    df["category"] = pd.to_numeric(df["category"])
    df = df[df["category"].isin(CATEGORY_NAMES)]
    if "llm_scores" not in df:  # output from an older evals.py with a single judge run
        df["llm_scores"] = df["llm_score"].apply(lambda x: [x])

    rows = {}
    for cat, group in df.groupby("category"):
        rows[CATEGORY_NAMES[cat]] = summarise(group)
    rows["Overall"] = summarise(df)

    print(f"{'Category':<12} {'F1':>7} {'B1':>7} {'J':>16} {'n':>6}")
    for name in TABLE_ORDER + ["Overall"]:
        if name not in rows:
            continue
        r = rows[name]
        j = f"{r['J']:.2f} ± {r['J_std']:.2f}"
        print(f"{name:<12} {r['F1']:7.2f} {r['B1']:7.2f} {j:>16} {r['count']:6d}")
    print(f"\nJ computed over {rows['Overall']['runs']} judge run(s).")


if __name__ == "__main__":
    main()
