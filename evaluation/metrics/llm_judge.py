import argparse
import json
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from mem0.memory.utils import extract_json

try:
    from metrics.adversarial_rules import ADVERSARIAL_CORRECT, ADVERSARIAL_LABELS, label_from_checks  # noqa: F401
except ImportError:  # run as a script from inside metrics/
    from adversarial_rules import ADVERSARIAL_CORRECT, ADVERSARIAL_LABELS, label_from_checks  # noqa: F401

load_dotenv()
client = OpenAI()

ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def evaluate_llm_judge(question, gold_answer, generated_answer):
    """Evaluate the generated answer against the gold answer using an LLM judge."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": ACCURACY_PROMPT.format(
                    question=question, gold_answer=gold_answer, generated_answer=generated_answer
                ),
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    label = json.loads(extract_json(response.choices[0].message.content))["label"]
    return 1 if label == "CORRECT" else 0


# ---------------------------------------------------------------------------
# Adversarial questions (LOCOMO category 5)
#
# These questions have no gold answer. Each asks about something that is not
# true of the person named in the question; `adversarial_answer` is the "trap":
# the answer to the original question the adversarial one was built from
# (usually a fact about the other speaker). The right behaviour is to decline
# or to correct the premise, so the gold-answer judge above cannot score them.
# Not part of the paper's evaluation (the paper excludes category 5).
# ---------------------------------------------------------------------------


ADVERSARIAL_PROMPT = """
You are checking an answer to an ADVERSARIAL question about a conversation between two people.

The question is built on a false premise: what it asks about is NOT true of the person named in the question. It was made by taking a real fact from the conversation and changing the person or a detail. You are given:
    (1) the question,
    (2) the trap answer: the answer that would be right only if the false premise were true,
    (3) the evidence: the original conversation line(s) the trap answer comes from, with the real speaker,
    (4) the generated answer to check.

Answer three yes/no checks about the GENERATED ANSWER. Read only what it literally says.

CHECK 1 - gives_trap: Does the generated answer state the trap answer as true for the person or thing named in the question?
  Yes if it gives the trap answer, the same idea in other words, a more general or more specific version of it, or part of a list in it. Examples: trap "strawberry", answer "berries" -> yes. Trap "daily brushing, baths and nail trims", answer "grooming and regular care" -> yes. Trap "glad", answer "excited to take part" -> yes. Also yes if the answer hedges but still offers it ("not sure, but probably X").
  No if the answer denies it, gives it for a different person, or says something else.
  If the trap answer is "Yes" or "No": yes only if the generated answer agrees with it as a statement of fact.

CHECK 2 - explicit_refusal: Does the generated answer explicitly say that the information is missing or that the premise is false?
  Yes only if the answer contains such a statement in its own words: for example "not mentioned", "no information", "unknown", "X does not have a tattoo", "X did not go on a trip", "no kitten was adopted by X", "X has dogs, not cats". Copy those words into refusal_quote.
  No if the answer simply gives some answer. A DIFFERENT ANSWER IS NOT A REFUSAL. Example: question "What instrument does Ana play?", trap "violin", answer "Acoustic guitar" -> no: it accepts that Ana plays an instrument and says nothing is wrong with the question. Example: trap "a bookmark", answer "Something yummy" -> no. If you cannot copy refusing words from the answer, the check is no.

CHECK 3 - reassigns: Does the generated answer explicitly make a DIFFERENT person or thing than the one named in the question the subject of the fact?
  Yes for: question "Why did Ana get a third turtle?", answer "Ben got a third turtle". Yes for: question "What did Ana find relaxing about her trip?", answer "Ana asked about Ben's trip".
  No if the other person is only mentioned as part of an answer about the person in the question. Example: question "Who inspired Ana to volunteer?", answer "Ben inspired Ana" -> no.

Question: {question}
Trap answer: {trap_answer}
Evidence:
{evidence}
Generated answer: {generated_answer}

Return a json object with exactly these keys:
  "reasoning": one short sentence,
  "gives_trap": true or false,
  "explicit_refusal": true or false,
  "refusal_quote": the refusing words copied from the generated answer, or "" if there are none,
  "reassigns": true or false.
"""


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "yes", "false", "no"):
        return value.strip().lower() in ("true", "yes")
    raise ValueError(f"adversarial judge returned a non-boolean check: {value!r}")


def evaluate_adversarial_judge(question, trap_answer, generated_answer, evidence=None, model="gpt-4o-mini", llm=None):
    """Label an answer to an adversarial (category 5) question.

    The judge answers three yes/no checks and the label is derived from them in
    code (metrics/adversarial_rules.py). Asking for the label directly made gpt-4o-mini
    call any answer that differed from the trap "DECLINED".

    `evidence` is a list of "Speaker: text" lines for the question's evidence
    turns. Returns (label, reasoning, checks); label is one of ADVERSARIAL_LABELS.
    """
    evidence_text = "\n".join(f"- {line}" for line in (evidence or [])) or "(not available)"
    response = (llm or client).chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": ADVERSARIAL_PROMPT.format(
                    question=question,
                    trap_answer=trap_answer,
                    evidence=evidence_text,
                    generated_answer=generated_answer,
                ),
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    parsed = json.loads(extract_json(response.choices[0].message.content))
    checks = {key: _as_bool(parsed.get(key)) for key in ("gives_trap", "explicit_refusal", "reassigns")}
    checks["refusal_quote"] = str(parsed.get("refusal_quote") or "").strip()
    # The raw checks are returned (and stored) as the judge gave them; the label
    # applies the quote rule, so a "refusal" the answer does not contain is ignored.
    return label_from_checks(generated_answer, checks), str(parsed.get("reasoning", "")).strip(), checks


def main():
    """Main function to evaluate RAG results using LLM judge."""
    parser = argparse.ArgumentParser(description="Evaluate RAG results using LLM judge")
    parser.add_argument(
        "--input_file",
        type=str,
        default="results/default_run_v4_k30_new_graph.json",
        help="Path to the input dataset file",
    )

    args = parser.parse_args()

    dataset_path = args.input_file
    output_path = f"results/llm_judge_{dataset_path.split('/')[-1]}"

    with open(dataset_path, "r") as f:
        data = json.load(f)

    LLM_JUDGE = defaultdict(list)
    RESULTS = defaultdict(list)

    index = 0
    for k, v in data.items():
        for x in v:
            question = x["question"]
            gold_answer = x["answer"]
            generated_answer = x["response"]
            category = x["category"]

            # Skip category 5
            if int(category) == 5:
                continue

            # Evaluate the answer
            label = evaluate_llm_judge(question, gold_answer, generated_answer)
            LLM_JUDGE[category].append(label)

            # Store the results
            RESULTS[index].append(
                {
                    "question": question,
                    "gt_answer": gold_answer,
                    "response": generated_answer,
                    "category": category,
                    "llm_label": label,
                }
            )

            # Save intermediate results
            with open(output_path, "w") as f:
                json.dump(RESULTS, f, indent=4)

            # Print current accuracy for all categories
            print("All categories accuracy:")
            for cat, results in LLM_JUDGE.items():
                if results:  # Only print if there are results for this category
                    print(f"  Category {cat}: {np.mean(results):.4f} ({sum(results)}/{len(results)})")
            print("------------------------------------------")
        index += 1

    # Save final results
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=4)

    # Print final summary
    print("PATH: ", dataset_path)
    print("------------------------------------------")
    for k, v in LLM_JUDGE.items():
        print(k, np.mean(v))


if __name__ == "__main__":
    main()
