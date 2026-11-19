"""Answer LOCOMO adversarial questions (category 5) from a finished Mem0 Cloud run.

mem0's benchmark suite (mem0/evaluation = mem0ai/memory-benchmarks) skips
category 5: those questions have no gold answer, so benchmarks.locomo.run
cannot load them. This script fills the gap without touching that suite:

  1. reads the Mem0 user ids the main run saved (_ingestion_<conv>.json), so
     the memories already in Mem0 Cloud are reused and nothing is re-ingested;
  2. for each category-5 question, searches Mem0 and generates an answer with
     the suite's own search call, answer prompt and answerer model;
  3. writes the answers in the format ../evaluation/eval_adversarial.py reads,
     which then judges them (DECLINED / REATTRIBUTED = correct, TRAP / OTHER = wrong).

Run through run_adversarial_cloud.slurm (needs PYTHONPATH = mem0/evaluation).
"""

import argparse
import asyncio
import json
from pathlib import Path

from benchmarks.common.llm_client import LLMClient
from benchmarks.common.mem0_client import Mem0Client, format_search_results
from benchmarks.locomo.prompts import get_answer_generation_prompt
from benchmarks.locomo.run import get_sorted_sessions  # also loads mem0/evaluation/.env


async def answer_question(qa, user_id, reference_date, mem0, answerer, top_k):
    question = qa["question"]
    formatted, _ = format_search_results(await mem0.search(question, user_id, top_k=top_k))
    prompt = get_answer_generation_prompt(question, formatted[:top_k], reference_date=reference_date, user_profile=None)
    response = await answerer.generate(system="", user=prompt)
    if "ANSWER:" in response:
        response = response.rsplit("ANSWER:", 1)[-1].strip()
    return {
        "question": question,
        "category": 5,
        "adversarial_answer": qa.get("adversarial_answer", ""),
        "evidence": qa.get("evidence", []),
        "response": response,
        "num_memories": len(formatted[:top_k]),
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", required=True, help="results/locomo/predicted_<project-name> of the main run")
    parser.add_argument("--dataset-path", default="datasets/locomo/locomo10.json")
    parser.add_argument("--output-file", default=None, help="default: <run-dir>/adversarial_answers_top<k>.json")
    parser.add_argument("--answerer-model", default="gpt-5", help="same default as benchmarks.locomo.run")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--conversations", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--max-questions", type=int, default=None, help="per conversation (smoke test)")
    parser.add_argument("--max-workers", type=int, default=10, help="questions in flight at once")
    parser.add_argument("--rpm", type=int, default=200)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_file = Path(args.output_file or run_dir / f"adversarial_answers_top{args.top_k}.json")
    dataset = json.loads(Path(args.dataset_path).read_text())
    results = json.loads(output_file.read_text()) if output_file.exists() else {}

    mem0 = Mem0Client(mode="cloud", rpm=args.rpm)
    answerer = LLMClient(model=args.answerer_model, provider=args.provider, rpm=args.rpm)
    semaphore = asyncio.Semaphore(args.max_workers)

    async def bounded(qa, user_id, reference_date):
        async with semaphore:
            return await answer_question(qa, user_id, reference_date, mem0, answerer, args.top_k)

    try:
        for conv_idx in [int(c) for c in args.conversations.split(",")]:
            checkpoint = run_dir / f"_ingestion_{conv_idx}.json"
            if not checkpoint.exists():
                raise SystemExit(f"{checkpoint} not found: conversation {conv_idx} was not fully ingested by the main run")
            user_id = json.loads(checkpoint.read_text())["user_id"]

            entry = dataset[conv_idx]
            sessions = get_sorted_sessions(entry["conversation"])
            reference_date = sessions[-1][1] if sessions else None  # same as the main run

            questions = [qa for qa in entry["qa"] if qa.get("category") == 5]
            if args.max_questions is not None:
                questions = questions[: args.max_questions]
            done = {r["question"] for r in results.get(str(conv_idx), [])}
            todo = [qa for qa in questions if qa["question"] not in done]
            print(f"Conversation {conv_idx} ({user_id}): {len(todo)} to answer, {len(done)} already done", flush=True)
            if not todo:
                continue

            answers = await asyncio.gather(*(bounded(qa, user_id, reference_date) for qa in todo), return_exceptions=True)
            failed = [a for a in answers if isinstance(a, Exception)]
            results.setdefault(str(conv_idx), []).extend(a for a in answers if not isinstance(a, Exception))
            output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            if failed:
                raise SystemExit(f"{len(failed)} questions failed in conversation {conv_idx} (first: {failed[0]!r}); rerun to retry them")
    finally:
        await mem0.close()

    total = sum(len(v) for v in results.values())
    empty = sum(r["num_memories"] == 0 for v in results.values() for r in v)
    print(f"{total} adversarial answers in {output_file}" + (f"   WARNING: {empty} had no memories retrieved" if empty else ""))


if __name__ == "__main__":
    asyncio.run(main())
