"""One-off: rebuild add_progress.json for a run made before resume support existed.

mem0 saves each add() call's messages to history.db (table `messages`, last 10
per speaker) only after that batch's memories are stored. For every speaker we
find the latest stored batch whose messages appear exactly once as consecutive
turns in the dataset; the turn count up to the end of that batch is that
speaker's progress. Batches are re-aligned to the add step's session/batch
boundaries.

Usage: python seed_add_progress.py [--dry-run] [--force]
"""

import argparse
import json
import os
import sqlite3
from itertools import groupby

from dotenv import load_dotenv

load_dotenv()

BATCH = 2


def session_keys(conv):
    return [k for k in conv if k not in ("speaker_a", "speaker_b") and "date" not in k and "timestamp" not in k]


def batch_ends(conv):
    """All valid 'turns done' values: batch boundaries within each session."""
    ends, offset = {0}, 0
    for k in session_keys(conv):
        n = len(conv[k])
        ends.update(offset + min(i + BATCH, n) for i in range(0, n, BATCH))
        offset += n
    return sorted(ends)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing add_progress.json")
    args = ap.parse_args()

    store = os.path.abspath(os.getenv("MEM0_STORE_DIR", "mem0_store"))
    out = os.path.join(store, "add_progress.json")
    if os.path.exists(out) and not (args.force or args.dry_run):
        raise SystemExit(f"{out} already exists; use --force to overwrite")

    data = json.load(open("dataset/locomo10.json"))
    db = sqlite3.connect(f"file:{os.path.join(store, 'history.db')}?mode=ro", uri=True)

    progress, total_done, total = {}, 0, 0
    for idx, item in enumerate(data):
        conv = item["conversation"]
        texts = [f"{t['speaker']}: {t['text']}" for k in session_keys(conv) for t in conv[k]]
        ends = batch_ends(conv)
        for spk in (conv["speaker_a"], conv["speaker_b"]):
            uid = f"{spk}_{idx}"
            rows = db.execute(
                "select created_at, content from messages where session_scope=? order by created_at",
                (f"user_id={uid}",),
            ).fetchall()
            batches = [[m for _, m in g] for _, g in groupby(rows, key=lambda r: r[0])]
            done = None
            for b in reversed(batches):
                hits = [i for i in range(len(texts) - len(b) + 1) if texts[i : i + len(b)] == b]
                if len(b) > 1 and len(hits) == 1:
                    done = hits[0] + len(b)
                    break
            if done is None:
                print(f"{uid:12s} could not locate progress -> will restart this speaker from 0")
                continue  # absent from progress => add step wipes and restarts this speaker
            done = max(e for e in ends if e <= done)
            progress[uid] = done
            total_done += done
            total += len(texts)
            print(f"{uid:12s} {done:4d}/{len(texts)} turns")

    all_turns = sum(len(conv[k]) for it in data for conv in [it["conversation"]] for k in session_keys(conv)) * 2
    print(f"\n~{100 * total_done / all_turns:.0f}% of speaker-turns already ingested")
    if args.dry_run:
        return
    with open(out, "w") as f:
        json.dump(progress, f, indent=2, sort_keys=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
