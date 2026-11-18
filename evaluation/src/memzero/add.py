"""Ingest LOCOMO conversations into OSS Mem0 (paper's add step), with resume.

Changes from the paper's original script:
  * Resumable: after every successful batch, the number of dialogue turns
    ingested for that speaker is written to MEM0_STORE_DIR/add_progress.json.
    A rerun skips finished batches instead of wiping and starting over.
    Use `--fresh` (run_experiments.py) to wipe and start from zero.
  * Fails loudly: the original swallowed errors inside speaker threads and
    silently skipped the rest of that session. Now a persistent error stops
    the whole run (progress is kept) and is reported at the end.
  * Retries with exponential backoff for transient errors (rate limits,
    timeouts); stops immediately on fatal ones (no credit, bad API key).
  * Logs to MEM0_STORE_DIR/add.log.
  * Each session's date is given to mem0's extraction prompt as the
    Observation Date, so "yesterday"/"last week" resolve to 2023 session
    dates rather than the day of ingestion (see oss.py).
Batching, speaker roles, threading and session order are unchanged.
"""

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm

from src.memzero.oss import build_memory, locomo_date, observation_date, store_dir

load_dotenv()

FATAL_MARKERS = ("insufficient_quota", "invalid_api_key", "Incorrect API key", "billing")

logger = logging.getLogger("mem0_add")


def _setup_logging(path):
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s"))
    logger.addHandler(fh)


class StopRun(Exception):
    pass


class MemoryADD:
    def __init__(self, data_path=None, batch_size=2, is_graph=False, fresh=False, max_retries=6):
        if is_graph:
            raise NotImplementedError("Graph memory (Mem0g) is not set up for the OSS backend.")
        # OSS Memory; the paper's custom instructions are set in src/memzero/oss.py
        self.mem0_client = build_memory()
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph

        self.progress_path = os.path.join(store_dir(), "add_progress.json")
        _setup_logging(os.path.join(store_dir(), "add.log"))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._errors = []
        if fresh and os.path.exists(self.progress_path):
            os.remove(self.progress_path)
        # Refuse to resume into a store built before the observation-date fix:
        # its memories carry ingestion-day dates and would mix with correct ones.
        marker = os.path.join(store_dir(), "OBSERVATION_DATE_FIX")
        if os.path.exists(self.progress_path) and not os.path.exists(marker):
            raise SystemExit(
                f"{store_dir()} was ingested without the observation-date fix.\n"
                "Use a new MEM0_STORE_DIR, or rerun with --fresh to wipe and re-ingest."
            )
        open(marker, "w").write("session dates passed as mem0 Observation Date\n")
        self.progress = self._load_progress()
        logger.info("start: fresh=%s, %d speakers with saved progress", fresh, len(self.progress))

        if data_path:
            self.load_data()

    # ---------- progress ----------
    def _load_progress(self):
        if os.path.exists(self.progress_path):
            with open(self.progress_path) as f:
                return json.load(f)
        return {}

    def _set_progress(self, user_id, turns_done):
        with self._lock:
            self.progress[user_id] = turns_done
            tmp = self.progress_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.progress, f, indent=2, sort_keys=True)
            os.replace(tmp, self.progress_path)

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    # ---------- adding ----------
    def add_memory(self, user_id, message, metadata):
        for attempt in range(self.max_retries):
            try:
                # Session date -> extraction prompt's Observation Date (see oss.py).
                with observation_date(locomo_date(metadata["timestamp"])):
                    self.mem0_client.add(message, user_id=user_id, metadata=metadata)
                return
            except Exception as e:
                if any(m in str(e) for m in FATAL_MARKERS):
                    raise StopRun(f"fatal API error for {user_id}: {e}") from e
                if attempt == self.max_retries - 1:
                    raise
                wait = min(2 ** (attempt + 1), 60)
                logger.warning("%s attempt %d failed (%s); retrying in %ds", user_id, attempt + 1, e, wait)
                time.sleep(wait)

    def add_memories_for_speaker(self, user_id, messages, timestamp, offset, desc):
        """Add one session for one speaker. `offset` = turns in all earlier sessions."""
        try:
            done = self.progress.get(user_id, 0)
            starts = [i for i in range(0, len(messages), self.batch_size)
                      if offset + min(i + self.batch_size, len(messages)) > done]
            for i in tqdm(starts, desc=desc, leave=False):
                if self._stop.is_set():
                    return
                batch = messages[i : i + self.batch_size]
                self.add_memory(user_id, batch, metadata={"timestamp": timestamp})
                self._set_progress(user_id, offset + i + len(batch))
        except Exception as e:
            logger.error("STOPPING: %s failed at turn %d: %r", user_id, self.progress.get(user_id, 0), e)
            with self._lock:
                self._errors.append(f"{user_id}: {e!r}")
            self._stop.set()

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        # Speakers with no saved progress start clean (as in the paper's script).
        for uid in (speaker_a_user_id, speaker_b_user_id):
            if uid not in self.progress:
                self.mem0_client.delete_all(user_id=uid)
                self._set_progress(uid, 0)

        offset = 0
        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue
            if self._stop.is_set():
                return

            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]
            n = len(chats)

            if min(self.progress.get(speaker_a_user_id, 0), self.progress.get(speaker_b_user_id, 0)) >= offset + n:
                offset += n  # session already ingested for both speakers
                continue

            messages = []
            messages_reverse = []
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")

            # add memories for the two users on different threads
            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_a_user_id, messages, timestamp, offset, f"conv {idx} {key} {speaker_a}"),
            )
            thread_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_b_user_id, messages_reverse, timestamp, offset, f"conv {idx} {key} {speaker_b}"),
            )

            thread_a.start()
            thread_b.start()
            thread_a.join()
            thread_b.join()
            offset += n

        logger.info("conversation %d complete (%d turns)", idx, offset)
        print(f"Conversation {idx} complete")

    def process_all_conversations(self, max_workers=10):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_conversation, item, idx) for idx, item in enumerate(self.data)]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    self._errors.append(repr(e))
                    self._stop.set()

        total = sum(len(v) for item in self.data for k, v in item["conversation"].items()
                    if k.startswith("session_") and isinstance(v, list))
        done = sum(self.progress.values()) / 2
        if self._errors:
            msg = "\n  ".join(self._errors[:10])
            raise SystemExit(
                f"\nAdd step STOPPED (progress saved, ~{100 * done / total:.0f}% of turns done).\n  {msg}\n"
                f"Fix the problem (e.g. top up credit) and rerun the same command to resume.\n"
                f"Details: {os.path.join(store_dir(), 'add.log')}"
            )
        logger.info("all conversations complete")
        print("All conversations ingested.")
