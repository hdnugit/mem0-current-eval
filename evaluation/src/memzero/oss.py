"""Open-source (self-hosted) Mem0 backend for the paper's LOCOMO evaluation.

Replaces the hosted ``MemoryClient`` used in the original paper code with the
OSS ``Memory`` class, keeping the paper's settings where possible:
  - LLM: gpt-4o-mini (env MODEL), temperature 0
  - Embeddings: text-embedding-3-small (env EMBEDDING_MODEL)
  - The same custom instructions the paper set on the Mem0 platform project

Storage is a local, on-disk Qdrant database plus a SQLite history DB under
MEM0_STORE_DIR (default: ./mem0_store), so the `add` and `search` steps can run
as separate processes.
"""

import os
import threading

from dotenv import load_dotenv

load_dotenv()
# Opt out of mem0 telemetry before mem0 is imported (it reads this at import time).
os.environ.setdefault("MEM0_TELEMETRY", "False")

import contextlib  # noqa: E402
from datetime import datetime  # noqa: E402

import mem0.memory.main as _mem0_main  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from mem0 import Memory  # noqa: E402

# ---------------------------------------------------------------------------
# Observation-date fix
#
# mem0 v2's extraction prompt resolves relative time ("yesterday", "last week")
# against an "Observation Date". OSS Memory.add() never passes one (its
# `timestamp=` argument is platform-only and raises), so the prompt builder
# falls back to *today's* date, and LOCOMO memories (set in 2023) come out
# dated in the year of ingestion. We patch the prompt builder used by
# Memory._add_to_vector_store to take the session date from a thread-local,
# set by `observation_date(...)` around each add() call. add() builds the
# prompt synchronously in the calling thread, so per-thread state is safe with
# the add step's speaker threads. The mem0 package itself is not modified.
# ---------------------------------------------------------------------------
_obs = threading.local()
_original_prompt_builder = _mem0_main.generate_additive_extraction_prompt


def locomo_date(ts):
    """'1:56 pm on 8 May, 2023' -> '2023-05-08'."""
    return datetime.strptime(ts.strip(), "%I:%M %p on %d %B, %Y").date().isoformat()


@contextlib.contextmanager
def observation_date(iso_date):
    """Use `iso_date` as both Observation and Current Date for add() calls in this thread."""
    prev = getattr(_obs, "date", None)
    _obs.date = iso_date
    try:
        yield
    finally:
        _obs.date = prev


def _prompt_with_observation_date(*args, **kwargs):
    date = getattr(_obs, "date", None)
    if date is None:
        raise RuntimeError(
            "mem0 add() called without an observation date; wrap it in "
            "src.memzero.oss.observation_date(...) so relative dates resolve correctly"
        )
    kwargs["timestamp"] = date
    # Current Date = session date too, so no ingestion-day date is in the prompt at all.
    kwargs["current_date"] = date
    return _original_prompt_builder(*args, **kwargs)


_mem0_main.generate_additive_extraction_prompt = _prompt_with_observation_date

# Verbatim from the paper's src/memzero/add.py (sent to the platform via update_project).
CUSTOM_INSTRUCTIONS = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""

EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class _LockedClient:
    """Serialise every call into an embedded (local) Qdrant client.

    Local-mode Qdrant has no internal locking, and the paper's ingest adds
    memories for all 20 speakers concurrently. LLM/embedding calls stay
    parallel; only the (fast) vector-store calls are serialised.
    """

    def __init__(self, client):
        self._client = client
        self._lock = threading.RLock()

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        def locked(*args, **kwargs):
            with self._lock:
                return attr(*args, **kwargs)

        return locked


def store_dir():
    path = os.path.abspath(os.getenv("MEM0_STORE_DIR", "mem0_store"))
    os.makedirs(path, exist_ok=True)
    return path


def build_memory():
    qdrant_dir = os.path.join(store_dir(), "qdrant")
    os.makedirs(qdrant_dir, exist_ok=True)

    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    collection = os.getenv("MEM0_COLLECTION", "locomo")

    client = QdrantClient(path=qdrant_dir, force_disable_check_same_thread=True)

    config = {
        "llm": {
            "provider": "openai",
            "config": {"model": os.getenv("MODEL", "gpt-4o-mini"), "temperature": 0.0},
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": embedding_model, "embedding_dims": EMBEDDING_DIMS.get(embedding_model, 1536)},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection,
                "embedding_model_dims": EMBEDDING_DIMS.get(embedding_model, 1536),
                "client": client,
                "on_disk": True,
            },
        },
        "history_db_path": os.path.join(store_dir(), "history.db"),
        "custom_instructions": CUSTOM_INSTRUCTIONS,
    }
    memory = Memory.from_config(config)

    locked = _LockedClient(client)
    memory.vector_store.client = locked
    # Create the entity store now (it reuses vector_store.client) so it also gets the
    # locked client, and so worker threads don't race to create it lazily.
    _ = memory.entity_store
    return memory
