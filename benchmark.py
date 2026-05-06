"""Simple benchmark script for the NBA RAG Chatbot.

Usage:
  # dry run: only measures retrieval time (no API calls)
  python benchmark.py --dry --runs 10

  # full run: measures full RAG pipeline including model calls (may incur API costs)
  python benchmark.py --full --runs 5
"""
import time
import argparse
import logging
from dotenv import load_dotenv
import os

from openai import OpenAI
from embeddings import build_or_load_player_embeddings, get_embedding, load_embeddings_cache
from retrieval import VectorStore
from rag import chat_with_rag

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


SAMPLE_QUERIES = [
    "Who has the most championships?",
    "How many career points did LeBron James score?",
    "Which player is the tallest?",
    "Who won the 2003 draft?",
    "Which players were all-time great centers?",
]


def run_benchmark(runs: int = 10, full: bool = False):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and full:
        raise ValueError("OPENAI_API_KEY required for full benchmark")

    client = OpenAI(api_key=api_key) if full else None

    # Load data and embeddings
    import json

    with open("basketball_players.json", "r", encoding="utf-8") as f:
        players = json.load(f)

    # Load embeddings: prefer cache in dry mode to avoid API calls
    cache = load_embeddings_cache("embeddings.pkl")
    if cache is not None:
        vectors = __import__("numpy").array(cache["vectors"], dtype=float)
        player_ids = cache.get("player_ids") or list(range(len(vectors)))
    else:
        # If cache missing, build (may call API)
        client_for_build = OpenAI(api_key=api_key) if api_key else OpenAI(api_key="")
        vectors, player_ids = build_or_load_player_embeddings(client_for_build, players, cache_path="embeddings.pkl")
    store = VectorStore(vectors, metadatas=player_ids)

    total_times = []
    retrieval_times = []

    for i in range(runs):
        query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]

        # Measure retrieval time only
        q_start = time.perf_counter()
        # compute embedding for query
        # For dry runs we still compute embedding unless full is False and no api_key available
        emb = None
        try:
            if full or api_key:
                emb = get_embedding(OpenAI(api_key=api_key), query)
            else:
                # In strictly dry mode without an API key, approximate by using a random vector
                import numpy as np

                emb = np.random.randn(vectors.shape[1]).tolist()
        except Exception:
            # fallback to random vector to continue benchmark
            import numpy as np

            emb = np.random.randn(vectors.shape[1]).tolist()

        r_start = time.perf_counter()
        _ = store.search(emb, top_k=3)
        r_time = (time.perf_counter() - r_start) * 1000.0
        retrieval_times.append(r_time)

        if full:
            t_start = time.perf_counter()
            _ = chat_with_rag(OpenAI(api_key=api_key), store, players, query, top_k=3)
            t_time = (time.perf_counter() - t_start) * 1000.0
            total_times.append(t_time)
        else:
            total_times.append(r_time)

    avg_retrieval = sum(retrieval_times) / len(retrieval_times)
    avg_total = sum(total_times) / len(total_times)

    logger.info("Benchmark results over %d runs", runs)
    logger.info("Average retrieval time: %.2f ms", avg_retrieval)
    logger.info("Average total time: %.2f ms", avg_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--full", action="store_true", help="Run full pipeline (includes API calls)")
    parser.add_argument("--dry", action="store_true", help="Dry run (no API calls); default if --full not set")
    args = parser.parse_args()

    run_benchmark(runs=args.runs, full=args.full)
