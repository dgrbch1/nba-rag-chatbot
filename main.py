"""
Main CLI for the NBA RAG Chatbot.

This script initializes the OpenAI client, precomputes or loads cached
embeddings for all players once at startup, builds an in-memory vector store,
and runs a simple REPL loop that uses RAG to answer questions.
"""

import json
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

from embeddings import build_or_load_player_embeddings
from retrieval import VectorStore
from rag import chat_with_rag

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_players(path: str = "basketball_players.json"):
    """Load player data from JSON file and return a list of dicts.

    Uses robust error handling to surface problems early.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Player data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment. Create a .env file.")
    # Initialize the OpenAI client using the key from `.env`.
    # No keys are stored in this repository; set `OPENAI_API_KEY` locally.
    client = OpenAI(api_key=api_key)
    logger.info("Loading player data...")
    players = load_players()
    logger.info("Loaded %d players", len(players))

    # Precompute embeddings once and cache to disk (embeddings.pkl)
    # Build or load the embedding cache. This avoids repeated/expensive API calls
    # and speeds up development for recruiters running the demo locally.
    logger.info("Loading or computing embeddings (this runs once)...")
    vectors, player_ids = build_or_load_player_embeddings(
        client, players, cache_path="embeddings.pkl"
    )
    logger.info("Embeddings ready: %d vectors, dim=%d", vectors.shape[0], vectors.shape[1])

    # Build in-memory vector store using player index as metadata
    # `VectorStore` provides a simple nearest-neighbor lookup over embeddings.
    vector_store = VectorStore(vectors, metadatas=player_ids)

    logger.info("NBA RAG Chatbot — ask questions, type 'exit' to quit.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        # Ignore empty input
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Use the RAG pipeline: retrieve, augment, generate
        try:
            # `chat_with_rag` does the heavy lifting: it retrieves relevant
            # player records, formats them into a prompt, calls the LLM,
            # and returns a human-readable answer.
            answer = chat_with_rag(client, vector_store, players, user_input, top_k=3)
            print("\nChatbot:")
            print(answer)
            print()
        except Exception as exc:
            logger.exception("Unhandled error while processing query")
            print(f"Error: {exc}")
            print()


if __name__ == "__main__":
    main()
