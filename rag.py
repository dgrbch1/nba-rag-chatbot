"""RAG orchestration: retrieval + augmentation + generation.

This module exposes `chat_with_rag` which:
  1. Retrieves relevant items from a vector store
  2. Formats the augmented context
  3. Calls OpenAI chat to generate an answer using `gpt-4o-mini`

Error handling is applied to catch API failures and missing fields.
"""
from typing import List, Dict, Any
import os
import time
import logging

from openai import OpenAI

from embeddings import get_embedding
from retrieval import VectorStore

logger = logging.getLogger(__name__)


def _format_player_text(player: Dict[str, Any]) -> str:
    """Safe formatting for a player record into readable context text."""
    name = player.get("name", "Unknown")
    team = player.get("team", "Unknown")
    position = player.get("position", "Unknown")
    height = player.get("height", "Unknown")
    weight = player.get("weight", "Unknown")
    draft_year = player.get("draft_year", "Unknown")
    stats = player.get("career_stats", {}) or {}
    points = stats.get("points", "N/A")
    rebounds = stats.get("rebounds", "N/A")
    assists = stats.get("assists", "N/A")
    achievements = player.get("achievements", []) or []

    text = (
        f"Name: {name}\n"
        f"Team: {team}\n"
        f"Position: {position}\n"
        f"Height: {height}\n"
        f"Weight: {weight}\n"
        f"Draft Year: {draft_year}\n"
        f"Career Points: {points}\n"
        f"Career Rebounds: {rebounds}\n"
        f"Career Assists: {assists}\n"
        f"Achievements: {', '.join(achievements)}\n"
    )
    return text


def format_context(players: List[Dict[str, Any]]) -> str:
    """Create a single context blob from a list of player dicts."""
    if not players:
        return ""
    parts = ["Here is information about NBA players that may be relevant:\n"]
    for p in players:
        parts.append(_format_player_text(p))
    return "\n".join(parts)


def chat_with_rag(
    client: OpenAI,
    vector_store: VectorStore,
    players: List[Dict[str, Any]],
    user_query: str,
    top_k: int = 3,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """High-level RAG: retrieve, augment, generate.

    - Computes the query embedding once
    - Retrieves top_k players using the provided vector store
    - Sends a system prompt containing only the retrieved context
    - Returns the assistant's response or a friendly error message
    """
    if not user_query or not user_query.strip():
        return "Please enter a non-empty question."

    total_start = time.perf_counter()

    try:
        # RETRIEVE: compute query embedding once
        query_emb = get_embedding(client, user_query)
    except Exception as exc:
        logger.exception("Failed to compute query embedding")
        return f"Error computing query embedding: {exc}"

    # measure retrieval time
    retrieval_start = time.perf_counter()
    try:
        results = vector_store.search(query_emb, top_k=top_k)
    except Exception as exc:
        logger.exception("Retrieval failed")
        return f"Error during retrieval: {exc}"
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

    # Map result metadata (indexes) back to player dicts
    retrieved_players = []
    for meta, score in results:
        if isinstance(meta, int) and 0 <= meta < len(players):
            retrieved_players.append(players[meta])

    logger.info("Retrieved %d documents for the query", len(retrieved_players))

    # AUGMENT: format context
    context_block = format_context(retrieved_players)

    system_message = (
        "You are a helpful NBA assistant. Answer using ONLY the provided context. "
        "If the answer is not contained in the context, reply: 'I don't have that information.'\n\n"
        + context_block
    )

    # GENERATE: ask the model
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_query},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = resp.choices[0].message.content
    except Exception as exc:
        logger.exception("OpenAI chat completion failed")
        return f"Error generating response: {exc}"

    total_ms = (time.perf_counter() - total_start) * 1000.0
    # Log timings to console in a compact format
    logger.info("Retrieval: %.1f ms | Total: %.1f ms", retrieval_ms, total_ms)

    return answer
