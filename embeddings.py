import os
import pickle
import json
import time
import logging
from typing import List, Tuple, Dict, Any

import numpy as np
from openai import OpenAI

# Embedding model to use
EMBEDDING_MODEL = "text-embedding-3-small"

logger = logging.getLogger(__name__)


def _player_to_text(player: Dict[str, Any]) -> str:
    """Convert a player dict into a single block of text (robust to missing fields)."""
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
        f"Achievements: {', '.join(achievements)}"
    )
    return text


def get_embedding(client: OpenAI, text: str) -> List[float]:
    """Get embedding for a single text using the OpenAI client.

    This is a single-call wrapper so callers compute the query embedding exactly once.
    """
    # The OpenAI client may raise API errors; let callers handle exceptions.
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def load_embeddings_cache(path: str):
    """Load embeddings cache from a pickle or json file.

    Returns a dict with keys: 'vectors' (list of list floats), 'player_ids' (list)
    or None if file not found or unreadable.
    """
    if not os.path.exists(path):
        logger.debug("Embeddings cache not found: %s", path)
        return None

    try:
        # Prefer pickle for speed/precision
        if path.endswith(".pkl") or path.endswith(".pickle"):
            with open(path, "rb") as f:
                data = pickle.load(f)
                logger.info("Loaded embeddings cache from %s (pickle)", path)
                return data

        # Support json fallback
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info("Loaded embeddings cache from %s (json)", path)
            return data
    except Exception as exc:
        logger.warning("Failed to load embeddings cache (%s): %s", path, exc)
        return None


def save_embeddings_cache(path: str, data: Dict[str, Any]):
    """Save embeddings cache to a file (pickle if extension suggests it).

    `data` should be JSON-serializable if using .json, otherwise pickle is used.
    """
    try:
        if path.endswith(".pkl") or path.endswith(".pickle"):
            with open(path, "wb") as f:
                pickle.dump(data, f)
                logger.info("Saved embeddings cache to %s (pickle)", path)
        else:
            # Convert numpy arrays to lists for json
            json_safe = {
                "vectors": [list(map(float, v)) for v in data["vectors"]],
                "player_ids": data["player_ids"],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(json_safe, f, indent=2)
                logger.info("Saved embeddings cache to %s (json)", path)
    except Exception as exc:
        logger.warning("Failed to save embeddings cache (%s): %s", path, exc)
        # Don't raise during save; caller can warn if needed
        pass


def build_or_load_player_embeddings(
    client: OpenAI,
    players: List[Dict[str, Any]],
    cache_path: str = "embeddings.pkl",
) -> Tuple[np.ndarray, List[int]]:
    """Load embeddings from cache or compute them and save to cache.

    Returns:
      - vectors: np.ndarray shape (N, dim)
      - player_ids: list of indices (0..N-1) matching players list order

    Embeddings are computed from a textual representation of each player.
    """
    cached = load_embeddings_cache(cache_path)
    if cached is not None:
        vectors = np.array(cached["vectors"], dtype=float)
        player_ids = cached.get("player_ids") or list(range(len(vectors)))
        logger.info("Embeddings cache hit: %s", cache_path)
        return vectors, player_ids

    logger.info("Embeddings cache miss; computing embeddings for %d players", len(players))
    start = time.perf_counter()
    vectors = []
    player_ids = []
    for i, player in enumerate(players):
        try:
            text = _player_to_text(player)
            emb = get_embedding(client, text)
            vectors.append(emb)
            player_ids.append(i)
        except Exception as exc:
            logger.warning("Failed to compute embedding for player index %d: %s", i, exc)

    vectors = np.array(vectors, dtype=float)
    elapsed = (time.perf_counter() - start) * 1000.0
    logger.info("Computed %d embeddings in %.1f ms", vectors.shape[0], elapsed)

    # Save cache (best-effort)
    to_save = {"vectors": vectors.tolist(), "player_ids": player_ids}
    save_embeddings_cache(cache_path, to_save)

    return vectors, player_ids
