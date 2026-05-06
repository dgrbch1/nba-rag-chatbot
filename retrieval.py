"""Retrieval utilities: fast cosine similarity search and optional FAISS support.

This module exposes a simple VectorStore class that accepts precomputed vectors
and metadata and performs efficient top-k similarity search.
"""
from typing import List, Tuple, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Try optional FAISS import; keep fallback to numpy if missing
try:
    import faiss
    _HAS_FAISS = True
    logger.info("FAISS is available; VectorStore can use it for fast search")
except Exception:
    faiss = None
    _HAS_FAISS = False
    logger.info("FAISS not available; using numpy-based search")


def _normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize vectors along axis=1."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return v / norms


class VectorStore:
    """In-memory vector store using numpy. Optionally uses FAISS if available.

    - vectors: 2D numpy array (N, D)
    - metadatas: list of metadata objects (e.g., player dicts or indices)
    """

    def __init__(self, vectors: np.ndarray, metadatas: List[Any]):
        self.vectors = np.array(vectors, dtype=float)
        self.metadatas = list(metadatas)
        self._norm_vectors = _normalize(self.vectors)

        # Optional FAISS index for faster search on larger datasets
        self._use_faiss = False
        self._faiss_index = None
        if _HAS_FAISS and self.vectors.size > 0:
            try:
                dim = self.vectors.shape[1]
                # FAISS prefers float32
                vecs32 = self._norm_vectors.astype('float32')
                index = faiss.IndexFlatIP(dim)
                index.add(vecs32)
                self._faiss_index = index
                self._use_faiss = True
                logger.info("Built FAISS index with %d vectors (dim=%d)", vecs32.shape[0], dim)
            except Exception as exc:
                logger.warning("Failed to build FAISS index, falling back to numpy: %s", exc)
                self._use_faiss = False

    def search(self, query_vector: List[float], top_k: int = 3) -> List[Tuple[Any, float]]:
        """Return top_k (metadata, score) pairs ordered by descending similarity."""
        q = np.array(query_vector, dtype=float)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        # normalize single query
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

        if self._use_faiss and self._faiss_index is not None:
            # FAISS expects float32 and returns (distances, indices)
            q32 = q_norm.astype('float32')
            distances, indices = self._faiss_index.search(q32, top_k)
            results = []
            for idx, score in zip(indices[0], distances[0]):
                if idx < 0 or idx >= len(self.metadatas):
                    continue
                results.append((self.metadatas[idx], float(score)))
            return results

        # numpy fallback: compute dot product with all normalized vectors to get cosine similarity
        scores = (self._norm_vectors @ q_norm.T).squeeze()

        # handle case where only one vector present
        if scores.ndim == 0:
            scores = np.array([scores])

        # get top k indices
        top_k = min(top_k, len(scores))
        idx = np.argpartition(-scores, range(top_k))[:top_k]
        # sort top indices by score
        idx = idx[np.argsort(-scores[idx])]

        results = [(self.metadatas[i], float(scores[i])) for i in idx]
        return results
