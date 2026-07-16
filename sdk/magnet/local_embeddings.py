"""
LocalEmbedder
-------------
On-device semantic embeddings using sentence-transformers (all-MiniLM-L6-v2).
No API key required. Model is downloaded once to ~/.agent-magnet/models/ on first use.

Falls back to BM25-style keyword matching if sentence-transformers is not installed
or the model cannot be downloaded (offline first run).
"""

from __future__ import annotations

import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL_DIR = Path.home() / ".agent-magnet" / "models"

_embedder: Any = None
_embedder_available: bool | None = None  # None = not yet tried


def _get_embedder() -> Any:
    global _embedder, _embedder_available
    if _embedder_available is not None:
        return _embedder  # already resolved (None if unavailable)

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        if not (_MODEL_DIR / _MODEL_NAME).exists():
            sys.stderr.write(
                "[agent-magnet] Downloading local embedding model (one-time, ~90 MB)...\n"
            )
            sys.stderr.flush()
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        # show_progress_bar is an encode()-time argument in current
        # sentence-transformers, not a constructor kwarg — passing it here
        # raised a TypeError on newer versions, which silently downgraded
        # every caller to the keyword-overlap fallback.
        _embedder = SentenceTransformer(_MODEL_NAME, cache_folder=str(_MODEL_DIR))
        _embedder_available = True
        logger.info(f"[magnet] Local embedder ready: {_MODEL_NAME}")
    except ImportError:
        logger.info(
            "[magnet] sentence-transformers not installed — "
            "using keyword fallback. Install with: pip install sentence-transformers"
        )
        _embedder_available = False
    except Exception as e:
        logger.warning(
            f"[magnet] Local embedder failed ({e}) — keyword fallback active."
        )
        _embedder_available = False

    return _embedder


def embed(text: str) -> list[float] | None:
    """Return a normalized 384-dim vector, or None if unavailable."""
    model = _get_embedder()
    if model is None:
        return None
    try:
        return model.encode(text, normalize_embeddings=True, show_progress_bar=False).tolist()
    except Exception as e:
        logger.debug(f"[embedder] encode failed: {e}")
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two pre-normalized vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def is_semantic_duplicate(text: str, candidates: list[str], threshold: float = 0.90) -> bool:
    """Return True if `text` is semantically near-identical to any candidate."""
    if not candidates:
        return False

    model = _get_embedder()
    if model is None:
        # Keyword fallback: Jaccard similarity on words
        text_words = set(re.findall(r"\w+", text.lower()))
        for c in candidates:
            c_words = set(re.findall(r"\w+", c.lower()))
            union = text_words | c_words
            if not union:
                continue
            if len(text_words & c_words) / len(union) > 0.65:
                return True
        return False

    try:
        vec = embed(text)
        if vec is None:
            return False
        for c in candidates:
            c_vec = embed(c)
            if c_vec and cosine_similarity(vec, c_vec) >= threshold:
                return True
    except Exception:
        pass
    return False


def rank_by_similarity(
    query: str,
    documents: list[dict],
    text_key: str = "text",
    top_k: int = 5,
) -> list[dict]:
    """
    Rank documents by similarity to query.
    Uses embedding vectors when available; keyword BM25-style when not.
    """
    if not documents:
        return []

    model = _get_embedder()

    if model is None:
        # BM25-style keyword fallback
        q_words = set(re.findall(r"\w+", query.lower()))
        scored: list[tuple[float, dict]] = []
        for doc in documents:
            doc_text = doc.get(text_key, "").lower()
            doc_words = set(re.findall(r"\w+", doc_text))
            hits = sum(1 for w in q_words if w in doc_text)
            score = hits / (math.log(max(len(doc_words), 1)) + 1)
            scored.append((score, doc))
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored[:top_k]]

    try:
        q_vec = embed(query)
        if q_vec is None:
            return documents[:top_k]
        scored_vecs: list[tuple[float, dict]] = []
        for doc in documents:
            d_vec = doc.get("_embedding") or embed(doc.get(text_key, ""))
            if d_vec:
                scored_vecs.append((cosine_similarity(q_vec, d_vec), doc))
            else:
                scored_vecs.append((0.0, doc))
        scored_vecs.sort(key=lambda x: -x[0])
        return [d for _, d in scored_vecs[:top_k]]
    except Exception as e:
        logger.debug(f"[embedder] rank_by_similarity failed: {e}")
        return documents[:top_k]
