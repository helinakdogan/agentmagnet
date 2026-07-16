"""
EpisodicStore
-------------
LAYER 2 — Episodic Layer

Stores important conversations in long-term memory.
Uses Qdrant for vector search; falls back to Redis sorted sets if Qdrant is unavailable.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import litellm  # type: ignore

try:
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.models import (  # type: ignore
        Distance, VectorParams, PointStruct, PayloadSchemaType,
        Filter, FieldCondition, MatchValue,
    )
    _HAS_QDRANT = True
except ImportError:
    _HAS_QDRANT = False

logger = logging.getLogger(__name__)

_EPISODIC_KEY_PREFIX = "magnet:episodic:"
_MAX_EPISODES = 50
_EPISODE_TTL = 60 * 60 * 24 * 90  # 90 days
_QDRANT_COLLECTION = "magnet_episodes"
_EMBEDDING_DIM = 1536  # text-embedding-3-small


class EpisodicStore:
    """
    Stores and recalls important conversations in episodic memory.

    Args:
        redis_client: Initialized Redis client. If None, relies on Qdrant.
        qdrant_url:   Qdrant cluster URL (optional).
        qdrant_api_key: Qdrant API key (optional).

    Behavior:
        - With Qdrant: semantic search via OpenAI embeddings + vector storage.
        - Without Qdrant: episodes are stored in Redis (ordered by importance
          at write time), but recall ranks them against the query using the
          local on-device embedding model — importance decides what gets
          kept, semantic similarity decides what's relevant right now.
        - Without both: same local semantic ranking over an in-memory list
          (for testing/development).
    """

    def __init__(
        self,
        redis_client: Any | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        openai_api_key: str | None = None,
    ) -> None:
        self._redis = redis_client
        self._qdrant: Any | None = None
        self._qdrant_available = False
        self._memory: dict[str, list[dict]] = {}
        self._openai_api_key = openai_api_key

        if qdrant_url:
            if not _HAS_QDRANT:
                logger.warning("EpisodicStore: qdrant_url provided but qdrant_client is not installed.")
            else:
                try:
                    self._qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                    
                    existing = [c.name for c in self._qdrant.get_collections().collections]
                    if _QDRANT_COLLECTION not in existing:
                        self._qdrant.create_collection(
                            collection_name=_QDRANT_COLLECTION,
                            vectors_config=VectorParams(
                                size=_EMBEDDING_DIM,
                                distance=Distance.COSINE,
                            ),
                        )
                        self._qdrant.create_payload_index(
                            collection_name=_QDRANT_COLLECTION,
                            field_name="tenant_id",
                            field_schema=PayloadSchemaType.KEYWORD,
                        )
                    self._qdrant_available = True
                    logger.info("EpisodicStore: Qdrant connected.")
                except Exception as e:
                    logger.warning(f"EpisodicStore: Qdrant connection failed, Redis fallback active: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_episode(
        self,
        tenant_id: str,
        messages: list[dict],
        summary: str | None = None,
        importance: float = 0.5,
    ) -> None:
        """
        Saves an important conversation episode to memory.

        Args:
            tenant_id:  Tenant ID in ``project_id:user_id`` format.
            messages:   List of conversation messages.
            summary:    Optional summary. Generated via _auto_summarize if omitted.
            importance: Importance score between 0.0-1.0. Skipped if below 0.4.
        """
        if importance < 0.4:
            logger.debug(
                f"EpisodicStore: importance={importance:.2f} < 0.4, skipped ({tenant_id})"
            )
            return

        episode: dict = {
            "tenant_id": tenant_id,
            "messages": messages[-6:],
            "summary": summary or self._auto_summarize(messages),
            "importance": importance,
            "stored_at": time.time(),
        }

        if self._qdrant_available:
            self._store_qdrant(tenant_id, episode)
        elif self._redis:
            self._store_redis(tenant_id, episode)
        else:
            self._store_memory(tenant_id, episode)

        logger.info(
            f"EpisodicStore: episode stored ({tenant_id}, importance={importance:.2f})"
        )

    def recall(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Retrieves episodic memories similar to the query.

        Args:
            tenant_id: Tenant ID in ``project_id:user_id`` format.
            query:     The semantic search query.
            top_k:     Maximum number of episodes to return.

        Returns:
            A list of the most relevant episode dictionaries.
        """
        if self._qdrant_available:
            return self._recall_qdrant(tenant_id, query, top_k)
        elif self._redis:
            return self._recall_redis(tenant_id, query, top_k)
        else:
            return self._recall_memory(tenant_id, query, top_k)

    # ------------------------------------------------------------------
    # Internal — storage backends
    # ------------------------------------------------------------------

    def _store_qdrant(self, tenant_id: str, episode: dict) -> None:
        """Stores the episode in Qdrant as vector + payload."""
        try:
            embedding = self._embed(episode["summary"])
            point_id = int(time.time() * 1000) % (2**63)
            self._qdrant.upsert(
                collection_name=_QDRANT_COLLECTION,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={**episode},
                    )
                ],
            )
        except Exception as e:
            logger.error(f"EpisodicStore: Qdrant write error: {e}")
            if self._redis:
                self._store_redis(tenant_id, episode)

    def _store_redis(self, tenant_id: str, episode: dict) -> None:
        """Stores the episode in Redis sorted sets using the importance score."""
        key = _EPISODIC_KEY_PREFIX + tenant_id
        self._redis.zadd(key, {json.dumps(episode, ensure_ascii=False): episode["importance"]})
        self._redis.zremrangebyrank(key, 0, -(_MAX_EPISODES + 1))
        self._redis.expire(key, _EPISODE_TTL)

    def _store_memory(self, tenant_id: str, episode: dict) -> None:
        """In-memory fallback storage for testing/development."""
        lst = self._memory.setdefault(tenant_id, [])
        lst.append(episode)
        lst.sort(key=lambda e: e["importance"], reverse=True)
        self._memory[tenant_id] = lst[:_MAX_EPISODES]

    def _recall_qdrant(self, tenant_id: str, query: str, top_k: int) -> list[dict]:
        """Performs a semantic search in Qdrant."""
        try:
            embedding = self._embed(query)
            response = self._qdrant.query_points(
                collection_name=_QDRANT_COLLECTION,
                query=embedding,
                query_filter=Filter(
                    must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
                ),
                limit=top_k,
            )
            return [hit.payload for hit in response.points]
        except Exception as e:
            logger.error(f"EpisodicStore: Qdrant search error: {e}")
            if self._redis:
                return self._recall_redis(tenant_id, query, top_k)
            return []

    def _recall_redis(self, tenant_id: str, query: str, top_k: int) -> list[dict]:
        """Ranks stored episodes against the query by semantic similarity
        using the local (on-device) embedding model — no API key required.
        Redis just holds the data here; relevance ranking happens in
        _rank_locally, not from the importance score alone."""
        key = _EPISODIC_KEY_PREFIX + tenant_id
        try:
            items = self._redis.zrevrange(key, 0, _MAX_EPISODES - 1)
            episodes = [json.loads(i) for i in items]
        except Exception as e:
            logger.error(f"EpisodicStore: Redis read error: {e}")
            return []
        return self._rank_locally(query, episodes, top_k)

    def _recall_memory(self, tenant_id: str, query: str, top_k: int) -> list[dict]:
        """In-memory fallback — same local semantic ranking as _recall_redis."""
        episodes = self._memory.get(tenant_id, [])
        return self._rank_locally(query, episodes, top_k)

    def _rank_locally(self, query: str, episodes: list[dict], top_k: int) -> list[dict]:
        """Semantic similarity ranking with the local embedding model
        (sentence-transformers, on-device, no API key) — falls back to a
        keyword-overlap score if that model isn't installed. Either way,
        this is a real relevance signal against the query, not a
        hardcoded-phrase gate deciding whether to look at all."""
        if not episodes:
            return []
        from .local_embeddings import rank_by_similarity
        return rank_by_similarity(query, episodes, text_key="summary", top_k=top_k)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _auto_summarize(self, messages: list[dict]) -> str:
        """
        Generates a quick summary without an LLM call by concatenating
        the recent user messages (max 300 characters).
        """
        user_msgs = [
            m["content"] for m in messages if m.get("role") == "user" and m.get("content")
        ]
        return " | ".join(user_msgs[-2:])[:300]

    def _embed(self, text: str) -> list[float]:
        """
        Generates a text-embedding-3-small vector via litellm.
        """
        kwargs: dict = {
            "model": "openai/text-embedding-3-small",
            "input": text,
        }
        if self._openai_api_key:
            kwargs["api_key"] = self._openai_api_key
        response = litellm.embedding(**kwargs)
        if isinstance(response.data[0], dict):
            return response.data[0]["embedding"]
        return response.data[0].embedding
