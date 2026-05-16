"""
KnowledgeStore
--------------
LAYER 3 — Knowledge Layer

Graph-based long-term entity memory.
- Primary: Redis-backed entity storage (always available if Redis configured)
- Optional: Neo4j for graph relationships (if NEO4J_URL configured)

Stores entities like:
  - dislike: {"type": "dislike", "content": "red color", "dimension": "color_preference"}
  - like: {"type": "like", "content": "markdown format", "dimension": "formatting"}
  - personality: {"type": "personality", "content": "prefers a friendly tone"}
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

try:
    from neo4j import GraphDatabase  # type: ignore
    _HAS_NEO4J = True
except ImportError:
    _HAS_NEO4J = False

logger = logging.getLogger(__name__)

_ENTITY_PREFIX = "vmm:entity:"
_RELATION_PREFIX = "vmm:relations:"
_ENTITY_TTL = 60 * 60 * 24 * 90  # 90 days


class KnowledgeStore:
    """
    Graph-based long-term entity memory with Redis fallback.

    Storage hierarchy:
      1. Neo4j (if url + auth configured) — rich graph queries
      2. Redis (if redis_client provided) — fast hash-based entity storage
      3. In-memory dict (development/testing fallback)

    Args:
        neo4j_url:    Neo4j bolt URL (e.g. "neo4j+s://xxx.databases.neo4j.io").
        neo4j_auth:   (username, password) tuple.
        redis_client: Initialized Redis client for fallback storage.
    """

    def __init__(
        self,
        neo4j_url: str | None = None,
        neo4j_auth: tuple[str, str] | None = None,
        redis_client: Any | None = None,
    ) -> None:
        self._redis = redis_client
        self._neo4j_driver: Any | None = None
        self._neo4j_available = False
        self._memory: dict[str, list[dict]] = {}  # in-memory fallback

        # Attempt Neo4j connection
        if neo4j_url and neo4j_auth and _HAS_NEO4J:
            try:
                self._neo4j_driver = GraphDatabase.driver(neo4j_url, auth=neo4j_auth)
                # Verify connectivity
                self._neo4j_driver.verify_connectivity()
                self._neo4j_available = True
                logger.info("KnowledgeStore: Neo4j connected successfully.")
            except Exception as e:
                logger.warning(f"KnowledgeStore: Neo4j connection failed, Redis fallback active: {e}")
                self._neo4j_driver = None
        elif neo4j_url and not _HAS_NEO4J:
            logger.warning("KnowledgeStore: NEO4J_URL set but 'neo4j' package not installed. pip install neo4j")
        
        if not self._neo4j_available and redis_client:
            logger.info("KnowledgeStore: Using Redis-backed entity storage.")
        elif not self._neo4j_available and not redis_client:
            logger.info("KnowledgeStore: Using in-memory storage (development mode).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_entity(self, tenant_id: str, entity: dict) -> None:
        """
        Stores an entity in the knowledge graph.

        Args:
            tenant_id: Tenant ID in "project_id:user_id" format.
            entity: Entity dict with fields:
                - type: "dislike" | "like" | "personality" | "fact" | custom
                - content: Human-readable description (e.g. "red color", "formal tone")
                - dimension: Category slug (e.g. "color_preference")
                - confidence: Optional float
        """
        entity_type = entity.get("type", "unknown")
        content = entity.get("content", "")
        if not content:
            return

        enriched = {
            **entity,
            "stored_at": time.time(),
            "tenant_id": tenant_id,
        }

        if self._neo4j_available:
            self._store_neo4j(tenant_id, enriched)
        elif self._redis:
            self._store_redis(tenant_id, entity_type, enriched)
        else:
            self._store_memory(tenant_id, enriched)

        logger.debug(f"KnowledgeStore: entity stored ({tenant_id}, type={entity_type}): {content!r}")

    def query_entities(
        self,
        tenant_id: str,
        query: str = "",
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Retrieves entities from the knowledge graph.

        Args:
            tenant_id: Tenant ID in "project_id:user_id" format.
            query: Optional keyword to filter by (searches content field).
            entity_type: Optional type filter ("dislike", "like", "personality", etc.)
            limit: Maximum number of entities to return.

        Returns:
            List of entity dicts.
        """
        if self._neo4j_available:
            return self._query_neo4j(tenant_id, query, entity_type, limit)
        elif self._redis:
            return self._query_redis(tenant_id, query, entity_type, limit)
        else:
            return self._query_memory(tenant_id, query, entity_type, limit)

    def link_entities(
        self,
        tenant_id: str,
        from_entity: str,
        to_entity: str,
        relation: str,
    ) -> None:
        """
        Creates a relationship between two entities.

        Args:
            tenant_id: Tenant ID.
            from_entity: Source entity content/name.
            to_entity: Target entity content/name.
            relation: Relationship type (e.g., "PREFERS", "AVOIDS", "RELATED_TO").
        """
        relation_data = {
            "from": from_entity,
            "to": to_entity,
            "relation": relation,
            "tenant_id": tenant_id,
            "stored_at": time.time(),
        }

        if self._neo4j_available:
            self._link_neo4j(tenant_id, from_entity, to_entity, relation)
        elif self._redis:
            key = f"{_RELATION_PREFIX}{tenant_id}"
            try:
                self._redis.rpush(key, json.dumps(relation_data, ensure_ascii=False))
                self._redis.expire(key, _ENTITY_TTL)
            except Exception as e:
                logger.error(f"KnowledgeStore: Redis relation store error: {e}")
        else:
            self._memory.setdefault(f"relations:{tenant_id}", []).append(relation_data)

        logger.debug(f"KnowledgeStore: relation stored ({tenant_id}): {from_entity} -{relation}-> {to_entity}")

    def build_knowledge_injection(self, tenant_id: str) -> str:
        """
        Builds a knowledge context string from stored entities.
        Used by the MemoryOrchestrator to enrich system prompts.

        Returns:
            Formatted string of known entities, or empty string if none.
        """
        dislikes = self.query_entities(tenant_id, entity_type="dislike", limit=15)
        likes = self.query_entities(tenant_id, entity_type="like", limit=15)
        personality = self.query_entities(tenant_id, entity_type="personality", limit=10)

        if not dislikes and not likes and not personality:
            return ""

        lines = ["[Long-term Knowledge — Entity Memory]"]

        if dislikes:
            lines.append("\nKnown dislikes (avoid these):")
            for e in dislikes:
                lines.append(f"  ✗ {e.get('content', '')}")

        if likes:
            lines.append("\nKnown likes:")
            for e in likes:
                lines.append(f"  ✓ {e.get('content', '')}")

        if personality:
            lines.append("\nPersonality/behavior notes:")
            for e in personality:
                lines.append(f"  → {e.get('content', '')}")

        return "\n".join(lines)

    def delete_all(self, tenant_id: str) -> None:
        """Deletes all entities for a tenant."""
        if self._neo4j_available:
            self._delete_neo4j(tenant_id)
        elif self._redis:
            try:
                keys = self._redis.keys(f"{_ENTITY_PREFIX}{tenant_id}:*")
                if keys:
                    self._redis.delete(*keys)
                self._redis.delete(f"{_RELATION_PREFIX}{tenant_id}")
            except Exception as e:
                logger.error(f"KnowledgeStore: Redis delete error: {e}")
        else:
            keys_to_del = [k for k in self._memory if tenant_id in k]
            for k in keys_to_del:
                del self._memory[k]

    # ------------------------------------------------------------------
    # Redis backend
    # ------------------------------------------------------------------

    def _redis_key(self, tenant_id: str, entity_type: str) -> str:
        return f"{_ENTITY_PREFIX}{tenant_id}:{entity_type}"

    def _store_redis(self, tenant_id: str, entity_type: str, entity: dict) -> None:
        key = self._redis_key(tenant_id, entity_type)
        try:
            # Avoid exact duplicates
            existing_raw = self._redis.lrange(key, 0, -1)
            existing_contents = set()
            for raw in existing_raw:
                try:
                    parsed = json.loads(raw)
                    existing_contents.add(parsed.get("content", ""))
                except Exception:
                    pass

            content = entity.get("content", "")
            if content and content not in existing_contents:
                self._redis.rpush(key, json.dumps(entity, ensure_ascii=False))
                self._redis.expire(key, _ENTITY_TTL)
        except Exception as e:
            logger.error(f"KnowledgeStore: Redis store error: {e}")

    def _query_redis(
        self,
        tenant_id: str,
        query: str = "",
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        try:
            if entity_type:
                keys = [self._redis_key(tenant_id, entity_type)]
            else:
                keys = self._redis.keys(f"{_ENTITY_PREFIX}{tenant_id}:*")

            results = []
            for key in keys:
                raw_list = self._redis.lrange(key, 0, -1)
                for raw in raw_list:
                    try:
                        entity = json.loads(raw)
                        if not query or query.lower() in entity.get("content", "").lower():
                            results.append(entity)
                    except Exception:
                        pass
            return results[:limit]
        except Exception as e:
            logger.error(f"KnowledgeStore: Redis query error: {e}")
            return []

    # ------------------------------------------------------------------
    # In-memory fallback
    # ------------------------------------------------------------------

    def _store_memory(self, tenant_id: str, entity: dict) -> None:
        key = f"{tenant_id}:{entity.get('type', 'unknown')}"
        lst = self._memory.setdefault(key, [])
        existing_contents = {e.get("content", "") for e in lst}
        if entity.get("content", "") not in existing_contents:
            lst.append(entity)

    def _query_memory(
        self,
        tenant_id: str,
        query: str = "",
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        results = []
        for key, lst in self._memory.items():
            if tenant_id not in key:
                continue
            if entity_type and not key.endswith(f":{entity_type}"):
                continue
            for entity in lst:
                if not query or query.lower() in entity.get("content", "").lower():
                    results.append(entity)
        return results[:limit]

    # ------------------------------------------------------------------
    # Neo4j backend
    # ------------------------------------------------------------------

    def _store_neo4j(self, tenant_id: str, entity: dict) -> None:
        try:
            with self._neo4j_driver.session() as session:
                session.run(
                    """
                    MERGE (u:User {tenant_id: $tenant_id})
                    MERGE (e:Entity {content: $content, type: $type, tenant_id: $tenant_id})
                    SET e.dimension = $dimension,
                        e.confidence = $confidence,
                        e.stored_at = $stored_at
                    MERGE (u)-[:HAS_ENTITY]->(e)
                    """,
                    tenant_id=tenant_id,
                    content=entity.get("content", ""),
                    type=entity.get("type", "unknown"),
                    dimension=entity.get("dimension", "general"),
                    confidence=entity.get("confidence", 0.5),
                    stored_at=entity.get("stored_at", time.time()),
                )
        except Exception as e:
            logger.error(f"KnowledgeStore: Neo4j store error: {e}")
            # Fallback to Redis if available
            if self._redis:
                self._store_redis(tenant_id, entity.get("type", "unknown"), entity)

    def _query_neo4j(
        self,
        tenant_id: str,
        query: str = "",
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        try:
            with self._neo4j_driver.session() as session:
                if entity_type:
                    result = session.run(
                        """
                        MATCH (u:User {tenant_id: $tenant_id})-[:HAS_ENTITY]->(e:Entity {type: $type})
                        WHERE $query = '' OR toLower(e.content) CONTAINS toLower($query)
                        RETURN e ORDER BY e.stored_at DESC LIMIT $limit
                        """,
                        tenant_id=tenant_id,
                        type=entity_type,
                        query=query,
                        limit=limit,
                    )
                else:
                    result = session.run(
                        """
                        MATCH (u:User {tenant_id: $tenant_id})-[:HAS_ENTITY]->(e:Entity)
                        WHERE $query = '' OR toLower(e.content) CONTAINS toLower($query)
                        RETURN e ORDER BY e.stored_at DESC LIMIT $limit
                        """,
                        tenant_id=tenant_id,
                        query=query,
                        limit=limit,
                    )
                return [dict(record["e"]) for record in result]
        except Exception as e:
            logger.error(f"KnowledgeStore: Neo4j query error: {e}")
            if self._redis:
                return self._query_redis(tenant_id, query, entity_type, limit)
            return []

    def _link_neo4j(
        self,
        tenant_id: str,
        from_entity: str,
        to_entity: str,
        relation: str,
    ) -> None:
        try:
            with self._neo4j_driver.session() as session:
                session.run(
                    f"""
                    MATCH (a:Entity {{content: $from_entity, tenant_id: $tenant_id}})
                    MATCH (b:Entity {{content: $to_entity, tenant_id: $tenant_id}})
                    MERGE (a)-[:{relation}]->(b)
                    """,
                    tenant_id=tenant_id,
                    from_entity=from_entity,
                    to_entity=to_entity,
                )
        except Exception as e:
            logger.error(f"KnowledgeStore: Neo4j link error: {e}")

    def _delete_neo4j(self, tenant_id: str) -> None:
        try:
            with self._neo4j_driver.session() as session:
                session.run(
                    """
                    MATCH (u:User {tenant_id: $tenant_id})-[:HAS_ENTITY]->(e:Entity)
                    DETACH DELETE e
                    """,
                    tenant_id=tenant_id,
                )
        except Exception as e:
            logger.error(f"KnowledgeStore: Neo4j delete error: {e}")
