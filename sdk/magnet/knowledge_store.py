"""
KnowledgeStore
--------------
LAYER 3 — Knowledge Layer

Graph-based long-term entity memory.
Primary (and only production backend): Neo4j for structured entity relationships.

Stores structured relationships like:
  (Mushroom:Subject {category: "ingredient"})-[:REJECTED_BY]->(User)
  (Bold text:Subject {category: "formatting"})-[:DISLIKED_BY]->(User)
  (Bullet points:Subject {category: "formatting"})-[:PREFERRED_BY]->(User)

Entity dict fields:
  - subject: The entity name (e.g. "mushrooms", "bold text")
  - category: Subject type (ingredient, formatting, tone, behavior, etc.)
  - relationship: PREFERRED_BY | REJECTED_BY | DISLIKED_BY | EXPECTED_BY
  - context: Optional str or dict with extra context
  - confidence: Optional float
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

_TYPE_TO_REL = {
    "dislike": "REJECTED_BY",
    "like": "PREFERRED_BY",
    "personality": "EXPECTED_BY",
    "fact": "KNOWN_BY",
}

_REL_TO_TYPE: dict[str, str] = {v: k for k, v in _TYPE_TO_REL.items()}
_REL_TO_TYPE["DISLIKED_BY"] = "dislike"


class KnowledgeStore:
    """
    Graph-based long-term entity memory.

    Stores structured Subject→User relationships in Neo4j.
    In-memory dict is used as a development fallback only.

    Args:
        neo4j_url:    Neo4j bolt URL (e.g. "neo4j+s://xxx.databases.neo4j.io").
        neo4j_auth:   (username, password) tuple.
        redis_client: Deprecated — ignored. Redis storage removed from knowledge layer.
    """

    def __init__(
        self,
        neo4j_url: str | None = None,
        neo4j_auth: tuple[str, str] | None = None,
        redis_client: Any | None = None,
    ) -> None:
        self._neo4j_driver: Any | None = None
        self._neo4j_available = False
        self._memory: dict[str, list[dict]] = {}

        if redis_client is not None:
            logger.warning(
                "KnowledgeStore: redis_client is deprecated and ignored. "
                "Knowledge layer uses Neo4j only for structured relationships."
            )

        if neo4j_url and neo4j_auth and _HAS_NEO4J:
            try:
                self._neo4j_driver = GraphDatabase.driver(neo4j_url, auth=neo4j_auth)
                self._neo4j_driver.verify_connectivity()
                self._neo4j_available = True
                logger.info("KnowledgeStore: Neo4j connected successfully.")
            except Exception as e:
                logger.warning(
                    f"KnowledgeStore: Neo4j connection failed, in-memory fallback active: {e}"
                )
                self._neo4j_driver = None
        elif neo4j_url and not _HAS_NEO4J:
            logger.warning(
                "KnowledgeStore: NEO4J_URL set but 'neo4j' package not installed. pip install neo4j"
            )

        if not self._neo4j_available:
            logger.info("KnowledgeStore: Using in-memory storage (development mode).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_entity(self, tenant_id: str, entity: dict) -> None:
        """
        Stores a structured entity relationship in the knowledge graph.

        Args:
            tenant_id: Tenant ID in "project_id:user_id" format.
            entity: Entity dict with fields:
                - subject: The entity name (e.g. "mushrooms", "bold text")
                - category: Subject type (ingredient, formatting, tone, etc.)
                - relationship: PREFERRED_BY | REJECTED_BY | DISLIKED_BY | EXPECTED_BY
                - context: Optional str or dict with extra context
                - confidence: Optional float
                Legacy fields also accepted:
                - type: "dislike" | "like" | "personality" | "fact"
                - content: maps to subject if subject not provided
                - dimension: maps to category if category not provided
        """
        subject = entity.get("subject") or entity.get("content", "")
        if not subject:
            return

        category = entity.get("category") or entity.get("dimension", "general")
        relationship = entity.get("relationship") or _TYPE_TO_REL.get(
            entity.get("type", ""), "RELATED_TO"
        )
        context = entity.get("context", "")
        if isinstance(context, dict):
            context = json.dumps(context, ensure_ascii=False)

        enriched = {
            "subject": subject,
            "category": category,
            "relationship": relationship,
            "context": context,
            "confidence": entity.get("confidence", 0.5),
            "stored_at": time.time(),
            "tenant_id": tenant_id,
        }

        if self._neo4j_available:
            self._store_neo4j(tenant_id, enriched)
        else:
            self._store_memory(tenant_id, enriched)

        logger.debug(
            f"KnowledgeStore: stored ({tenant_id}) "
            f"({subject!r}:{category})-[{relationship}]->(user)"
        )

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
            query: Optional keyword to filter by (searches subject field).
            entity_type: Optional type filter ("dislike", "like", "personality", etc.)
            limit: Maximum number of entities to return.

        Returns:
            List of entity dicts with subject, category, relationship, context fields.
        """
        if self._neo4j_available:
            return self._query_neo4j(tenant_id, query, entity_type, limit)
        return self._query_memory(tenant_id, query, entity_type, limit)

    def link_entities(
        self,
        tenant_id: str,
        from_entity: str,
        to_entity: str,
        relation: str,
    ) -> None:
        """
        Creates a relationship between two Subject nodes.

        Args:
            tenant_id: Tenant ID.
            from_entity: Source subject name.
            to_entity: Target subject name.
            relation: Relationship type (e.g., "RELATED_TO", "APPEARS_IN").
        """
        if not self._neo4j_available:
            logger.debug("KnowledgeStore: link_entities skipped — Neo4j not available")
            return
        self._link_neo4j(tenant_id, from_entity, to_entity, relation)
        logger.debug(
            f"KnowledgeStore: linked ({tenant_id}): {from_entity} -{relation}-> {to_entity}"
        )

    def build_knowledge_injection(self, tenant_id: str) -> str:
        """
        Builds a structured knowledge context string from stored entity relationships.
        Used by the MemoryOrchestrator to enrich system prompts.

        Returns:
            Formatted string of known entity relationships, or empty string if none.
        """
        rejected = self.query_entities(tenant_id, entity_type="dislike", limit=15)
        preferred = self.query_entities(tenant_id, entity_type="like", limit=15)
        expected = self.query_entities(tenant_id, entity_type="personality", limit=10)

        if not rejected and not preferred and not expected:
            return ""

        lines = ["[Long-term Knowledge — Entity Memory]"]

        if rejected:
            lines.append("\nKnown dislikes (avoid these):")
            for e in rejected:
                subject = e.get("subject", "")
                category = e.get("category", "")
                ctx = e.get("context", "")
                entry = f"  ✗ {subject}"
                if category and category not in ("general", ""):
                    entry += f" ({category})"
                if ctx:
                    entry += f" — {ctx}"
                lines.append(entry)

        if preferred:
            lines.append("\nKnown likes:")
            for e in preferred:
                subject = e.get("subject", "")
                category = e.get("category", "")
                entry = f"  ✓ {subject}"
                if category and category not in ("general", ""):
                    entry += f" ({category})"
                lines.append(entry)

        if expected:
            lines.append("\nPersonality/behavior expectations:")
            for e in expected:
                lines.append(f"  → {e.get('subject', '')}")

        return "\n".join(lines)

    def delete_all(self, tenant_id: str) -> None:
        """Deletes all entity relationships for a tenant."""
        if self._neo4j_available:
            self._delete_neo4j(tenant_id)
        else:
            keys_to_del = [k for k in self._memory if k.startswith(f"{tenant_id}:")]
            for k in keys_to_del:
                del self._memory[k]

    # ------------------------------------------------------------------
    # In-memory fallback (development only)
    # ------------------------------------------------------------------

    def _store_memory(self, tenant_id: str, entity: dict) -> None:
        key = f"{tenant_id}:{entity.get('relationship', 'RELATED_TO')}"
        lst = self._memory.setdefault(key, [])
        existing = {e.get("subject", "") for e in lst}
        if entity.get("subject", "") not in existing:
            lst.append(entity)

    def _query_memory(
        self,
        tenant_id: str,
        query: str = "",
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        target_rels: set[str] | None = None
        if entity_type:
            rel = _TYPE_TO_REL.get(entity_type)
            if rel:
                target_rels = {rel}
                if entity_type == "dislike":
                    target_rels.add("DISLIKED_BY")

        results = []
        for key, lst in self._memory.items():
            if not key.startswith(f"{tenant_id}:"):
                continue
            if target_rels:
                rel_in_key = key.split(":", 1)[1] if ":" in key else ""
                if rel_in_key not in target_rels:
                    continue
            for entity in lst:
                subject = entity.get("subject", "")
                if not query or query.lower() in subject.lower():
                    results.append(entity)
        return results[:limit]

    # ------------------------------------------------------------------
    # Neo4j backend
    # ------------------------------------------------------------------

    def _store_neo4j(self, tenant_id: str, entity: dict) -> None:
        subject = entity["subject"]
        category = entity["category"]
        relationship = entity["relationship"]
        try:
            with self._neo4j_driver.session() as session:
                session.run(
                    f"""
                    MERGE (u:User {{tenant_id: $tenant_id}})
                    MERGE (s:Subject {{name: $subject, tenant_id: $tenant_id}})
                    SET s.category = $category
                    MERGE (s)-[r:{relationship}]->(u)
                    SET r.context = $context,
                        r.confidence = $confidence,
                        r.stored_at = $stored_at
                    """,
                    tenant_id=tenant_id,
                    subject=subject,
                    category=category,
                    context=entity.get("context", ""),
                    confidence=entity.get("confidence", 0.5),
                    stored_at=entity.get("stored_at", time.time()),
                )
        except Exception as e:
            logger.error(f"KnowledgeStore: Neo4j store error: {e}")

    def _query_neo4j(
        self,
        tenant_id: str,
        query: str = "",
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        try:
            with self._neo4j_driver.session() as session:
                if entity_type and entity_type in _TYPE_TO_REL:
                    rels = [_TYPE_TO_REL[entity_type]]
                    if entity_type == "dislike":
                        rels.append("DISLIKED_BY")
                    rel_filter = "|".join(rels)
                    result = session.run(
                        f"""
                        MATCH (s:Subject {{tenant_id: $tenant_id}})-[r:{rel_filter}]->(u:User {{tenant_id: $tenant_id}})
                        WHERE $search_text = '' OR toLower(s.name) CONTAINS toLower($search_text)
                        RETURN s.name AS subject, s.category AS category,
                               type(r) AS relationship, r.context AS context,
                               r.confidence AS confidence, r.stored_at AS stored_at
                        ORDER BY r.stored_at DESC LIMIT $limit
                        """,
                        tenant_id=tenant_id,
                        search_text=query,
                        limit=limit,
                    )
                else:
                    result = session.run(
                        """
                        MATCH (s:Subject {tenant_id: $tenant_id})-[r]->(u:User {tenant_id: $tenant_id})
                        WHERE $search_text = '' OR toLower(s.name) CONTAINS toLower($search_text)
                        RETURN s.name AS subject, s.category AS category,
                               type(r) AS relationship, r.context AS context,
                               r.confidence AS confidence, r.stored_at AS stored_at
                        ORDER BY r.stored_at DESC LIMIT $limit
                        """,
                        tenant_id=tenant_id,
                        search_text=query,
                        limit=limit,
                    )
                return [
                    {
                        "subject": record["subject"],
                        "category": record["category"] or "general",
                        "relationship": record["relationship"],
                        "context": record["context"] or "",
                        "confidence": record["confidence"] or 0.5,
                        "stored_at": record["stored_at"],
                        "content": record["subject"],  # legacy compat
                        "type": _REL_TO_TYPE.get(record["relationship"], "unknown"),
                    }
                    for record in result
                ]
        except Exception as e:
            logger.error(f"KnowledgeStore: Neo4j query error: {e}")
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
                    MERGE (a:Subject {{name: $from_entity, tenant_id: $tenant_id}})
                    MERGE (b:Subject {{name: $to_entity, tenant_id: $tenant_id}})
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
                    MATCH (s:Subject {tenant_id: $tenant_id})-[r]->(u:User {tenant_id: $tenant_id})
                    DELETE r
                    WITH s WHERE NOT (s)--()
                    DELETE s
                    """,
                    tenant_id=tenant_id,
                )
        except Exception as e:
            logger.error(f"KnowledgeStore: Neo4j delete error: {e}")
