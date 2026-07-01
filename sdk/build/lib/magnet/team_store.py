"""
TeamStore
---------
Shared memory for team and org scopes.

REQUIRES Redis — local SQLite mode cannot support cross-machine sharing.
When called without a real Redis backend, raises TeamMemoryRequiresRedis
with an actionable message.

Key structure (separate from personal profile keys):
  vmm:team_profile:{project_id}:{team_id}
  vmm:org_profile:{project_id}:{org_id}

Personal profile keys (managed by ProfileStore) are NOT touched here:
  vmm:profile:{project_id}:{user_id}   (existing format, unchanged)
"""

from __future__ import annotations

import difflib
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_TEAM_PREFIX = "vmm:team_profile:"
_ORG_PREFIX  = "vmm:org_profile:"
_USER_PREFIX = "vmm:profile:"
_TTL = 60 * 60 * 24 * 30   # 30 days

REDIS_REQUIRED_MSG = (
    "Team memory requires shared storage. "
    "Set MAGNET_REDIS_URL for all team members to use the same Redis instance."
)

_SIM_THRESHOLD = 0.75   # difflib ratio for subject clustering
_MIN_USERS     = 2       # minimum users for auto-promotion


class TeamMemoryRequiresRedis(RuntimeError):
    """Raised when a team feature is called without a real Redis backend."""


def _is_local_mode(client: Any) -> bool:
    """True if client is SQLiteBackend (non-shared, machine-local storage)."""
    try:
        from .local_store import SQLiteBackend
        return isinstance(client, SQLiteBackend)
    except ImportError:
        return False


class TeamStore:
    """
    Stores and retrieves team/org-scoped memory profiles.

    All write/read operations require a real Redis client.
    If the client is absent or is a local SQLiteBackend, raises
    TeamMemoryRequiresRedis with a human-readable message.
    """

    def __init__(self, redis_client: Any | None) -> None:
        self._redis = redis_client

    def _require_redis(self) -> None:
        if not self._redis or _is_local_mode(self._redis):
            logger.warning("TeamStore: team feature attempted without Redis. " + REDIS_REQUIRED_MSG)
            raise TeamMemoryRequiresRedis(REDIS_REQUIRED_MSG)

    # ── Key helpers ────────────────────────────────────────────────────

    @staticmethod
    def _team_key(project_id: str, team_id: str) -> str:
        return f"{_TEAM_PREFIX}{project_id}:{team_id}"

    @staticmethod
    def _org_key(project_id: str, org_id: str) -> str:
        return f"{_ORG_PREFIX}{project_id}:{org_id}"

    @staticmethod
    def _user_scan_pattern(project_id: str) -> str:
        return f"{_USER_PREFIX}{project_id}:*"

    # ── Team/org profile CRUD ──────────────────────────────────────────

    def save_team_profile(self, team_id: str, project_id: str, profile: dict) -> None:
        self._require_redis()
        key = self._team_key(project_id, team_id)
        self._redis.setex(key, _TTL, json.dumps(profile, ensure_ascii=False))
        logger.info(f"[team] profile saved: {project_id}/{team_id}")

    def load_team_profile(self, team_id: str, project_id: str) -> dict | None:
        self._require_redis()
        raw = self._redis.get(self._team_key(project_id, team_id))
        return json.loads(raw) if raw else None

    def save_org_profile(self, org_id: str, project_id: str, profile: dict) -> None:
        self._require_redis()
        self._redis.setex(self._org_key(project_id, org_id), _TTL,
                          json.dumps(profile, ensure_ascii=False))

    def load_org_profile(self, org_id: str, project_id: str) -> dict | None:
        self._require_redis()
        raw = self._redis.get(self._org_key(project_id, org_id))
        return json.loads(raw) if raw else None

    # ── Explicit sharing ───────────────────────────────────────────────

    def share_to_team(
        self,
        user_id: str,
        project_id: str,
        fact_or_subject: str,
        team_id: str,
        personal_profile: dict | None,
    ) -> dict:
        """
        Copy one specific preference from the user's personal profile into team memory.
        Explicit and user-initiated — never called automatically.
        """
        self._require_redis()

        if not personal_profile:
            return {"status": "error", "reason": "no personal profile found for this user"}

        needle = fact_or_subject.lower()
        matched = [
            p for p in personal_profile.get("preferences", [])
            if isinstance(p, dict)
            and needle in (p.get("subject", "") + " " + p.get("natural_text", "")).lower()
        ]

        if not matched:
            return {
                "status": "error",
                "reason": f"no preference matching '{fact_or_subject}' in personal profile",
            }

        team_profile = self.load_team_profile(team_id, project_id) or {"preferences": []}
        team_prefs: list[dict] = team_profile.get("preferences", [])
        existing = {
            (p.get("subject", "").lower(), p.get("relation", ""))
            for p in team_prefs if isinstance(p, dict)
        }

        added = 0
        for pref in matched:
            sig = (pref.get("subject", "").lower(), pref.get("relation", ""))
            if sig not in existing:
                team_prefs.append({**pref, "shared_by": user_id, "shared_at": time.time()})
                existing.add(sig)
                added += 1

        team_profile["preferences"] = team_prefs
        team_profile["updated_at"] = time.time()
        self.save_team_profile(team_id, project_id, team_profile)

        logger.info(
            f"[team] {user_id} shared {added} pref(s) matching '{fact_or_subject}' "
            f"→ team={team_id} project={project_id}"
        )
        return {"status": "ok", "shared": added, "team_id": team_id, "project_id": project_id}

    def forget_team(self, team_id: str, project_id: str, fact_or_subject: str) -> dict:
        """Remove a specific preference from team memory by subject match."""
        self._require_redis()

        team_profile = self.load_team_profile(team_id, project_id)
        if not team_profile:
            return {"status": "ok", "removed": 0}

        needle = fact_or_subject.lower()
        before = len(team_profile.get("preferences", []))
        team_profile["preferences"] = [
            p for p in team_profile.get("preferences", [])
            if not (
                isinstance(p, dict)
                and needle in (p.get("subject", "") + " " + p.get("natural_text", "")).lower()
            )
        ]
        removed = before - len(team_profile["preferences"])
        team_profile["updated_at"] = time.time()
        self.save_team_profile(team_id, project_id, team_profile)
        return {"status": "ok", "removed": removed}

    # ── Auto-promotion ─────────────────────────────────────────────────

    def promote_to_team(
        self,
        team_id: str,
        project_id: str,
        min_users: int = _MIN_USERS,
    ) -> dict:
        """
        Scan all personal profiles in project_id.
        Find preferences shared by min_users+ distinct users (difflib clustering).
        Write common preferences to the team profile.
        Uses the same clustering algorithm as ConsolidationEngine._find_patterns().
        """
        self._require_redis()

        pattern = self._user_scan_pattern(project_id)
        all_prefs: list[tuple[str, dict]] = []  # (user_id, pref)
        prefix_len = len(f"{_USER_PREFIX}{project_id}:")

        try:
            for raw_key in self._redis.scan_iter(pattern):
                key = raw_key if isinstance(raw_key, str) else raw_key.decode()
                user_id = key[prefix_len:]
                raw = self._redis.get(key)
                if not raw:
                    continue
                try:
                    profile = json.loads(raw)
                except Exception:
                    continue
                for pref in profile.get("preferences", []):
                    if isinstance(pref, dict):
                        all_prefs.append((user_id, pref))
        except Exception as e:
            logger.error(f"[team] promote_to_team scan error: {e}")
            return {"status": "error", "reason": str(e)}

        # Cluster by (relation, ~subject) — same logic as ConsolidationEngine._find_patterns
        used = [False] * len(all_prefs)
        candidates: list[dict] = []

        for i, (uid_i, pref_i) in enumerate(all_prefs):
            if used[i]:
                continue
            relation = pref_i.get("relation", "")
            subj_i = pref_i.get("subject", "").lower()
            if not subj_i or not relation:
                continue

            cluster_users: set[str] = {uid_i}
            cluster_confs = [pref_i.get("confidence", 0.5)]

            for j, (uid_j, pref_j) in enumerate(all_prefs):
                if i == j or used[j] or uid_j in cluster_users:
                    continue
                if pref_j.get("relation") != relation:
                    continue
                subj_j = pref_j.get("subject", "").lower()
                if subj_j and difflib.SequenceMatcher(None, subj_i, subj_j).ratio() >= _SIM_THRESHOLD:
                    cluster_users.add(uid_j)
                    cluster_confs.append(pref_j.get("confidence", 0.5))
                    used[j] = True
            used[i] = True

            if len(cluster_users) >= min_users:
                avg_conf = sum(cluster_confs) / len(cluster_confs)
                candidates.append({
                    **pref_i,
                    "confidence": round(avg_conf, 3),
                    "source": "auto_promotion",
                    "promoted_from_users": list(cluster_users),
                    "promoted_at": time.time(),
                })

        if not candidates:
            return {"status": "ok", "promoted": 0, "candidates_scanned": len(all_prefs)}

        team_profile = self.load_team_profile(team_id, project_id) or {"preferences": []}
        team_prefs = team_profile.get("preferences", [])
        existing = {
            (p.get("subject", "").lower(), p.get("relation", ""))
            for p in team_prefs if isinstance(p, dict)
        }

        added = 0
        for pref in candidates:
            sig = (pref.get("subject", "").lower(), pref.get("relation", ""))
            if sig not in existing:
                team_prefs.append(pref)
                existing.add(sig)
                added += 1

        team_profile["preferences"] = team_prefs
        team_profile["updated_at"] = time.time()
        self.save_team_profile(team_id, project_id, team_profile)

        logger.info(f"[team] promote_to_team: {added} prefs promoted to {team_id}/{project_id}")
        return {"status": "ok", "promoted": added, "candidates_scanned": len(all_prefs)}

    # ── Project memory view ────────────────────────────────────────────

    def get_project_memory(self, project_id: str, team_id: str | None = None) -> dict:
        """
        Return a per-user breakdown of preferences learned within project_id.
        Optionally includes the team's shared profile.

        Returns:
          {
            "project_id": str,
            "contributors": {
              "ahmet": {"prefers": [...], "dislikes": [...], "expects": [...], "watch_out": [...]},
              ...
            },
            "team_shared": {"prefers": [...], "watch_out": [...]}   # only if team_id given
          }
        """
        self._require_redis()

        pattern = self._user_scan_pattern(project_id)
        prefix_len = len(f"{_USER_PREFIX}{project_id}:")
        contributors: dict[str, dict] = {}

        try:
            for raw_key in self._redis.scan_iter(pattern):
                key = raw_key if isinstance(raw_key, str) else raw_key.decode()
                user_id = key[prefix_len:]
                raw = self._redis.get(key)
                if not raw:
                    continue
                try:
                    profile = json.loads(raw)
                except Exception:
                    continue
                buckets: dict[str, list] = {
                    "prefers": [], "dislikes": [], "expects": [], "watch_out": []
                }
                for pref in profile.get("preferences", []):
                    if not isinstance(pref, dict):
                        continue
                    relation = pref.get("relation", "")
                    if relation not in buckets:
                        continue
                    buckets[relation].append({
                        "text": pref.get("natural_text", pref.get("subject", "")),
                        "confidence": round(pref.get("confidence", 0.5), 3),
                    })
                if any(buckets.values()):
                    contributors[user_id] = buckets
        except Exception as e:
            logger.error(f"[team] get_project_memory scan error: {e}")

        result: dict = {"project_id": project_id, "contributors": contributors}

        if team_id:
            team_profile = self.load_team_profile(team_id, project_id)
            team_shared: dict[str, list] = {"prefers": [], "watch_out": []}
            for pref in (team_profile or {}).get("preferences", []):
                if not isinstance(pref, dict):
                    continue
                relation = pref.get("relation", "")
                entry = {
                    "text": pref.get("natural_text", pref.get("subject", "")),
                    "confidence": round(pref.get("confidence", 0.5), 3),
                }
                if relation in team_shared:
                    team_shared[relation].append(entry)
            result["team_shared"] = team_shared

        return result

    # ── Merged injection ───────────────────────────────────────────────

    def build_merged_injection(
        self,
        user_profile: dict | None,
        team_id: str | None,
        org_id: str | None,
        project_id: str,
        reflector: Any,
    ) -> str:
        """
        Merge user > team > org preferences and build a single injection string.
        Most specific scope wins: user overrides team, team overrides org.
        Does NOT require Redis check — caller already has user_profile from ProfileStore.
        Silently skips team/org layers if Redis unavailable.
        """
        merged: list[dict] = []
        seen: set[tuple] = set()

        def _add(prefs: list[dict], source: str) -> None:
            for p in prefs:
                if not isinstance(p, dict):
                    continue
                sig = (p.get("subject", "").lower(), p.get("relation", ""))
                if sig not in seen:
                    merged.append({**p, "_scope": source})
                    seen.add(sig)

        # Add lowest-precedence first, then override with higher
        if org_id:
            try:
                org_p = self.load_org_profile(org_id, project_id)
                if org_p:
                    _add(org_p.get("preferences", []), "org")
            except TeamMemoryRequiresRedis:
                pass

        if team_id:
            try:
                team_p = self.load_team_profile(team_id, project_id)
                if team_p:
                    _add(team_p.get("preferences", []), "team")
            except TeamMemoryRequiresRedis:
                pass

        # User is highest precedence — added last so seen-set blocks lower-scope duplicates
        if user_profile:
            _add(user_profile.get("preferences", []), "user")

        if not merged and not user_profile:
            return ""

        merged_profile = {**(user_profile or {}), "preferences": merged}
        return reflector.build_injection(merged_profile)
