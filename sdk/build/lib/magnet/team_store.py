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
import random
import string
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


# ── MagnetTeamStore — new team memory (MemoryStore category format) ────────────
#
# Data model (all in Redis):
#   team:{team_id}:meta      → {name, owner, created_at, plan}
#   team:{team_id}:members   → {user_id: {role, joined_at}}
#   team:{team_id}:projects  → [project_name, ...]
#   team:member:{user_id}    → [team_id, ...]
#   vmm:team:{team_id}:{project} → {items: [{id, category, text, status, shared_by, ...}]}
#
# This mirrors the solo vmm:{user}:{profile}:{project} format exactly,
# so personal and team items can be merged and displayed uniformly.

_TEAM_ID_CHARS = string.ascii_lowercase + string.digits
_TEAM_META_TTL = 60 * 60 * 24 * 365   # 1 year
_TEAM_ITEM_TTL = 60 * 60 * 24 * 90    # 90 days (matches MemoryStore)
_AUTO_PROMOTE_THRESHOLD = 0.60          # difflib ratio for "same idea"

TEAM_NEEDS_REDIS_MSG = (
    "Team memory needs shared storage. "
    "You and your teammates must all set the same MAGNET_REDIS_URL. "
    "This is a Pro feature — see agentmagnet.app."
)


def _team_gen_id() -> str:
    return "team-" + "".join(random.choices(_TEAM_ID_CHARS, k=6))


def _tn(s: str) -> str:
    return s.strip().lower()


class MagnetTeamStore:
    """
    Team memory store — new category-based format.
    Requires Redis (raises TeamMemoryRequiresRedis on local-only mode).
    Stores shared project items in vmm:team:{team_id}:{project}.
    """

    def __init__(self, redis_client: Any | None) -> None:
        self._redis = redis_client

    def _require_redis(self) -> None:
        if not self._redis or _is_local_mode(self._redis):
            raise TeamMemoryRequiresRedis(TEAM_NEEDS_REDIS_MSG)

    # ── Key helpers ────────────────────────────────────────────────────

    @staticmethod
    def _meta_key(team_id: str) -> str:
        return f"team:{team_id}:meta"

    @staticmethod
    def _members_key(team_id: str) -> str:
        return f"team:{team_id}:members"

    @staticmethod
    def _projects_key(team_id: str) -> str:
        return f"team:{team_id}:projects"

    @staticmethod
    def _user_teams_key(user_id: str) -> str:
        return f"team:member:{_tn(user_id)}"

    @staticmethod
    def _project_key(team_id: str, project: str) -> str:
        return f"vmm:team:{team_id}:{_tn(project)}"

    # ── Team lifecycle ─────────────────────────────────────────────────

    def create_team(self, name: str, owner: str) -> str:
        """Create a team, set owner as first member. Returns new team_id."""
        self._require_redis()
        team_id = _team_gen_id()
        meta = {"name": name, "owner": owner, "created_at": time.time(), "plan": "pro_beta"}
        self._redis.setex(self._meta_key(team_id), _TEAM_META_TTL, json.dumps(meta, ensure_ascii=False))
        members: dict = {owner: {"role": "owner", "joined_at": time.time()}}
        self._redis.setex(self._members_key(team_id), _TEAM_META_TTL, json.dumps(members, ensure_ascii=False))
        self._redis.setex(self._projects_key(team_id), _TEAM_META_TTL, json.dumps([]))
        self._register_user_team(owner, team_id)
        logger.info(f"[team] created '{name}' ({team_id}) — owner: {owner}")
        return team_id

    def get_team_meta(self, team_id: str) -> dict | None:
        self._require_redis()
        raw = self._redis.get(self._meta_key(team_id))
        return json.loads(raw) if raw else None

    def join_team(self, team_id: str, user_id: str) -> bool:
        """User joins a team by id. Returns False if team doesn't exist."""
        self._require_redis()
        if not self.get_team_meta(team_id):
            return False
        members = self._get_members_dict(team_id)
        if user_id not in members:
            members[user_id] = {"role": "member", "joined_at": time.time()}
            self._redis.setex(self._members_key(team_id), _TEAM_META_TTL, json.dumps(members, ensure_ascii=False))
            self._register_user_team(user_id, team_id)
        return True

    def add_member(self, team_id: str, requester_id: str, user_id_to_add: str) -> tuple[bool, str]:
        """Owner adds a member. Returns (success, message)."""
        self._require_redis()
        members = self._get_members_dict(team_id)
        if members.get(requester_id, {}).get("role") != "owner":
            return False, "Only the team owner can add members directly. Share your team_id instead."
        if user_id_to_add in members:
            return True, f"'{user_id_to_add}' is already a member."
        members[user_id_to_add] = {"role": "member", "joined_at": time.time()}
        self._redis.setex(self._members_key(team_id), _TEAM_META_TTL, json.dumps(members, ensure_ascii=False))
        self._register_user_team(user_id_to_add, team_id)
        return True, f"'{user_id_to_add}' added to the team."

    def list_members(self, team_id: str) -> list[dict]:
        """Return list of {user_id, role, joined_at}."""
        self._require_redis()
        members = self._get_members_dict(team_id)
        return sorted(
            [{"user_id": uid, **info} for uid, info in members.items()],
            key=lambda m: m.get("joined_at", 0),
        )

    def get_teams_for_user(self, user_id: str) -> list[str]:
        try:
            self._require_redis()
            return self._get_user_teams(user_id)
        except TeamMemoryRequiresRedis:
            return []

    def is_project_shared(self, team_id: str, project: str) -> bool:
        """True if this project has been shared into the team's shared space."""
        try:
            self._require_redis()
            raw = self._redis.get(self._projects_key(team_id))
            projects: list = json.loads(raw) if raw else []
            return _tn(project) in [_tn(p) for p in projects]
        except Exception:
            return False

    # ── Shared project memory ──────────────────────────────────────────

    def load_team_items(self, team_id: str, project: str) -> list[dict]:
        self._require_redis()
        raw = self._redis.get(self._project_key(team_id, project))
        if not raw:
            return []
        data = json.loads(raw)
        return data.get("items", []) if isinstance(data, dict) else []

    def save_team_items(self, team_id: str, project: str, items: list[dict]) -> None:
        self._require_redis()
        payload = json.dumps({"items": items[-200:]}, ensure_ascii=False)
        self._redis.setex(self._project_key(team_id, project), _TEAM_ITEM_TTL, payload)

    def share_project(
        self,
        user: str,
        project: str,
        team_id: str,
        personal_items: list[dict],
    ) -> dict:
        """Copy all personal project items into team memory."""
        self._require_redis()
        existing = self.load_team_items(team_id, project)
        existing_sigs = {(it.get("category"), _tn(it.get("text", ""))) for it in existing}

        added = 0
        for item in personal_items:
            sig = (item.get("category"), _tn(item.get("text", "")))
            if sig not in existing_sigs:
                existing.append({**item, "shared_by": user, "shared_at": time.time()})
                existing_sigs.add(sig)
                added += 1

        self.save_team_items(team_id, project, existing)
        self._register_shared_project(team_id, project)
        logger.info(f"[team] {user} shared {added} items → team/{team_id}/{project}")
        return {"shared": added, "team_id": team_id, "project": project}

    def share_item(
        self,
        team_id: str,
        project: str,
        item_id: str,
        user_id: str,
        personal_items: list[dict],
    ) -> dict:
        """Share one specific item by id to team memory."""
        self._require_redis()
        item = next((i for i in personal_items if i.get("id") == item_id), None)
        if not item:
            return {"error": f"No item with id '{item_id}' found in personal memory."}

        existing = self.load_team_items(team_id, project)
        for ex in existing:
            if ex.get("id") == item_id or _tn(ex.get("text", "")) == _tn(item.get("text", "")):
                return {"already_shared": True, "text": item["text"][:80]}

        existing.append({**item, "shared_by": user_id, "shared_at": time.time()})
        self.save_team_items(team_id, project, existing)
        self._register_shared_project(team_id, project)
        logger.info(f"[team] {user_id} shared item [{item_id}] → team/{team_id}/{project}")
        return {"shared": 1, "item": item["text"][:80], "category": item.get("category")}

    def auto_promote_if_agreed(
        self,
        team_id: str,
        project: str,
        new_item: dict,
        current_user: str,
    ) -> bool:
        """
        If this new item semantically agrees with an item already in team memory
        from a DIFFERENT user, auto-promote it (add to team memory too).
        Returns True if promoted.
        """
        try:
            self._require_redis()
        except TeamMemoryRequiresRedis:
            return False

        team_items = self.load_team_items(team_id, project)
        new_text = _tn(new_item.get("text", ""))
        new_cat  = new_item.get("category")
        if not new_text or not new_cat:
            return False

        # Find a matching item in team memory from a different user
        agreed_with = None
        for ti in team_items:
            if ti.get("category") != new_cat:
                continue
            if ti.get("shared_by") in (current_user, "auto_promotion"):
                continue
            ratio = difflib.SequenceMatcher(None, new_text, _tn(ti.get("text", ""))).ratio()
            if ratio >= _AUTO_PROMOTE_THRESHOLD:
                agreed_with = ti
                break

        if not agreed_with:
            return False

        # Don't add if current user already has a similar item in team memory
        for ti in team_items:
            if ti.get("shared_by") != current_user or ti.get("category") != new_cat:
                continue
            if difflib.SequenceMatcher(None, new_text, _tn(ti.get("text", ""))).ratio() >= _AUTO_PROMOTE_THRESHOLD:
                return False

        team_item = {
            **new_item,
            "shared_by": current_user,
            "shared_at": time.time(),
            "source": "auto_promotion",
            "agreed_with": agreed_with.get("shared_by", "unknown"),
        }
        team_items.append(team_item)
        self.save_team_items(team_id, project, team_items)
        logger.info(
            f"[team] auto-promoted [{new_cat}] to team/{team_id}/{project}: {new_text[:60]!r} "
            f"(agreed with {agreed_with.get('shared_by')})"
        )
        return True

    # ── Display ────────────────────────────────────────────────────────

    def format_team_display(self, team_id: str, project: str) -> str:
        """Human-readable display of team shared memory for a project."""
        try:
            items = self.load_team_items(team_id, project)
        except TeamMemoryRequiresRedis:
            return TEAM_NEEDS_REDIS_MSG

        if not items:
            return f"No shared memory yet for team/{team_id} — {project}."

        from magnet.project_store import CATEGORIES, _LABELS, _DISPLAY_ORDER
        by_cat: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
        for e in items:
            c = e.get("category", "preference")
            if c in by_cat:
                by_cat[c].append(e)

        lines = [f"Team memory — {project} (team: {team_id}):"]
        for cat in _DISPLAY_ORDER:
            xs = by_cat.get(cat, [])
            if xs:
                lines.append(f"\n  {_LABELS[cat]}:")
                for item in xs:
                    item_id = item.get("id", "??????")
                    text = item["text"]
                    who = item.get("shared_by", "?")
                    src = " [auto]" if item.get("source") == "auto_promotion" else ""
                    lines.append(f"    [{item_id}] {text}  (by {who}{src})")
        return "\n".join(lines)

    # ── Internal helpers ───────────────────────────────────────────────

    def _get_members_dict(self, team_id: str) -> dict:
        raw = self._redis.get(self._members_key(team_id))
        return json.loads(raw) if raw else {}

    def _get_user_teams(self, user_id: str) -> list[str]:
        raw = self._redis.get(self._user_teams_key(user_id))
        return json.loads(raw) if raw else []

    def _register_user_team(self, user_id: str, team_id: str) -> None:
        teams = self._get_user_teams(user_id)
        if team_id not in teams:
            teams.append(team_id)
            self._redis.setex(self._user_teams_key(user_id), _TEAM_META_TTL, json.dumps(teams))

    def _register_shared_project(self, team_id: str, project: str) -> None:
        raw = self._redis.get(self._projects_key(team_id))
        projects: list = json.loads(raw) if raw else []
        if _tn(project) not in [_tn(p) for p in projects]:
            projects.append(_tn(project))
            self._redis.setex(self._projects_key(team_id), _TEAM_META_TTL, json.dumps(projects))
