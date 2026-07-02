"""
agent-magnet init
-----------------
One-command setup: detects installed AI tools, asks minimal questions,
and writes the correct MCP server config into each tool's config file.
"""

from __future__ import annotations

import getpass
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


# ── Config path resolution ─────────────────────────────────────────────────────

def _claude_code_path() -> Path:
    return Path.home() / ".claude" / "settings.json"


def _claude_desktop_path() -> Path:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        return Path(appdata) / "Claude" / "claude_desktop_config.json"
    return Path.home() / ".config" / "claude" / "claude_desktop_config.json"


def _cursor_path() -> Path:
    return Path.home() / ".cursor" / "mcp.json"


def _cursor_path_alt() -> Path:
    """Fallback Cursor MCP config (VS Code-style globalStorage path)."""
    system = platform.system()
    if system == "Darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Cursor"
            / "User"
            / "globalStorage"
            / "cursor.mcp"
            / "settings.json"
        )
    if system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        return Path(appdata) / "Cursor" / "User" / "globalStorage" / "cursor.mcp" / "settings.json"
    return Path.home() / ".config" / "Cursor" / "User" / "globalStorage" / "cursor.mcp" / "settings.json"


TOOLS = {
    "Claude Code": _claude_code_path,
    "Claude Desktop": _claude_desktop_path,
    "Cursor": _cursor_path,
}


# ── CLAUDE.md memory block ────────────────────────────────────────────────────

_CLAUDE_MD_MARKER = "## Agent Magnet Memory"

def _claude_md_block(user_id: str) -> str:
    return "\n".join([
        "## Agent Magnet Memory",
        "",
        f"At conversation start, call `recall` (agent-magnet) to load {user_id}'s active project",
        "memory and adopt it silently before your first response.",
        "",
        "Agent Magnet keeps a running memory. Count user messages in the conversation.",
        "After roughly every 8 user messages, silently call `checkpoint` (pass the recent messages)",
        "to save what was decided so far, then reset your count and continue.",
        "Do not interrupt the user's flow to do this — fold it silently into your response.",
        "",
        "Whenever the user explicitly states a decision, preference, correction, or something to",
        "watch out for, call `remember` immediately (in the background — never announce it).",
        "",
        "If the user types *profiles, call `list_profiles` and present the menu.",
        "If the user types *projects, call `list_projects` and present the menu.",
        "If the user types *save, call `save_now` (pass ALL messages so far) to do a",
        "cumulative save of everything up to this point.",
        "If the user types *status, call `get_status` to show memory status and usage.",
        "If the user types *forget <something>, call `forget_memory`. If they reference an item",
        "by its id (shown in brackets in show_project_memory), delete it directly; otherwise",
        "find the best text match and present it for confirmation before deleting.",
        "If the user says a goal is done/completed/finished, call `mark_done`.",
        "If the user asks where they left off / what were we doing / types *recap,",
        "call `recap` and deliver it as a natural catch-up, not a list.",
        "If the user types *memory, call `show_all_memory` for the active project full dump.",
        "If they say *memory all or ask to see everything across all projects,",
        "call `show_all_memory` with show_all=true for a cross-project overview.",
        "",
        "Team memory (Pro, free during beta — requires shared Redis):",
        "*team new <name> → call create_team. Returns a team_id to share with teammates.",
        "*team join <id> → call join_team. Then add MAGNET_TEAM_ID to MCP config.",
        "*team members → call list_team_members to show who's in the team.",
        "*team share → call share_project_to_team to share the active project with the team.",
        "*share <item_id> → call share_item_to_team to share one specific item.",
        "When working in a shared project (MAGNET_TEAM_ID set and project is shared),",
        "recall, show_project_memory, and recap automatically merge team memory.",
        "Team items are labeled [team] so the user can see what's shared vs personal.",
        "Never expose Redis URLs, storage keys, or backend details — say 'team memory' or 'shared'.",
        "",
        "When the user asks what you know or remember, call `recall` or `show_project_memory`",
        "and answer ONLY from what those tools return — never guess or invent.",
        "",
        "Never expose storage keys, project_ids, or backend details — show only profile/project names.",
    ])


def _write_claude_md(user_id: str, project_id: str = "", profile_id: str = "") -> bool:
    """Write the Agent Magnet memory block to ~/.claude/CLAUDE.md. Returns True if written."""
    claude_md = Path.home() / ".claude" / "CLAUDE.md"
    block = _claude_md_block(user_id)

    if claude_md.exists():
        existing = claude_md.read_text(encoding="utf-8")
        if _CLAUDE_MD_MARKER in existing:
            # Replace the existing block in-place
            start = existing.index(_CLAUDE_MD_MARKER)
            # Find the next top-level heading (##) after the block, or end of file
            next_section = existing.find("\n## ", start + len(_CLAUDE_MD_MARKER))
            if next_section == -1:
                content = existing[:start].rstrip() + "\n\n" + block + "\n"
            else:
                content = existing[:start].rstrip() + "\n\n" + block + "\n\n" + existing[next_section:].lstrip()
            claude_md.write_text(content, encoding="utf-8")
            return True
        content = existing.rstrip() + "\n\n" + block + "\n"
    else:
        claude_md.parent.mkdir(parents=True, exist_ok=True)
        content = "# Memory\n\n" + block + "\n"

    claude_md.write_text(content, encoding="utf-8")
    return True


# ── JSON helpers ───────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ── Binary detection ───────────────────────────────────────────────────────────

def _mcp_command() -> str:
    """Prefer the installed binary; fall back to python -m."""
    found = shutil.which("agent-magnet-mcp")
    if found:
        return found
    return f"{sys.executable} -m magnet.mcp_server"


def _hook_command(env_pairs: list[tuple[str, str]]) -> str:
    """Build the inline shell command for the Claude Code Stop hook."""
    hook_bin = shutil.which("agent-magnet-hook") or f"{sys.executable} -m magnet.hooks.save_session"
    prefix = " ".join(f"{k}={v}" for k, v in env_pairs)
    return f"{prefix} {hook_bin}"


# ── MCP server entry builder ───────────────────────────────────────────────────

def _build_mcp_entry(
    user_id: str, openai_key: str = "", redis_url: str = "",
    team_id: str = "", api_key: str = "", profile_id: str = "",
    magnet_team_id: str = "",
    # legacy params kept for backward compat with callers that pass positional args
    storage: str = "local",
) -> dict:
    env: dict[str, str] = {"MAGNET_USER_ID": user_id}
    if openai_key:
        env["MAGNET_OPENAI_KEY"] = openai_key
    if redis_url:
        env["MAGNET_REDIS_URL"] = redis_url
    if team_id:
        env["MAGNET_TEAM_ID"] = team_id
    if magnet_team_id:
        env["MAGNET_TEAM_ID"] = magnet_team_id
    if api_key:
        env["MAGNET_API_KEY"] = api_key
    if profile_id and profile_id != "default":
        env["MAGNET_PROFILE"] = profile_id
    # No MAGNET_LOCAL_MODE needed — SQLite is now the automatic default
    return {"command": _mcp_command(), "env": env}


# ── Per-tool config writers ────────────────────────────────────────────────────

def _write_claude_code(
    path: Path, user_id: str, openai_key: str = "", redis_url: str = "",
    team_id: str = "", api_key: str = "", profile_id: str = "",
    # legacy positional args kept for compat
    storage: str = "local",
) -> str:
    config = _read_json(path)

    config.setdefault("mcpServers", {})
    config["mcpServers"]["agent-magnet"] = _build_mcp_entry(
        user_id, openai_key, redis_url, team_id, api_key, profile_id
    )

    env_pairs: list[tuple[str, str]] = [("MAGNET_USER_ID", user_id)]
    if openai_key:
        env_pairs.append(("MAGNET_OPENAI_KEY", openai_key))
    if redis_url:
        env_pairs.append(("MAGNET_REDIS_URL", redis_url))
    if team_id:
        env_pairs.append(("MAGNET_TEAM_ID", team_id))
    if api_key:
        env_pairs.append(("MAGNET_API_KEY", api_key))
    if profile_id and profile_id != "default":
        env_pairs.append(("MAGNET_PROFILE", profile_id))

    hook_entry: dict[str, Any] = {
        "type": "command",
        "command": _hook_command(env_pairs),
        "timeout": 10,
    }

    config.setdefault("hooks", {})
    existing_stop = config["hooks"].get("Stop", [])
    new_stop = [b for b in existing_stop if not _is_magnet_hook(b)]
    new_stop.append({"matcher": "", "hooks": [hook_entry]})
    config["hooks"]["Stop"] = new_stop

    _write_json(path, config)
    return "MCP server + Stop hook"


def _is_magnet_hook(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    for h in block.get("hooks", []):
        if isinstance(h, dict) and "magnet" in h.get("command", ""):
            return True
    return False


def _write_mcp_only(
    path: Path, user_id: str, openai_key: str = "", redis_url: str = "",
    team_id: str = "", api_key: str = "", profile_id: str = "",
    storage: str = "local",
) -> str:
    config = _read_json(path)
    config.setdefault("mcpServers", {})
    config["mcpServers"]["agent-magnet"] = _build_mcp_entry(
        user_id, openai_key, redis_url, team_id, api_key, profile_id
    )
    _write_json(path, config)
    return "MCP server"


# ── Input helpers ──────────────────────────────────────────────────────────────

def _ask(prompt: str, default: str = "") -> str:
    try:
        val = input(prompt).strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def _ask_secret(prompt: str) -> str:
    try:
        val = getpass.getpass(prompt)
        return val.strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def _ask_bool(prompt: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    raw = _ask(f"{prompt} {hint}: ", "")
    if not raw:
        return default
    return raw.lower() in ("y", "yes")


# ── Main init command ──────────────────────────────────────────────────────────

def cmd_init() -> None:
    print()
    print("  Agent Magnet Setup")
    print("  " + "─" * 38)
    print()

    # 1. Detect tools
    print("  Scanning for AI tools...")
    print()
    found: dict[str, Path] = {}
    for name, path_fn in TOOLS.items():
        path = path_fn()
        if path.exists() or path.parent.exists():
            found[name] = path
            status = "found" if path.exists() else "config dir found"
            print(f"    ✓ {name} ({status})")
        else:
            print(f"    – {name} (not detected)")

    if not found:
        print()
        print("  No AI tools detected. You can still configure manually.")
        print("  See: https://github.com/helinakdogan/magnet#setup")
        print()
        return

    # 2. Just two questions
    print()
    print("  Configuration")
    print("  " + "─" * 38)
    print()

    user_id = _ask("  Your name/identifier: ")
    if not user_id:
        user_id = getpass.getuser()
        print(f"  Using system username: {user_id}")

    print()
    openai_key = _ask_secret(
        "  OpenAI key for smarter extraction (press Enter to skip — memory works great without it): "
    )
    if not openai_key:
        print("  Skipped — on-device memory and semantic search will be used.")
    print()

    # Advanced options
    redis_url = ""
    api_key = ""
    profile_id = "default"

    print("  Team setup (optional)")
    print("  " + "─" * 38)
    print()
    print("  Team memory lets you share decisions with teammates.")
    print("  It requires all team members to use the same Redis instance.")
    print()
    team_id = _ask("  Team ID (press Enter to skip — needs shared Redis): ")
    if team_id:
        if not redis_url:
            print()
            redis_url = _ask("  Redis URL for team (redis://...): ")
        if redis_url:
            print(f"  Team: {team_id} · Redis: configured ✓")
        else:
            print("  Note: team memory won't work without MAGNET_REDIS_URL.")
    print()

    # 3. Bootstrap default profile + project in local storage
    try:
        from magnet.local_store import SQLiteBackend
        from magnet.project_store import MemoryStore
        _store = MemoryStore(SQLiteBackend())
        _store.create_profile(user_id, "personal")
        _store.create_project(user_id, "personal", "general")
    except Exception as exc:
        print(f"  Warning: could not init local storage: {exc}")

    # Write active.json
    active_path = Path.home() / ".agent-magnet" / "active.json"
    try:
        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text(
            json.dumps({"profile": "personal", "project": "general"}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"  Warning: could not write active.json: {exc}")

    # 4. Write configs
    print("  Writing configuration...")
    print()

    writers = {
        "Claude Code": _write_claude_code,
        "Claude Desktop": _write_mcp_only,
        "Cursor": _write_mcp_only,
    }

    for tool_name, path in found.items():
        writer = writers.get(tool_name, _write_mcp_only)
        try:
            action = writer(path, user_id, openai_key, redis_url, team_id, api_key, profile_id)
            print(f"    ✓ {tool_name} — {action}")
        except Exception as exc:
            print(f"    ✗ {tool_name} — failed: {exc}")

    if "Claude Code" in found:
        try:
            written = _write_claude_md(user_id)
            if written:
                print(f"    ✓ ~/.claude/CLAUDE.md — memory auto-trigger added")
            else:
                print(f"    – ~/.claude/CLAUDE.md — already configured, skipped")
        except Exception as exc:
            print(f"    ✗ ~/.claude/CLAUDE.md — failed: {exc}")

    # 5. Summary
    db_path = Path.home() / ".agent-magnet" / "memory.db"
    print()
    print("  " + "━" * 38)
    print("   Agent Magnet is configured!")
    print("  " + "━" * 38)
    print()
    print("  Memory works — stored locally on your machine.")
    print("  No accounts, no cloud, no keys needed.")
    print()
    print(f"  Active context: personal / general")
    print(f"  Memory stored at: {db_path}")
    print()
    print("  Restart your AI tools, then type *profiles or *projects to get started.")
    if not openai_key:
        print()
        print("  For smarter extraction, add MAGNET_OPENAI_KEY to your tool's MCP config.")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args: list[str] | None = None) -> None:
    argv = args if args is not None else sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: agent-magnet <command>")
        print()
        print("Commands:")
        print("  init    Configure AI tools with Agent Magnet")
        print()
        return

    if argv[0] == "init":
        cmd_init()
        return

    print(f"Unknown command: {argv[0]}")
    print("Run 'agent-magnet --help' for usage.")
    sys.exit(1)


if __name__ == "__main__":
    main()
