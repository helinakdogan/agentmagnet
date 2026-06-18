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

def _build_mcp_entry(user_id: str, openai_key: str, storage: str, redis_url: str) -> dict:
    env: dict[str, str] = {
        "MAGNET_USER_ID": user_id,
        "MAGNET_PROJECT_ID": "default",
    }
    if openai_key:
        env["MAGNET_OPENAI_KEY"] = openai_key
    if storage == "local":
        env["MAGNET_LOCAL_MODE"] = "1"
    else:
        env["MAGNET_REDIS_URL"] = redis_url

    return {"command": _mcp_command(), "env": env}


# ── Per-tool config writers ────────────────────────────────────────────────────

def _write_claude_code(
    path: Path, user_id: str, openai_key: str, storage: str, redis_url: str
) -> str:
    config = _read_json(path)

    # MCP server
    config.setdefault("mcpServers", {})
    config["mcpServers"]["agent-magnet"] = _build_mcp_entry(
        user_id, openai_key, storage, redis_url
    )

    # Stop hook — inline env vars so the hook process can read them
    env_pairs: list[tuple[str, str]] = [("MAGNET_USER_ID", user_id), ("MAGNET_PROJECT_ID", "default")]
    if openai_key:
        env_pairs.append(("MAGNET_OPENAI_KEY", openai_key))
    if storage == "local":
        env_pairs.append(("MAGNET_LOCAL_MODE", "1"))
    else:
        env_pairs.append(("MAGNET_REDIS_URL", redis_url))

    hook_entry: dict[str, Any] = {
        "type": "command",
        "command": _hook_command(env_pairs),
        "timeout": 10,
    }

    config.setdefault("hooks", {})
    existing_stop = config["hooks"].get("Stop", [])

    # Merge: replace existing agent-magnet hook or append a new block
    magnet_hook_block = {"matcher": "", "hooks": [hook_entry]}
    new_stop = [b for b in existing_stop if not _is_magnet_hook(b)]
    new_stop.append(magnet_hook_block)
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


def _write_mcp_only(path: Path, user_id: str, openai_key: str, storage: str, redis_url: str) -> str:
    config = _read_json(path)
    config.setdefault("mcpServers", {})
    config["mcpServers"]["agent-magnet"] = _build_mcp_entry(
        user_id, openai_key, storage, redis_url
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
        # For Claude Code / Claude Desktop / Cursor: detect by parent dir existence
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

    # 2. Questions
    print()
    print("  Configuration")
    print("  " + "─" * 38)
    print()

    user_id = _ask("  Your name/identifier (for memory): ")
    if not user_id:
        user_id = getpass.getuser()
        print(f"  Using system username: {user_id}")

    openai_key = _ask_secret("  OpenAI API key (or press Enter to set later): ")

    use_local = _ask_bool("  Use local storage? No Redis needed", default=True)

    redis_url = ""
    storage = "local"
    if not use_local:
        redis_url = _ask("  Redis URL (e.g. redis://localhost:6379): ")
        if not redis_url:
            print("  No Redis URL provided — falling back to local storage.")
            use_local = True
        else:
            storage = "redis"

    # 3. Write configs
    print()
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
            action = writer(path, user_id, openai_key, storage, redis_url)
            print(f"    ✓ {tool_name} — {action}")
        except Exception as exc:
            print(f"    ✗ {tool_name} — failed: {exc}")

    # 4. Summary
    print()
    print("  " + "━" * 38)
    print("   Agent Magnet is configured!")
    print("  " + "━" * 38)
    print()
    print("  Restart your AI tools to activate memory.")
    if use_local:
        db_path = Path.home() / ".agent-magnet" / "memory.db"
        print(f"  Memory stored at: {db_path}")
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
