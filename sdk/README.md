# Agent Magnet

<!-- mcp-name: io.github.helinakdogan/agent-magnet -->

Self-learning memory infrastructure for AI products.
It learns from what users do — not what they say.

## Installation

```
pip install agent-magnet
```

## Two Integration Modes

### 1. MCP Server (Free, Self-Hosted)
Run on your own infrastructure. You control your data.

Add to your MCP config (Claude Desktop / Cursor / any MCP client):

```json
{
  "mcpServers": {
    "agent-magnet": {
      "command": "agent-magnet-mcp",
      "env": {
        "MAGNET_REDIS_URL": "your_redis_url",
        "MAGNET_OPENAI_KEY": "your_openai_key"
      }
    }
  }
}
```

Tools available:
- `get_profile` — get learned memory profile for a user
- `inject_memory` — get memory injection string for system prompt
- `add_signal` — record a behavioral signal
- `get_cold_start` — get onboarding profile for new users
- `get_team_profile` — get shared team memory (requires Redis)
- `get_merged_injection` — merged user + team + org memory injection
- `get_project_memory` — per-user breakdown of what was learned in a project
- `share_to_team` — explicitly share a personal preference to team memory
- `forget_team` — remove a preference from team memory
- `add_team_signal` — record a signal directly to team scope

### 2. Proxy (Hosted, Dashboard included)
Change one line. We handle the infrastructure.

```python
from openai import OpenAI

client = OpenAI(
    api_key="mg_sk_...",
    base_url="https://magnet-gateway.onrender.com/v1",
    default_headers={"x-session-id": "user_123"}
)
```

Get your API key: agentmagnet.app

## Team Memory

Memory works for individual users out of the box. To share memory across a team, add Redis and a shared `MAGNET_TEAM_ID`.

**Solo (local SQLite, no Redis needed):**
```bash
agent-magnet init
# Use local storage? Y
# Team ID: (press Enter to skip)
```
Each person's preferences are stored privately on their machine.

**Team (shared Redis):**
```bash
agent-magnet init
# Use local storage? N
# Redis URL: redis://your-redis-host:6379
# Team ID: acme-eng
```
All team members point to the same Redis. Memory is scoped per-user but team-level insights are available.

**What team memory gives you:**

```
get_project_memory(project_id="acme-app")
# →
# {
#   "contributors": {
#     "ahmet": {"prefers": ["short responses", "Turkish"], "watch_out": ["never use em-dashes"]},
#     "ayse":  {"prefers": ["detailed explanations"], "dislikes": ["bullet lists"]}
#   },
#   "team_shared": {
#     "prefers": ["short responses"],   ← promoted because 2+ users share it
#     "watch_out": []
#   }
# }
```

**Explicitly share a preference to team:**
```
share_to_team(user_id="ahmet", fact_or_subject="short responses", team_id="acme-eng")
```

**Team memory requires Redis.** If you try to use team tools in local mode you'll get:
> Team memory requires shared storage. Set MAGNET_REDIS_URL for all team members to use the same Redis instance.

## How It Learns

Magnet observes behavioral signals — corrections, rejections,
implicit patterns — and builds a living profile per user.
No configuration required.

Three memory layers:
- **Behavioral** (Redis) — real-time, every request
- **Episodic** (Qdrant) — semantic recall when relevant
- **Knowledge** (Neo4j) — long-term entity relationships

## Free vs Premium

Agent Magnet is fully usable without an account. The free tier is not a demo — it's the real thing.

**Free (no account needed)**
- Local SQLite memory — no Redis, no external services
- Single-user behavioral memory (preferences, corrections, forgetting)
- Cross-tool identity — same memory across Claude Code, Cursor, any MCP client
- Context compression (`compress_context`, `retrieve_original`)
- Bring your own OpenAI key (BYOK)

**Premium (API key from agentmagnet.app)**
- Team memory — share learned preferences across a team with a shared Redis
- Hosted storage — Magnet-managed Redis, no infra to run
- Compression analytics (`compression_stats`)
- Priority support

To enable premium features, set `MAGNET_API_KEY=mg_sk_...` in your environment or pass it during `agent-magnet init`.

## License
MIT
