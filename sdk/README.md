# Agent Magnet

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

## How It Learns

Magnet observes behavioral signals — corrections, rejections,
implicit patterns — and builds a living profile per user.
No configuration required.

Three memory layers:
- **Behavioral** (Redis) — real-time, every request
- **Episodic** (Qdrant) — semantic recall when relevant
- **Knowledge** (Neo4j) — long-term entity relationships

## Claude Code Setup

1. Install:

```
pip install agent-magnet
```

2. Get a free Redis URL: [upstash.com](https://upstash.com) (takes 1 minute)

3. Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "",
        "hooks": [{
          "type": "command",
          "command": "MAGNET_REDIS_URL=your_redis_url MAGNET_OPENAI_KEY=your_openai_key MAGNET_USER_ID=your_name python -m magnet.hooks.save_session",
          "timeout": 10
        }]
      }
    ]
  }
}
```

4. Add to the same file (`mcpServers` section, for reading memory):

```json
{
  "mcpServers": {
    "agent-magnet": {
      "command": "agent-magnet-mcp",
      "env": {
        "MAGNET_REDIS_URL": "your_redis_url",
        "MAGNET_OPENAI_KEY": "your_openai_key",
        "MAGNET_USER_ID": "your_name"
      }
    }
  }
}
```

5. Restart Claude Code.

Done. Magnet now saves what it learns when a session ends, and loads it when a new session starts — type `load my memory` to load manually, or it happens automatically via the hook.

Use the same `MAGNET_USER_ID` across Claude Code, Cursor, and Codex to share memory between tools.

## License
MIT
