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

## License
MIT
