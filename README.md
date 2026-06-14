<p align="center">
  <img src="assets/logo.png" alt="Magnet" width="700">
</p>

<p align="center">
  <a href="https://agentmagnet.app/docs">
    <img src="https://img.shields.io/badge/Docs-agentmagnet.app-8B5CF6?style=for-the-badge">
  </a>
  <a href="https://github.com/helinakdogan/magnet-gateway/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-A855F7?style=for-the-badge">
  </a>
  <a href="https://agentmagnet.app">
    <img src="https://img.shields.io/badge/Built%20by-Agent%20Magnet-C084FC?style=for-the-badge">
  </a>
  <img src="https://img.shields.io/pypi/v/agent-magnet?label=PyPI&labelColor=111827&color=8B5CF6" alt="PyPI">
  <img src="https://img.shields.io/github/last-commit/helinakdogan/magnet-gateway?label=Last%20commit&labelColor=111827&color=C084FC" alt="Last Commit">
</p>

> Your AI forgets every user the moment the session ends.  
> Magnet fixes that — without changing your code.

---

## How It Works

`User sends message → Magnet injects memory → LLM responds → Magnet learns`

- Learns from corrections, rejections, and implicit patterns — not just conversations
- Builds a persistent profile that improves with every interaction
- Knows what to forget: permanent, contextual, and transient signals decay at different rates
- Cross-user learning: patterns from one user improve cold-start for the next

---

## Two Ways to Integrate

### 1. Proxy Mode — zero code changes

Works with **OpenAI, Anthropic, Google Gemini**, and any OpenAI-compatible client.

```python
from openai import OpenAI

client = OpenAI(
    api_key="mg_sk_...",
    base_url="https://magnet-gateway.onrender.com/v1",
    default_headers={"x-session-id": "user_123"}
)

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",  # or anthropic/claude-haiku-4-5, google/gemini-flash
    messages=[{"role": "user", "content": "Hello"}]
)
```

Get your API key: **[agentmagnet.app](https://agentmagnet.app)**

### 2. MCP Server — self-hosted, your data stays with you

Works with **Claude Desktop, Cursor**, and any MCP client.

```bash
pip install agent-magnet
```

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

**MCP tools available:**
- `get_profile` — get the learned memory profile for a user
- `inject_memory` — get a memory string ready to inject into system prompt
- `add_signal` — record a behavioral signal (correction, rejection, preference)
- `get_cold_start` — get an onboarding profile for a new user based on aggregate patterns

### 3. SDK Mode — deep integration

```bash
pip install agent-magnet
```

```python
from magnet import BehavioralMemory

memory = BehavioralMemory(reflector_model="openai/gpt-4o-mini")

context = memory.get_injection(user_id="alice")
memory.add(messages, user_id="alice")
```

---

## Why Magnet

| | Traditional RAG | Mem0 / Zep | Magnet |
|---|---|---|---|
| **Setup** | Weeks | Days (SDK) | ✅ 1 minute |
| **Learning** | Static | Explicit only | ✅ From behavior |
| **Forgetting** | None | None | ✅ Multi-parameter decay |
| **Cross-user learning** | No | No | ✅ Consolidation engine |
| **Model support** | Any | Any | ✅ OpenAI, Anthropic, Gemini |
| **Self-hosted** | Yes | Partial | ✅ MCP + on-premise SDK |

---

## Architecture

Three memory layers — each one builds on the last.

**Layer 1 — Behavioral (Redis)**  
Always on, zero latency. Learns preferences, corrections, and rejections in real time. Signals decay by type: permanent (e.g. "hates mushrooms"), contextual (e.g. "prefers bullet lists"), transient (e.g. "wants short answers today").

**Layer 2 — Episodic (Qdrant)**  
Semantic recall from past sessions. Triggered only when relevant — no bloat, no noise.

**Layer 3 — Knowledge (Neo4j)**  
Long-term entity relationships. `PREFERRED_BY`, `REJECTED_BY`, `EXPECTED_BY` — structured understanding of who the user is.

**Consolidation Engine**  
Runs every 24 hours. Extracts cross-user patterns anonymously. New users don't start from zero.

---

## Configuration

| Variable | Description |
|----------|-------------|
| `MAGNET_REDIS_URL` | Redis for behavioral layer |
| `MAGNET_OPENAI_KEY` | Used by the reflector model |
| `QDRANT_URL` | Episodic memory layer |
| `NEO4J_URL` | Knowledge graph layer |

---

## Documentation

Full docs at **[agentmagnet.app/docs](https://agentmagnet.app/docs)**

---

## Claude Code Setup

1. Install:

```bash
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
          "command": "MAGNET_REDIS_URL=your_redis_url MAGNET_OPENAI_KEY=your_openai_key MAGNET_USER_ID=your_name MAGNET_PROJECT_ID=your_project_uuid python -m magnet.hooks.save_session",
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
        "MAGNET_USER_ID": "your_name",
        "MAGNET_PROJECT_ID": "your_project_uuid"
      }
    }
  }
}
```

`MAGNET_PROJECT_ID` is the project UUID from your [Agent Magnet dashboard](https://agentmagnet.app). It must match the project your proxy API key belongs to — this is what lets Claude Code and Cursor share the same memory.

5. Restart Claude Code.

Done. Magnet now saves what it learns when a session ends, and loads it when a new session starts — type `load my memory` to load manually, or it happens automatically via the hook.

Use the same `MAGNET_USER_ID` across Claude Code, Cursor, and Codex to share memory between tools.

---

## Cursor Setup

### Option A — MCP (read memory, manual save)

1. Install:

```bash
pip install agent-magnet
```

2. Get a free Redis URL: [upstash.com](https://upstash.com)

3. Add to Cursor MCP config (Settings → MCP):

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

4. Restart Cursor. Type `load my memory` at the start of a session, and `save this session` at the end.

Use the same `MAGNET_USER_ID` as Claude Code to share memory between tools.

### Option B — Proxy (fully automatic, no manual commands)

1. Go to Cursor Settings → Models → OpenAI API Key
2. Set "Override OpenAI Base URL" to: `https://magnet-gateway.onrender.com/v1`
3. Use your Agent Magnet API key (get one at [agentmagnet.app](https://agentmagnet.app))
4. Add header `x-session-id: your_name` in the same settings

Now every request automatically learns and recalls memory — no manual commands needed. Best for users who want zero friction.

---

## Contributing

- **Issues**: [Report a bug or request a feature](https://github.com/helinakdogan/magnet-gateway/issues)
- **X**: [@AgentMagnetAI](https://twitter.com/AgentMagnetAI)

If Magnet saved you from a bad context window, give it a ⭐

---

## License

MIT — see [LICENSE](LICENSE). Built by [Agent Magnet](https://agentmagnet.app).

<!-- Topics: ai-agent-memory, llm-memory, persistent-memory, mcp-server, openai-proxy, anthropic, gemini, self-hosted-ai, rag-alternative, multi-agent, cross-session-memory, behavioral-learning, python, langchain, crewai -->