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

How it works end-to-end:
- **Session start** — Claude automatically reads your memory profile and uses it
- **During the session** — Claude learns from your corrections, preferences, and rejections
- **Session end** — a Stop hook saves everything to Redis before Claude Code closes

### Step 1 — Install

```bash
pipx install agent-magnet
```

Get a free Redis URL at [upstash.com](https://upstash.com) (takes 1 minute).

### Step 2 — Add the Stop hook and MCP server

In `~/.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "",
        "hooks": [{
          "type": "command",
          "command": "MAGNET_REDIS_URL=your_redis_url MAGNET_OPENAI_KEY=your_openai_key MAGNET_USER_ID=your_name MAGNET_PROJECT_ID=default /path/to/pipx/venvs/agent-magnet/bin/python -m magnet.hooks.save_session",
          "timeout": 10
        }]
      }
    ]
  },
  "mcpServers": {
    "agent-magnet": {
      "command": "agent-magnet-mcp",
      "env": {
        "MAGNET_REDIS_URL": "your_redis_url",
        "MAGNET_OPENAI_KEY": "your_openai_key",
        "MAGNET_USER_ID": "your_name",
        "MAGNET_PROJECT_ID": "default"
      }
    }
  }
}
```

To find your pipx Python path: `pipx environment | grep PIPX_HOME`  
Then the full path is: `{PIPX_HOME}/venvs/agent-magnet/bin/python`

### Step 3 — Tell Claude to load memory automatically

Create `~/.claude/CLAUDE.md` (global instructions Claude reads at the start of every session):

```markdown
# Memory

At the start of every conversation, call the `inject_memory` MCP tool (agent-magnet) with:
- user_id: "your_name"
- project_id: "default"

Use the returned memory profile as context for the conversation.
```

This is the critical step. Without it, memory is saved but never loaded into the conversation.

### Step 4 — Restart Claude Code

That's it. From now on:
- Every new conversation starts with your memory profile loaded
- Every closed session is saved automatically
- No manual commands needed

Use the same `MAGNET_USER_ID` across Claude Code, Cursor, and Codex to share memory between tools.

### What you can say during a session

Memory loads automatically at the start, but Claude doesn't always proactively record things mid-session. These phrases work reliably:

| What you want | What to say |
|---|---|
| Load your profile into this conversation | `get my data from agent-magnet` |
| Save something you just said | `record it to agent-magnet` |
| Save the whole session now | `save this session to my memory` |
| Check what Magnet knows about you | `what's in my agent-magnet profile` |

You don't need exact phrasing — Claude understands intent and will call the right MCP tool. But if it doesn't, these always work.

---

## Cursor Setup

### Option A — MCP (automatic load, manual save)

Cursor doesn't support Stop hooks, so sessions must be saved manually.

1. Install: `pipx install agent-magnet`
2. Get a free Redis URL at [upstash.com](https://upstash.com)
3. Add to Cursor MCP config (Settings → MCP):

```json
{
  "mcpServers": {
    "agent-magnet": {
      "command": "agent-magnet-mcp",
      "env": {
        "MAGNET_REDIS_URL": "your_redis_url",
        "MAGNET_OPENAI_KEY": "your_openai_key",
        "MAGNET_USER_ID": "your_name",
        "MAGNET_PROJECT_ID": "default"
      }
    }
  }
}
```

4. Add to Cursor Rules (Settings → Rules for AI):

```
At the start of every conversation, call the inject_memory MCP tool (agent-magnet) with user_id="your_name" and project_id="default". Use the result as context.
```

5. At the end of a session, type: `save this session to my memory`

Use the same `MAGNET_USER_ID` as Claude Code — memory is shared across tools.

### Option B — Proxy (fully automatic)

1. Go to Cursor Settings → Models
2. Set "Override OpenAI Base URL" to: `https://magnet-gateway.onrender.com/v1`
3. Enter your Agent Magnet API key from [agentmagnet.app](https://agentmagnet.app)
4. Add header: `x-magnet-user-id: your_name`

Every request automatically saves and recalls memory. No manual commands, no setup beyond this.

---

## Contributing

- **Issues**: [Report a bug or request a feature](https://github.com/helinakdogan/magnet-gateway/issues)
- **X**: [@AgentMagnetAI](https://twitter.com/AgentMagnetAI)

If Magnet saved you from a bad context window, give it a ⭐

---

## License

MIT — see [LICENSE](LICENSE). Built by [Agent Magnet](https://agentmagnet.app).

<!-- Topics: ai-agent-memory, llm-memory, persistent-memory, mcp-server, openai-proxy, anthropic, gemini, self-hosted-ai, rag-alternative, multi-agent, cross-session-memory, behavioral-learning, python, langchain, crewai -->