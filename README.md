<p align="center">
  <img src="assets/logo.png" alt="Magnet" width="700">
</p>

<p align="center">
  <a href="https://agentmagnet.app/docs"><img src="https://img.shields.io/badge/Docs-agentmagnet.app-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://github.com/helinakdogan/magnet-gateway/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://agentmagnet.app"><img src="https://img.shields.io/badge/Built%20by-Agent%20Magnet-blueviolet?style=for-the-badge" alt="Built by Agent Magnet"></a>
  <img src="https://img.shields.io/pypi/v/magnet-gateway" alt="PyPI">
  <img src="https://img.shields.io/github/stars/helinakdogan/magnet-gateway" alt="Stars">
  <img src="https://img.shields.io/github/last-commit/helinakdogan/magnet-gateway" alt="Last Commit">
</p>

> Your AI forgets every user the moment the session ends.
> Magnet fixes that — without changing your code.

---

## How It Works

`User sends message → Magnet injects memory → LLM responds → Magnet learns`

- Learns from corrections, not just conversations
- Builds a profile that gets smarter with every interaction
- Compresses thousands of messages into a lightweight JSON snapshot

---

## Quick Start

### Proxy Mode (2 steps)

```bash
# Step 1: Start services
docker compose up -d
```

```python
# Step 2: Change your base URL — nothing else
import openai

client = openai.Client(
    base_url="http://localhost:8000/v1",
    api_key="your-openai-api-key"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_headers={"x-session-id": "user-123"}  # Magnet tracks memory per user
)
```

### SDK Mode

```bash
pip install agent-magnet
```

```python
from magnet import BehavioralMemory

memory = BehavioralMemory(reflector_model="openai/gpt-4o-mini")

# Get context for a user
context = memory.get_injection(user_id="alice")

# Add a conversation to memory
memory.add(messages, user_id="alice")
```

---

## Why Magnet

| | Traditional RAG | Magnet |
|---|---|---|
| **Setup** | Vector DB + embeddings + retrieval pipeline | ✅ Drop-in proxy or one import |
| **Latency** | Adds retrieval roundtrip on every call | ✅ O(1) injection, async learning |
| **Learning** | Static — you update it manually | ✅ Adapts from every interaction |
| **Privacy** | Shared embedding pool | ✅ Per-user, self-hosted, no data sharing |

---

## Architecture

Your AI remembers what matters across three layers — each one builds on the last.

**Layer 1 — Redis** (always on, real-time preferences and corrections)  
**Layer 2 — Qdrant** (episodic recall, semantic memory from past sessions)  
**Layer 3 — Neo4j** (relationships and long-term knowledge graph)

---

## Configuration

Set these in your `.env` file:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Used by the reflector model to analyze interactions. |
| `REDIS_URL` | e.g. `redis://localhost:6379`. Used for Layer 1. |
| `QDRANT_URL` | Used for Layer 2 episodic memory. |
| `NEO4J_URL` | Used for Layer 3 graph knowledge. |

---

## Documentation

Full docs at **[agentmagnet.app/docs](https://agentmagnet.app/docs)**:

| Section | What's Covered |
|---------|---------------|
| **Quickstart** | Install → setup → first interaction in 2 minutes |
| **Architecture** | Details on the 3-layer memory engine |
| **Proxy Mode** | How to use Magnet as a transparent gateway |
| **SDK Usage** | Deep integration into Python applications |
| **Self Hosting** | Instructions for running Redis, Qdrant, and Neo4j |

---

## Contributing

Open an issue or submit a pull request — check `CONTRIBUTING.md` for guidelines.

- **Discord**: [Join our Community](#) *(Coming Soon!)*
- **Issues**: [Report a bug or request a feature](https://github.com/helinakdogan/magnet-gateway/issues)
- **X**: [@AgentMagnetAI](https://twitter.com/AgentMagnetAI)

If Magnet saved you from a bad context window, give it a ⭐

---

## License

MIT — see [LICENSE](LICENSE). Built by [Agent Magnet](https://agentmagnet.app).

---

<!-- Topics: ai-agent, llm, memory, personalization, openai, python, self-hosted, rag-alternative -->
