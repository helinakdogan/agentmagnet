<p align="center">
  <img src="assets/logo.png" alt="Magnet" width="200">
</p>

# Magnet

<p align="center">
  <a href="https://agentmagnet.app/docs"><img src="https://img.shields.io/badge/Docs-agentmagnet.app-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://github.com/helinakdogan/magnet-gateway/issues"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/helinakdogan/magnet-gateway/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://agentmagnet.app"><img src="https://img.shields.io/badge/Built%20by-Agent%20Magnet-blueviolet?style=for-the-badge" alt="Built by Agent Magnet"></a>
</p>

**The behavioral adaptation engine for AI products built by [Agent Magnet](https://agentmagnet.app).** It's the only AI gateway with a built-in learning loop — it creates behavioral profiles from experience, improves them during use, nudges itself to persist preferences, and builds a deepening model of how your users interact with AI across sessions. Run it on your infrastructure with zero latency overhead. 

Use any model you want — OpenAI, Anthropic, Google Gemini, or any LiteLLM-compatible endpoint. Switch models dynamically without rewriting your application. No code changes, no lock-in.



<table>
<tr><td><b>Zero Latency Overhead</b></td><td>Memory is injected in <code>O(1)</code> time before the LLM call. Unlike traditional RAG systems that increase response time, Magnet gets out of the way thanks to its asynchronous architecture.</td></tr>
<tr><td><b>Lives where you do</b></td><td>Works with any OpenAI/LiteLLM compatible client by just changing the base URL. No SDKs to install if you use Proxy Mode.</td></tr>
<tr><td><b>A closed learning loop</b></td><td>Agent-curated memory that learns user preferences, corrections, and formatting choices dynamically. Synthesized into a profile and injected into the System Prompt.</td></tr>
<tr><td><b>Smart Token Optimization</b></td><td>Stop burning tokens on huge context windows. Magnet compresses a chat of thousands of messages into a refined lightweight JSON profile (e.g., 'likes short answers, wants markdown').</td></tr>
<tr><td><b>3-Layered Adaptation Engine</b></td><td>Layer 1: Behavioral Adaptation (Always Active). Layer 2: Episodic Memory (Conditionally Active via Vector DB). Layer 3: Knowledge Graph (Graph-Based via Neo4j).</td></tr>
<tr><td><b>Enterprise-Grade Privacy</b></td><td>Magnet never keeps user messages in a global pool. It anonymizes data with Differential Privacy and K-Anonymity (k=5) standards. PII data leaves no trace.</td></tr>
</table>

---

## Quick Install

### 1. Run the Gateway (Proxy Mode)

The easiest way to use Magnet is to run it as a proxy.

```bash
git clone https://github.com/helinakdogan/magnet-gateway.git
cd magnet-gateway

# Configure your environment variables
cp .env.example .env        

# Start required services (Redis, Qdrant, Neo4j)
docker compose up -d        

# Install proxy dependencies
pip install -r proxy/requirements.txt

# Start the Proxy Gateway (Runs on localhost:8000)
python proxy/main.py        
```

Now, point your LLM client to the proxy:

```python
import openai

client = openai.Client(
    base_url="http://localhost:8000/v1",
    api_key="your-openai-api-key"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_headers={"x-session-id": "user-123"}
)
```

### 2. Deep Integration (SDK Mode)

If you want to integrate Magnet directly into your Python backend:

```bash
pip install git+https://github.com/helinakdogan/magnet-gateway#subdirectory=sdk
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

## Configuration

Set these in your `.env` file:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Used by the reflector model to analyze behavior. |
| `REDIS_URL` | e.g. `redis://localhost:6379`. Used for Layer 1. |
| `QDRANT_URL` | Used for Layer 2 episodic memory. |
| `NEO4J_URL` | Used for Layer 3 graph knowledge. |

---

## Documentation

All documentation lives at **[agentmagnet.app/docs](https://agentmagnet.app/docs)**:

| Section | What's Covered |
|---------|---------------|
| **Quickstart** | Install → setup → first interaction in 2 minutes |
| **Architecture** | Details on the 3-Layered Adaptation Engine |
| **Proxy Mode** | How to use Magnet as a transparent gateway |
| **SDK Usage** | Deep integration into Python applications |
| **Self Hosting** | Instructions for running Redis, Qdrant, and Neo4j |

---

## Community

We welcome contributions! Please open an issue or submit a pull request. Check out our `CONTRIBUTING.md` for guidelines.

- 💬 **Discord**: [Join our Community](#) *(Coming Soon!)*
- 🐛 **Issues**: [Report a bug or request a feature](https://github.com/helinakdogan/magnet-gateway/issues)
- 🐦 **Twitter**: [@AgentMagnetAI](https://twitter.com/AgentMagnetAI)

---

## License

MIT — see [LICENSE](LICENSE).

Built by [Agent Magnet](https://agentmagnet.app).
