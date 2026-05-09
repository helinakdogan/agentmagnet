# Magnet

Behavioral memory layer for AI agents.
Learns from what users DO, not just what they SAY.

Magnet acts as a transparent proxy and an SDK that observes user interactions, extracts behavioral signals (corrections, preferences, rejections), and dynamically injects context-aware profiles into your LLM prompts with zero added latency.

## 3-Layered Cognitive Memory Architecture

Magnet doesn't just dump all chat history into a vector database. It uses a tiered, cognitive architecture to balance latency, cost, and context relevance:

1. **Layer 1: Behavioral Memory (Always Active):** Learns user preferences, corrections, and formatting choices dynamically. This synthesized profile is injected into the System Prompt in `O(1)` time with zero added latency.
2. **Layer 2: Episodic Memory (Conditionally Active):** Stores important conversations based on "importance scores". It is only activated and searched (via Vector DB like Qdrant) when the user explicitly refers to past events (e.g., "like we discussed earlier").
3. **Layer 3: Knowledge Memory (Graph-Based):** A long-term, graph-based memory layer (Neo4j) designed to track structured entities and relationships over time.

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/magnet
cd magnet

# Configure your environment variables and API keys
cp .env.example .env        

# Start the Redis container
docker compose up -d        

# Install dependencies
pip install -r sdk/requirements.txt

# Start the Proxy Gateway (Runs on localhost:8000)
python proxy/main.py        

# (Optional) Start the Local Dashboard (Runs on localhost:8501)
streamlit run dashboard/app.py  
```

## SDK Usage

```bash
pip install git+https://github.com/YOUR_USERNAME/magnet#subdirectory=sdk
```

```python
from magnet import BehavioralMemory

memory = BehavioralMemory(reflector_model="openai/gpt-4o-mini")
memory.add(messages, user_id="alice")
```
