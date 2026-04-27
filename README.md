# Magnet

Behavioral memory layer for AI agents.
Learns from what users DO, not just what they SAY.

Magnet acts as a transparent proxy and an SDK that observes user interactions, extracts behavioral signals (corrections, preferences, rejections), and dynamically injects context-aware profiles into your LLM prompts with zero added latency.

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
