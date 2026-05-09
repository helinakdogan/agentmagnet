import os
import sys
import re
import redis
import logging
import hashlib
import datetime
import time
import litellm  
import json
from fastapi import FastAPI, Request, Header, HTTPException, BackgroundTasks, Body, Depends
from dotenv import load_dotenv

from proxy.auth import verify_api_key
from fastapi.middleware.cors import CORSMiddleware
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk"))
from magnet import BehavioralMemory

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = logging.getLogger(__name__)

app = FastAPI(title="Magnet Proxy", version="2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_url = os.getenv("REDIS_URL")
r_sync = None

if redis_url:
    try:
        r_sync = redis.from_url(redis_url, decode_responses=True)
        r_sync.ping()
        logger.info("Successfully connected to Redis.")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis at REDIS_URL. Falling back to in-memory mode. Error: {e}")
        r_sync = None
else:
    logger.info("REDIS_URL is not set. Redis features disabled (falling back to in-memory mode).")

vmm_memory = BehavioralMemory(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    redis_client=r_sync,
    signal_threshold=3,
    enable_aggregate=True,
)

# Cost-per-token lookup (input token price in USD)
GPT_4O_COST_PER_TOKEN = 0.000005  # $5 per 1M tokens (default baseline)
MODEL_COSTS: dict[str, float] = {
    "openai/gpt-4o-mini": 0.00000015,
    "openai/gpt-4o": 0.000005,
    "anthropic/claude-haiku-4-5": 0.00000025,
    "anthropic/claude-sonnet-4-6": 0.000003,
    "google/gemini-2.0-flash": 0.000000075,
}


def compute_cost_saved(model: str, p_tokens: int, c_tokens: int) -> float:
    """Calculate the dollar amount saved by routing to a cheaper model vs GPT-4o baseline."""
    total_tokens = p_tokens + c_tokens
    default_cost = total_tokens * GPT_4O_COST_PER_TOKEN
    actual_cost = total_tokens * MODEL_COSTS.get(model, GPT_4O_COST_PER_TOKEN)
    return max(0.0, default_cost - actual_cost)


async def safe_memory_add(vmm, messages_to_save: list, project_id: str, user_id: str, metadata: dict):
    """
    Asynchronously adds conversation data to the behavioral memory.
    Runs in the background to avoid blocking the request-response cycle.
    """
    try:
        await vmm.async_add(messages_to_save, user_id=user_id, project_id=project_id, metadata=metadata)
    except Exception as e:
        logger.error(f"Background memory add failed for session {user_id}: {str(e)}")


def get_ab_group(session_id: str) -> str:
    """
    Deterministically assigns a user to an A/B test group.
    Uses an MD5 hash of the session ID for consistent group assignment.
    """
    val = int(hashlib.md5(session_id.encode()).hexdigest()[:8], 16)
    return "test" if val % 100 < 50 else "control"


async def log_telemetry(
    redis_client,
    project_id: str,
    session_id: str,
    group: str,
    latency_ms: int,
    p_tokens: int,
    c_tokens: int,
    model: str,
    last_user_msg: str,
    routing_reason: str = "Default model used",
    cost_saved: float = 0.0,
):
    """
    Logs performance, A/B testing, routing, and cost-savings metrics to Redis,
    scoped by project_id. Runs as a background task.
    """
    if not redis_client:
        return

    project_stats_key = f"project:{project_id}:stats"
    project_events_key = f"project:{project_id}:events"

    try:
        correction_patterns = [
            r"\bno\b", r"\bnot\b", r"\bwrong\b", r"\bnot like that\b", r"\bfix\b",
            r"\bhay[ıi]r\b", r"\byanl[ıi][sş]\b", r"\b[oö]yle de[gğ]il\b", r"\bd[uü]zelt\b",
            r"\bupdate\b", r"\bchange\b", r"\binstead\b", r"\bmodify\b"
        ]
        rejection_patterns = [
            r"\bignore\b", r"\bforget\b", r"\bstart over\b", r"\bcancel\b", r"\bstop\b",
            r"\bbo[sş]ver\b", r"\biptal\b", r"\bgerek yok\b",
            r"\breject\b", r"\brefuse\b"
        ]

        last_msg_str = str(last_user_msg).lower()
        is_correction = any(re.search(p, last_msg_str) for p in correction_patterns)
        is_rejection = any(re.search(p, last_msg_str) for p in rejection_patterns)

        turn_number = redis_client.incr(f"project:{project_id}:turns:{session_id}")

        pipeline = redis_client.pipeline()
        pipeline.hincrby(project_stats_key, 'total_requests', 1)
        pipeline.hincrby(project_stats_key, 'total_tokens', p_tokens + c_tokens)
        pipeline.hincrby(project_stats_key, 'total_latency_ms', latency_ms)
        pipeline.hset(project_stats_key, 'last_activity', datetime.datetime.utcnow().isoformat())

        # Accumulate dollar savings from routing decisions
        if cost_saved > 0:
            pipeline.hincrbyfloat(project_stats_key, 'total_cost_saved', cost_saved)

        event = {
            "timestamp": time.time(),
            "user_id": session_id,
            "group": group,
            "turn_number": turn_number,
            "latency_ms": latency_ms,
            "tokens": p_tokens + c_tokens,
            "model": model,
            "selected_model": model,
            "routing_reason": routing_reason,
            "cost_saved": round(cost_saved, 6),
            "is_correction": is_correction,
            "is_rejection": is_rejection,
            "message": last_msg_str[:250],
        }
        pipeline.lpush(project_events_key, json.dumps(event, ensure_ascii=False))
        pipeline.ltrim(project_events_key, 0, 99)  # Keep last 100 events
        pipeline.execute()
    except Exception as e:
        logger.error(f"Telemetry logging failed: {e}")


@app.post("/v1/chat/completions")
async def chat_completions(
    background_tasks: BackgroundTasks,
    body: dict = Body(..., examples=[{
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Hello, explain Python to me in one sentence."}
        ],
        "temperature": 0.7
    }]),
    x_session_id: str = Header("default_session"),
    auth_data: dict = Depends(verify_api_key)
):
    messages = body.get("messages", [])

    ab_group = get_ab_group(x_session_id)

    # A/B Test: Inject behavioral profile only for users in the 'test' group.
    if ab_group == "test":
        injection = vmm_memory.get_injection(user_id=x_session_id, project_id=auth_data["project_id"], current_messages=messages)
        if injection:
            system_msg = next((m for m in messages if m.get("role") == "system"), None)
            if system_msg:
                system_msg["content"] += f"\n\n{injection}"
            else:
                messages.insert(0, {"role": "system", "content": f"You are a smart assistant.\n\n{injection}"})
            body["messages"] = messages

    # Determine routing decision (may return None if no router configured)
    recommended = vmm_memory.get_recommended_model(
        user_id=x_session_id,
        messages=messages,
        project_id=auth_data["project_id"],
    )

    routing_reason = "Default model used"
    if recommended:
        litellm_model = recommended.selected_model
        routing_reason = getattr(recommended, "reason", "Behavioral profile matched") or "Behavioral profile matched"
    else:
        requested_model = body.get("model", "openai/gpt-4o-mini")
        litellm_model = requested_model if "/" in requested_model else f"openai/{requested_model}"

    start_time = time.time()

    try:
        openai_api_key = auth_data.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        response = await litellm.acompletion(
            model=litellm_model,
            messages=messages,
            temperature=body.get("temperature", 0.7),
            max_tokens=body.get("max_tokens", None),
            api_key=openai_api_key,
        )
        latency_ms = int((time.time() - start_time) * 1000)
        ai_message = response.choices[0].message.content

        messages_to_save = messages + [{"role": "assistant", "content": ai_message}]
        metadata = {k: v for k, v in body.items() if k not in ["messages", "model"]}

        background_tasks.add_task(
            safe_memory_add,
            vmm_memory,
            messages_to_save,
            auth_data["project_id"],
            x_session_id,
            metadata
        )

        p_tokens, c_tokens = 0, 0
        if hasattr(response, "usage") and response.usage:
            p_tokens = getattr(response.usage, "prompt_tokens", 0)
            c_tokens = getattr(response.usage, "completion_tokens", 0)

        cost_saved = compute_cost_saved(litellm_model, p_tokens, c_tokens)

        user_msg = messages[-1].get("content", "") if messages else ""
        background_tasks.add_task(
            log_telemetry,
            r_sync,
            auth_data["project_id"],
            x_session_id,
            ab_group,
            latency_ms,
            p_tokens,
            c_tokens,
            litellm_model,
            user_msg,
            routing_reason,
            cost_saved,
        )

        if hasattr(response, "model_dump"):
            return response.model_dump()
        return dict(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM API Error: {str(e)}")


@app.get("/health")
async def health_check():
    """Provides a basic health check endpoint for the proxy."""
    try:
        if r_sync:
            r_sync.ping()
            redis_status = "connected"
        else:
            redis_status = "disabled"
        return {"status": "ok", "redis": redis_status, "uptime": time.time() - app.state.startup_time}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service Unavailable: {str(e)}")


@app.on_event("startup")
async def startup_event():
    app.state.startup_time = time.time()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
