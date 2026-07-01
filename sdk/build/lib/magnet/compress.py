"""
Compressor
----------
Lossless-ish context compression for large text blocks.

  compress(text)   -> (compressed_text, metadata)
  decompress(...)  -> original_text  (fully reversible via local cache)

Token estimate: len(text) / 4  (no tokenizer dependency).
Original is saved to ~/.agent-magnet/compress_cache/{sha16}.orig
so retrieve_original always works even after process restart.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR   = Path.home() / ".agent-magnet" / "compress_cache"
_STATS_FILE  = Path.home() / ".agent-magnet" / "compress_stats.jsonl"
_TOKEN_RATIO = 4      # chars per token (rough estimate)
_MIN_CHARS   = 500    # skip compression for short text
_HEAD_LINES  = 30     # long-text: keep first N
_TAIL_LINES  = 20     # long-text: keep last N
_LONG_THRESH = 100    # lines threshold for long_text strategy


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _tok(text: str) -> int:
    return max(1, len(text) // _TOKEN_RATIO)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _cache_path(key: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{key}.orig"


def _save_original(key: str, text: str) -> None:
    _cache_path(key).write_text(text, encoding="utf-8")


def _load_original(key: str) -> str | None:
    p = _cache_path(key)
    return p.read_text(encoding="utf-8") if p.exists() else None


def _record_stat(original_tokens: int, compressed_tokens: int, strategy: str) -> None:
    try:
        _STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _STATS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.time(),
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "saved_tokens": original_tokens - compressed_tokens,
                "strategy": strategy,
            }) + "\n")
    except Exception:
        pass


# ── Content type detection ─────────────────────────────────────────────────────

_LOG_PAT = re.compile(
    r"(\d{4}[-/]\d{2}[-/]\d{2}"
    r"|\d{2}:\d{2}:\d{2}"
    r"|\b(?:DEBUG|INFO|WARNING|ERROR|CRITICAL|WARN|FATAL)\b"
    r"|Traceback"
    r"|at \w+\.\w+\()"
)


def _detect(text: str) -> str:
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list) and len(parsed) > 3 and all(isinstance(i, dict) for i in parsed[:3]):
                return "json_array"
        except (json.JSONDecodeError, ValueError):
            pass

    lines = text.splitlines()
    if len(lines) > 10:
        matches = sum(1 for l in lines[:20] if _LOG_PAT.search(l))
        if matches >= 5:
            return "log"

    if len(lines) > _LONG_THRESH:
        return "long_text"

    return "whitespace"


# ── Strategies ─────────────────────────────────────────────────────────────────

def _compress_json_array(text: str) -> tuple[str, str]:
    parsed = json.loads(text.strip())
    total  = len(parsed)
    keys   = sorted({k for item in parsed for k in (item.keys() if isinstance(item, dict) else [])})
    samples = json.dumps(parsed[:3], indent=2, ensure_ascii=False)
    out = (
        f"[JSON array — {total} items]\n"
        f"Schema keys: {json.dumps(keys)}\n"
        f"First 3 samples:\n{samples}\n"
        f"... {total - 3} more items with same schema omitted"
    )
    return out, "json_array"


def _compress_log(text: str) -> tuple[str, str]:
    lines  = text.splitlines()
    result: list[str] = []
    seen:   dict[str, int] = {}
    in_trace   = False
    trace_buf: list[str] = []

    def _flush_trace() -> None:
        nonlocal in_trace, trace_buf
        if not trace_buf:
            return
        if len(trace_buf) > 4:
            result.append(trace_buf[0])
            result.append(f"  ... [{len(trace_buf) - 2} frames omitted] ...")
            result.append(trace_buf[-1])
        else:
            result.extend(trace_buf)
        in_trace  = False
        trace_buf = []

    for line in lines:
        stripped = line.strip()
        if "Traceback" in line or (in_trace and re.match(r"\s{2,}File ", line)):
            in_trace = True
            trace_buf.append(line)
            continue
        if in_trace:
            trace_buf.append(line)
            if stripped and not stripped.startswith(" ") and not stripped.startswith("\t"):
                _flush_trace()
            continue

        key = re.sub(r"\d+", "N", stripped)[:80]
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            result.append(line)
        elif seen[key] == 2:
            result.append("  [above line repeated, further duplicates hidden]")

    _flush_trace()
    return "\n".join(result), "log"


def _compress_long_text(text: str) -> tuple[str, str]:
    lines   = text.splitlines()
    total   = len(lines)
    omitted = total - _HEAD_LINES - _TAIL_LINES
    out = (
        "\n".join(lines[:_HEAD_LINES])
        + f"\n\n[... {omitted} lines omitted — use retrieve_original to see full text ...]\n\n"
        + "\n".join(lines[-_TAIL_LINES:])
    )
    return out, "long_text"


def _compress_whitespace(text: str) -> tuple[str, str]:
    lines  = text.splitlines()
    result: list[str] = []
    blank  = 0
    for line in lines:
        sl = line.rstrip()
        if not sl:
            blank += 1
            if blank <= 1:
                result.append("")
        else:
            blank = 0
            result.append(sl)
    return "\n".join(result).strip(), "whitespace"


# ── Public API ─────────────────────────────────────────────────────────────────

class Compressor:
    """
    Lossless context compressor backed by a local file cache.

    compress() never loses data — originals are written to
    ~/.agent-magnet/compress_cache/ and can be retrieved at any time
    via decompress() or the retrieve_original MCP tool.
    """

    def compress(self, text: str, content_type: str | None = None) -> tuple[str, dict]:
        orig_tok = _tok(text)

        if len(text) < _MIN_CHARS:
            return text, {
                "strategy": "none", "cache_key": None,
                "original_tokens": orig_tok, "compressed_tokens": orig_tok, "saved_tokens": 0,
            }

        detected = content_type or _detect(text)

        try:
            if detected == "json_array":
                compressed, strategy = _compress_json_array(text)
            elif detected == "log":
                compressed, strategy = _compress_log(text)
            elif detected == "long_text":
                compressed, strategy = _compress_long_text(text)
            else:
                compressed, strategy = _compress_whitespace(text)
        except Exception as exc:
            logger.warning(f"[compress] {detected} strategy failed ({exc}), falling back to whitespace")
            compressed, strategy = _compress_whitespace(text)

        comp_tok = _tok(compressed)

        # Only keep if we actually saved ≥10 %
        if comp_tok >= orig_tok * 0.90:
            return text, {
                "strategy": "none", "cache_key": None,
                "original_tokens": orig_tok, "compressed_tokens": orig_tok, "saved_tokens": 0,
            }

        cache_key = _sha(text)
        _save_original(cache_key, text)
        _record_stat(orig_tok, comp_tok, strategy)

        saved = orig_tok - comp_tok
        logger.info(f"[compress] {orig_tok:,} → {comp_tok:,} tokens (strategy={strategy}, saved={saved:,})")

        return compressed, {
            "strategy": strategy,
            "cache_key": cache_key,
            "original_tokens": orig_tok,
            "compressed_tokens": comp_tok,
            "saved_tokens": saved,
        }

    def decompress(self, compressed_text: str, metadata: dict) -> str:
        key = metadata.get("cache_key")
        if not key:
            return compressed_text
        original = _load_original(key)
        if original is None:
            logger.warning(f"[compress] cache miss for key={key}")
            return compressed_text
        return original

    def retrieve_by_key(self, cache_key: str) -> str | None:
        return _load_original(cache_key)

    def stats(self) -> dict:
        if not _STATS_FILE.exists():
            return {"total_events": 0, "total_saved_tokens": 0, "strategies": {}}
        events: list[dict] = []
        try:
            with _STATS_FILE.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            return {"total_events": 0, "total_saved_tokens": 0, "strategies": {}}

        total_saved = sum(e.get("saved_tokens", 0) for e in events)
        by_strategy: dict[str, int] = {}
        for e in events:
            s = e.get("strategy", "unknown")
            by_strategy[s] = by_strategy.get(s, 0) + 1

        return {
            "total_events": len(events),
            "total_original_tokens": sum(e.get("original_tokens", 0) for e in events),
            "total_saved_tokens": total_saved,
            "estimated_saved_usd": round(total_saved * 0.000005, 6),
            "strategies": by_strategy,
        }
