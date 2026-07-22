import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call

from magnet.aggregate_store import AggregateSignalStore
from magnet.local_store import SQLiteBackend
from magnet.postgres_store import _PgPipeline
from magnet.signals import SignalDetector


def _sqlite_backend():
    tmp = tempfile.TemporaryDirectory()
    backend = SQLiteBackend(Path(tmp.name) / "memory.db")
    return tmp, backend


def test_sqlite_expired_collections_are_not_returned():
    tmp, backend = _sqlite_backend()
    try:
        backend.rpush("signals", "one")
        backend.zadd("episodes", {"one": 1.0})
        backend.expire("signals", -1)
        backend.expire("episodes", -1)

        assert backend.llen("signals") == 0
        assert backend.lrange("signals", 0, -1) == []
        assert backend.zcard("episodes") == 0
        assert backend.zrevrange("episodes", 0, -1) == []
    finally:
        tmp.cleanup()


def test_sqlite_hash_expiration_and_delete_use_the_public_key():
    tmp, backend = _sqlite_backend()
    try:
        backend.hset("history", "temperature", '["0.1"]')
        backend.expire("history", -1)
        assert backend.hgetall("history") == {}

        backend.hset("history", "temperature", '["0.2"]')
        assert backend.delete("history") == 1
        assert backend.hgetall("history") == {}
    finally:
        tmp.cleanup()


def test_sqlite_pipeline_supports_signal_and_aggregate_operations():
    tmp, backend = _sqlite_backend()
    try:
        results = (
            backend.pipeline()
            .delete("history")
            .hset("history", "temperature", '["0.1"]')
            .expire("history", 60)
            .incr("aggregate")
            .execute()
        )

        assert results[-1] == 1
        assert backend.hgetall("history") == {"temperature": '["0.1"]'}
        assert backend.get("aggregate") == "1"

        backend.set("expired-aggregate", "9", ex=-1)
        assert backend.incr("expired-aggregate") == 1
    finally:
        tmp.cleanup()


def test_parameter_history_survives_detector_recreation_on_sqlite():
    tmp, backend = _sqlite_backend()
    try:
        first = SignalDetector(param_change_threshold=3, redis_client=backend)
        first.detect([], "session", {"temperature": 0.1})

        second = SignalDetector(param_change_threshold=3, redis_client=backend)
        assert second.detect([], "session", {"temperature": 0.2}) == []
        signals = second.detect([], "session", {"temperature": 0.3})

        assert signals[0]["type"] == "parameter_change"
    finally:
        tmp.cleanup()


def test_aggregate_store_records_counters_on_sqlite():
    tmp, backend = _sqlite_backend()
    try:
        AggregateSignalStore(backend).record(
            "preference", "coding", "tone", "concise"
        )

        assert backend.get("magnet:agg:preference:coding:tone:concise") == "1"
        assert backend.get("magnet:agg:count:preference:coding") == "1"
    finally:
        tmp.cleanup()


def test_postgres_pipeline_exposes_required_operations():
    backend = MagicMock()
    backend.delete.return_value = 1
    backend.hset.return_value = None
    backend.incr.return_value = 2

    results = (
        _PgPipeline(backend)
        .delete("history")
        .hset("history", "temperature", '["0.1"]')
        .incr("aggregate")
        .execute()
    )

    assert results == [1, None, 2]
    assert backend.method_calls == [
        call.delete("history"),
        call.hset("history", "temperature", '["0.1"]'),
        call.incr("aggregate"),
    ]
