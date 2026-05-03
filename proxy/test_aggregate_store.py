import pytest
import fakeredis
from magnet.aggregate_store import AggregateSignalStore

@pytest.fixture
def redis_client():
    return fakeredis.FakeStrictRedis()

def test_record_and_get_prior_below_k(redis_client):
    store = AggregateSignalStore(redis_client, min_k=5, epsilon=1.0)
    store.record("preference", "coding", "response_length", "short")
    # Only 1 record, should return None due to k-anonymity
    prior = store.get_prior("coding", "response_length")
    assert prior is None

def test_record_and_get_prior_above_k(redis_client):
    store = AggregateSignalStore(redis_client, min_k=5, epsilon=1.0)
    for _ in range(6):
        store.record("preference", "coding", "response_length", "short")
    
    prior = store.get_prior("coding", "response_length")
    assert prior is not None
    assert "short" in prior

def test_reject_pii_signal_types(redis_client):
    store = AggregateSignalStore(redis_client, min_k=5, epsilon=1.0)
    # Invalid dimension shouldn't record
    store.record("preference", "coding", "user_id_123", "short")
    prior = store.get_prior("coding", "user_id_123")
    assert prior is None