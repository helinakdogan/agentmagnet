from unittest.mock import patch

from magnet import local_embeddings


def test_keyword_fallback_excludes_zero_overlap_documents():
    documents = [{"summary": "Python dependency decision"}]

    with patch.object(local_embeddings, "_get_embedder", return_value=None):
        result = local_embeddings.rank_by_similarity(
            "weather forecast", documents, text_key="summary", top_k=1
        )

    assert result == []


def test_keyword_fallback_keeps_related_documents():
    related = {"summary": "Python dependency decision"}
    unrelated = {"summary": "Quarterly sales forecast"}

    with patch.object(local_embeddings, "_get_embedder", return_value=None):
        result = local_embeddings.rank_by_similarity(
            "Python dependency", [unrelated, related], text_key="summary", top_k=2
        )

    assert result == [related]


def test_embedding_ranking_filters_candidates_below_minimum_similarity():
    related = {"summary": "related"}
    unrelated = {"summary": "unrelated"}
    vectors = {
        "query": [1.0, 0.0],
        "related": [0.8, 0.6],
        "unrelated": [0.0, 1.0],
    }

    with (
        patch.object(local_embeddings, "_get_embedder", return_value=object()),
        patch.object(local_embeddings, "embed", side_effect=lambda text: vectors[text]),
    ):
        result = local_embeddings.rank_by_similarity(
            "query", [unrelated, related], text_key="summary", top_k=2
        )

    assert result == [related]


def test_embedding_failure_falls_back_without_returning_unrelated_documents():
    documents = [{"summary": "Python dependency decision"}]

    with (
        patch.object(local_embeddings, "_get_embedder", return_value=object()),
        patch.object(local_embeddings, "embed", return_value=None),
    ):
        result = local_embeddings.rank_by_similarity(
            "weather forecast", documents, text_key="summary", top_k=1
        )

    assert result == []
