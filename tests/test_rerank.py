
import pytest

from evaluations.rerank import rerank_candidates


class FakeCrossEncoder:
    """A fake model that returns predefined relevance scores."""

    def __init__(self, scores):
        self.scores = scores
        self.received_pairs = None
        self.call_count = 0

    def predict(self, pairs):
        self.received_pairs = pairs
        self.call_count += 1
        return self.scores


@pytest.fixture
def sample_candidates():
    return [
        (
            "q1",
            "What is DPO?",
            {
                "rank": 1,
                "point_id": "chunk_a",
                "score": 0.95,
                "document": "paper_a.pdf",
                "text": "Chunk A",
            },
        ),
        (
            "q2",
            "What is ReAct?",
            {
                "rank": 1,
                "point_id": "chunk_b",
                "score": 0.91,
                "document": "paper_b.pdf",
                "text": "Chunk B",
            },
        ),
        (
            "q1",
            "What is DPO?",
            {
                "rank": 2,
                "point_id": "chunk_c",
                "score": 0.85,
                "document": "paper_a.pdf",
                "text": "Chunk C",
            },
        ),
        (
            "q2",
            "What is ReAct?",
            {
                "rank": 2,
                "point_id": "chunk_d",
                "score": 0.80,
                "document": "paper_b.pdf",
                "text": "Chunk D",
            },
        ),
    ]


def test_cross_encoder_receives_correct_pairs(sample_candidates):
    model = FakeCrossEncoder([0.5, 0.8, 0.9, 0.2])

    rerank_candidates(sample_candidates, model)

    assert model.received_pairs == [
        ("What is DPO?", "Chunk A"),
        ("What is ReAct?", "Chunk B"),
        ("What is DPO?", "Chunk C"),
        ("What is ReAct?", "Chunk D"),
    ]


def test_reranking_groups_by_query(sample_candidates):
    model = FakeCrossEncoder([0.5, 0.8, 0.9, 0.2])

    results = rerank_candidates(sample_candidates, model)

    assert set(results.keys()) == {"q1", "q2"}
    assert len(results["q1"]) == 2
    assert len(results["q2"]) == 2


def test_reranking_orders_by_score(sample_candidates):
    model = FakeCrossEncoder([0.5, 0.8, 0.9, 0.2])

    results = rerank_candidates(sample_candidates, model)

    # q1: chunk C should move ahead of chunk A
    assert results["q1"][0]["point_id"] == "chunk_c"
    assert results["q1"][0]["reranked_rank"] == 1

    assert results["q1"][1]["point_id"] == "chunk_a"
    assert results["q1"][1]["reranked_rank"] == 2

    # q2: original ranking should remain unchanged
    assert results["q2"][0]["point_id"] == "chunk_b"
    assert results["q2"][1]["point_id"] == "chunk_d"


def test_original_metadata_is_preserved(sample_candidates):
    model = FakeCrossEncoder([0.5, 0.8, 0.9, 0.2])

    results = rerank_candidates(sample_candidates, model)

    first_result = results["q1"][0]

    assert first_result["rank"] == 2
    assert first_result["score"] == 0.85
    assert first_result["document"] == "paper_a.pdf"
    assert first_result["text"] == "Chunk C"

    assert first_result["reranked_rank"] == 1
    assert first_result["reranker_score"] == pytest.approx(0.9)
    assert isinstance(first_result["reranker_score"], float)


def test_empty_candidates():
    model = FakeCrossEncoder([])

    results = rerank_candidates([], model)

    assert results == {}
    assert model.call_count == 0


def test_score_count_mismatch(sample_candidates):
    # Four candidates, but only two scores returned
    model = FakeCrossEncoder([0.5, 0.8])

    with pytest.raises(
        ValueError, match="Cross-encoder score count mismatch"
    ):
        rerank_candidates(sample_candidates, model)
