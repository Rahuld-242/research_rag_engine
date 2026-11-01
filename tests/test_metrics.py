
import pytest
import math

from evaluations.metrics import (
    precision_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    mean_precision_at_k,
    dcg_at_k,
    idcg_at_k,
    ndcg_at_k, 
    mean_ndcg_at_k
)


def test_precision_at_5():
    grades = [0, 1, 0, 1, 1]
    assert precision_at_k(grades, 5) == pytest.approx(0.6)


def test_precision_at_10():
    grades = [2, 1, 0, 2, 1, 0, 2, 1, 0, 0]
    assert precision_at_k(grades, 10) == pytest.approx(0.6)


def test_strict_precision():
    grades = [2, 1, 0, 2, 1]
    assert precision_at_k(grades, 5, relevant_threshold=2) == pytest.approx(0.4)


def test_no_relevant_chunks():
    grades = [0, 0, 0, 0, 0]
    assert precision_at_k(grades, 5) == pytest.approx(0.0)


@pytest.mark.parametrize("k", [0, -1, 6])
def test_invalid_k(k):
    grades = [2, 1, 0, 1, 2]
    with pytest.raises(ValueError):
        precision_at_k(grades, k)


def test_rr_first_rank():
    assert reciprocal_rank([2, 1, 0, 2]) == pytest.approx(1.0)


def test_rr_third_rank():
    assert reciprocal_rank([0, 1, 2, 2]) == pytest.approx(1 / 3)


def test_rr_last_rank():
    assert reciprocal_rank([1, 0, 1, 2]) == pytest.approx(0.25)


def test_rr_no_relevant_result():
    assert reciprocal_rank([1, 0, 1, 0]) == 0.0


def test_rr_empty_grades():
    assert reciprocal_rank([]) == 0.0
    
def test_mrr_multiple_queries():
    grades_by_query = {
        "q1": [2, 1, 0],  # RR = 1.0
        "q2": [1, 2, 0],  # RR = 0.5
        "q3": [0, 0, 1],  # RR = 0.0
    }

    assert mean_reciprocal_rank(grades_by_query) == pytest.approx(0.5)


def test_mrr_empty_dictionary():
    assert mean_reciprocal_rank({}) == 0.0


def test_mrr_all_queries_without_grade_2():
    grades_by_query = {
        "q1": [1, 0, 1],
        "q2": [0, 1, 0],
    }

    assert mean_reciprocal_rank(grades_by_query) == 0.0


def test_mrr_custom_threshold():
    grades_by_query = {
        "q1": [0, 1, 2],  # RR = 1/2
        "q2": [0, 0, 1],  # RR = 1/3
    }

    assert mean_reciprocal_rank(
        grades_by_query, relevant_threshold=1
    ) == pytest.approx(5 / 12)
    
    

def test_mean_precision_at_5():
    grades_by_query = {
        "q1": [1, 2, 1, 1, 2],  # P@5 = 1.0
        "q2": [0, 1, 0, 1, 1],  # P@5 = 0.6
        "q3": [2, 2, 1, 2, 0],  # P@5 = 0.8
    }

    assert mean_precision_at_k(
        grades_by_query, 5
    ) == pytest.approx(0.8)


def test_mean_precision_strict():
    grades_by_query = {
        "q1": [2, 1, 2, 0, 1],  # P@5 = 0.4
        "q2": [2, 2, 1, 2, 0],  # P@5 = 0.6
    }

    assert mean_precision_at_k(
        grades_by_query, 5, relevant_threshold=2
    ) == pytest.approx(0.5)


def test_mean_precision_empty():
    assert mean_precision_at_k({}, 5) == 0.0


def test_mean_precision_invalid_k():
    grades_by_query = {
        "q1": [2, 1, 0],
        "q2": [1, 2, 0, 1, 2],
    }

    with pytest.raises(ValueError):
        mean_precision_at_k(grades_by_query, 5)

def test_dcg_basic():
    assert dcg_at_k([2, 0, 1], 3) == pytest.approx(3.5)


def test_dcg_all_irrelevant():
    assert dcg_at_k([0, 0, 0], 3) == 0.0


def test_dcg_at_2():
    assert dcg_at_k([2, 0, 1], 2) == pytest.approx(3.0)


def test_dcg_first_rank():
    assert dcg_at_k([2], 1) == pytest.approx(3.0)


@pytest.mark.parametrize("k", [0, -1, 4])
def test_dcg_invalid_k(k):
    with pytest.raises(ValueError):
        dcg_at_k([2, 0, 1], k)


def test_idcg_basic():
    grades = [1, 0, 2]
    expected = 3 + 1 / math.log2(3)
    assert idcg_at_k(grades, 3) == pytest.approx(expected)


def test_idcg_uses_entire_list():
    grades = [0, 0, 2, 1]
    expected = 3 + 1 / math.log2(3)
    assert idcg_at_k(grades, 2) == pytest.approx(expected)


def test_idcg_all_irrelevant():
    assert idcg_at_k([0, 0, 0], 3) == 0.0


@pytest.mark.parametrize("k", [0, -1, 4])
def test_idcg_invalid_k(k):
    with pytest.raises(ValueError):
        idcg_at_k([1, 0, 2], k)
        

def test_ndcg_ideal_ranking():
    grades = [2, 1, 0]
    assert ndcg_at_k(grades, 3) == pytest.approx(1.0)


def test_ndcg_non_ideal_ranking():
    grades = [2, 0, 1]
    expected = 3.5 / (3 + 1 / math.log2(3))
    assert ndcg_at_k(grades, 3) == pytest.approx(expected)


def test_ndcg_all_irrelevant():
    assert ndcg_at_k([0, 0, 0], 3) == 0.0


def test_ndcg_relevance_outside_top_k():
    grades = [0, 0, 2, 1]
    assert ndcg_at_k(grades, 2) == 0.0


def test_ndcg_perfect_top_k():
    grades = [2, 2, 0, 1]
    assert ndcg_at_k(grades, 2) == pytest.approx(1.0)


@pytest.mark.parametrize("k", [0, -1, 4])
def test_ndcg_invalid_k(k):
    with pytest.raises(ValueError):
        ndcg_at_k([2, 1, 0], k)
        

def test_mean_ndcg_ideal_rankings():
    grades_by_query = {
        "q1": [2, 1, 0],
        "q2": [2, 2, 1],
    }
    assert mean_ndcg_at_k(grades_by_query, 3) == pytest.approx(1.0)


def test_mean_ndcg_mixed_rankings():
    grades_by_query = {
        "q1": [2, 1, 0],  # NDCG = 1.0
        "q2": [0, 0, 2],  # NDCG = 0.5
    }
    assert mean_ndcg_at_k(grades_by_query, 3) == pytest.approx(0.75)


def test_mean_ndcg_empty():
    assert mean_ndcg_at_k({}, 5) == 0.0


def test_mean_ndcg_invalid_k():
    grades_by_query = {
        "q1": [2, 1, 0],
        "q2": [1, 2, 0],
    }
    with pytest.raises(ValueError):
        mean_ndcg_at_k(grades_by_query, 5)

