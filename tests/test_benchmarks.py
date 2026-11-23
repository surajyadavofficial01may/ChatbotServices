"""Benchmark tests: metric correctness (offline) and the retrieval-quality regression gate (real models).

The gate's floors sit just under the measured production numbers
(docs/rag-improvement-roadmap.md), so a change that makes retrieval worse fails here.
"""

import logging
import math
import os
import warnings

import pytest

from benchmarks import metrics

real_models = pytest.mark.skipif(
    os.getenv("RUN_MODEL_TESTS") != "1", reason="Downloads the real FastEmbed models; set RUN_MODEL_TESTS=1 to enable."
)


# ----------------------------------------------------------------------------
# metrics (offline)
# ----------------------------------------------------------------------------


def test_ranking_metrics_on_a_known_ranking():
    ranked = ["a", "x", "b", "y", "z"]
    relevant = {"a", "b", "c"}
    assert metrics.hit_at_k(ranked, relevant, 5) == 1.0
    assert metrics.recall_at_k(ranked, relevant, 5) == 2 / 3
    assert metrics.precision_at_k(ranked, relevant, 5) == 2 / 5
    assert metrics.reciprocal_rank(ranked, relevant) == 1.0
    expected_dcg = 1 / math.log2(2) + 1 / math.log2(4)
    expected_idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert math.isclose(metrics.ndcg_at_k(ranked, relevant, 5), expected_dcg / expected_idcg)


def test_metrics_when_nothing_relevant_is_retrieved():
    assert metrics.hit_at_k(["x"], {"a"}, 5) == 0.0
    assert metrics.reciprocal_rank(["x", "y"], {"a"}) == 0.0
    assert metrics.precision_at_k([], {"a"}, 5) == 0.0


def test_reciprocal_rank_uses_first_relevant_position():
    assert metrics.reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3


def test_refusal_detection_ignores_partial_answers():
    assert metrics.is_refusal("That information is not available in the portfolio knowledge base.")
    long_partial = (
        "He built EVEE [1] and Orbit [2]. " * 8 + "His salary is not available in the portfolio knowledge base."
    )
    assert not metrics.is_refusal(long_partial)
    assert metrics.is_refusal("I’m sorry, but I can’t comply with that.")


def test_fact_recall_and_citation_parsing():
    assert metrics.fact_recall("CGPA of 8.35 at KIIT", ["8.35", "kiit"]) == 1.0
    assert metrics.fact_recall("CGPA of 8.35", ["8.35", "KIIT"]) == 0.5
    assert metrics.fact_recall("accuracy of 81 % using Jira Fixer", ["81%", "Jira Fixer"]) == 1.0
    assert metrics.cited_indices("A [1] and B [3].") == [1, 3]


# ----------------------------------------------------------------------------
# retrieval-quality regression gate (real models)
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def production_results():
    from benchmarks import evaluate

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    try:
        return evaluate.run_retrieval("production")
    finally:
        logging.disable(logging.NOTSET)


@real_models
def test_overall_retrieval_quality_does_not_regress(production_results):
    assert production_results["hit@5"] >= 0.95
    assert production_results["mrr"] >= 0.90
    assert production_results["ndcg@5"] >= 0.85
    assert production_results["recall@5"] >= 0.85


@real_models
def test_exact_identifiers_and_typos_stay_perfect(production_results):
    assert production_results["by_type"]["exact"]["hit@5"] == 1.0
    assert production_results["by_type"]["typo"]["hit@5"] == 1.0


@real_models
def test_list_questions_keep_their_routing_gain(production_results):
    assert production_results["by_type"]["list"]["hit@5"] == 1.0
    assert production_results["by_type"]["list"]["recall@5"] >= 0.6


@real_models
def test_gate_never_rejects_an_answerable_question(production_results):
    assert production_results["positive_false_rejection_rate"] == 0.0
