"""Unit tests for CLIP-Catalog evaluation metrics and commercial onboarding bridge.

16 pure-NumPy / Python tests verifying mathematical invariants, edge cases,
and commercial decision guardrails without GPU dependencies.
"""

import os
import sys
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.eval import (
    recall_at_k,
    mean_reciprocal_rank,
    compute_retrieval_metrics,
    compute_zero_shot_accuracy
)
from src.onboarding_bridge import OnboardingBridge


# -----------------------------------------------------------------------------
# Metric Correctness & Ranking Invariant Tests (Tests 1 - 7)
# -----------------------------------------------------------------------------

def test_recall_at_1_perfect_ranking():
    """Test 1: Recall@1 is exactly 1.0 when every query hit ranks at position 1."""
    ranks = [1, 1, 1, 1, 1]
    r1 = recall_at_k(ranks, k=1)
    assert np.isclose(r1, 1.0), f"Expected Recall@1 == 1.0, got {r1}"


def test_recall_at_5_cutoff_boundary():
    """Test 2: Targets ranking at rank 5 are included in Recall@5; rank 6 excluded."""
    ranks = [1, 3, 5, 6, None]
    r5 = recall_at_k(ranks, k=5)
    # Hits at 1, 3, 5 -> 3 hits out of 5 queries = 0.6
    assert np.isclose(r5, 0.6), f"Expected Recall@5 == 0.6, got {r5}"


def test_recall_at_k_empty_queries():
    """Test 3: Empty query list yields 0.0 recall without error."""
    assert recall_at_k([], k=5) == 0.0


def test_mrr_first_position_hits():
    """Test 4: MRR is 1.0 when all queries hit at rank 1."""
    ranks = [1, 1, 1]
    mrr = mean_reciprocal_rank(ranks)
    assert np.isclose(mrr, 1.0), f"Expected MRR == 1.0, got {mrr}"


def test_mrr_decay_gradient():
    """Test 5: MRR accurately reflects inverse rank decay 1/1, 1/2, 1/4."""
    ranks = [1, 2, 4]
    # (1/1 + 1/2 + 1/4) / 3 = (1 + 0.5 + 0.25) / 3 = 1.75 / 3 ≈ 0.58333
    expected_mrr = (1.0 + 0.5 + 0.25) / 3.0
    actual_mrr = mean_reciprocal_rank(ranks)
    assert np.isclose(actual_mrr, expected_mrr, atol=1e-5), f"Expected {expected_mrr}, got {actual_mrr}"


def test_mrr_zero_hits():
    """Test 6: MRR is 0.0 when none of the queries find targets."""
    ranks = [None, None, 0]
    mrr = mean_reciprocal_rank(ranks)
    assert np.isclose(mrr, 0.0), f"Expected MRR == 0.0, got {mrr}"


def test_compute_retrieval_metrics_consistency():
    """Test 7: Multi-metric helper computes consistent recall and MRR dictionary."""
    gts = ["SKU_1", "SKU_2", "SKU_3"]
    retrieved = [
        ["SKU_1", "SKU_4", "SKU_5"],        # rank 1
        ["SKU_X", "SKU_Y", "SKU_2"],        # rank 3
        ["SKU_A", "SKU_B", "SKU_C", "SKU_3"] # rank 4
    ]
    res = compute_retrieval_metrics(gts, retrieved, ks=(1, 3, 5))
    assert np.isclose(res["recall@1"], 1.0 / 3.0)
    assert np.isclose(res["recall@3"], 2.0 / 3.0)
    assert np.isclose(res["recall@5"], 3.0 / 3.0)
    expected_mrr = (1.0 + 1.0 / 3.0 + 1.0 / 4.0) / 3.0
    assert np.isclose(res["mrr"], expected_mrr, atol=1e-5)


# -----------------------------------------------------------------------------
# Zero-Shot Classification Accuracy Tests (Tests 8 - 10)
# -----------------------------------------------------------------------------

def test_zero_shot_top1_accuracy_perfect():
    """Test 8: Perfect probability allocation yields 100% top-1 accuracy."""
    y_true = ["Dairy", "Produce", "Snacks"]
    categories = ["Dairy", "Produce", "Snacks"]
    probs = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7]
    ])
    accs = compute_zero_shot_accuracy(y_true, probs, categories)
    assert np.isclose(accs["top1_accuracy"], 1.0)


def test_zero_shot_top1_accuracy_imperfect():
    """Test 9: 1 mistake out of 3 yields 66.67% accuracy."""
    y_true = ["Dairy", "Produce", "Snacks"]
    categories = ["Dairy", "Produce", "Snacks"]
    probs = np.array([
        [0.9, 0.05, 0.05],  # Correct (Dairy)
        [0.8, 0.1, 0.1],    # Wrong (predicted Dairy instead of Produce)
        [0.1, 0.2, 0.7]     # Correct (Snacks)
    ])
    accs = compute_zero_shot_accuracy(y_true, probs, categories)
    assert np.isclose(accs["top1_accuracy"], 2.0 / 3.0, atol=1e-5)


def test_zero_shot_topk_accuracy_superset():
    """Test 10: Top-K accuracy is always greater than or equal to Top-1 accuracy."""
    y_true = ["Dairy", "Produce", "Snacks", "Beverages"]
    categories = ["Dairy", "Produce", "Snacks", "Beverages"]
    probs = np.array([
        [0.5, 0.3, 0.1, 0.1],
        [0.4, 0.35, 0.15, 0.1],
        [0.2, 0.4, 0.3, 0.1],
        [0.1, 0.2, 0.3, 0.4]
    ])
    accs = compute_zero_shot_accuracy(y_true, probs, categories, top_k=2)
    assert accs["top2_accuracy"] >= accs["top1_accuracy"]


# -----------------------------------------------------------------------------
# OnboardingBridge Business Guardrail Tests (Tests 11 - 16)
# -----------------------------------------------------------------------------

def test_bridge_auto_tag_high_confidence():
    """Test 11: High confidence and clear margin trigger auto_tag."""
    bridge = OnboardingBridge(auto_tag_confidence=0.70, min_top1_margin=0.15)
    scores = {"Fresh Produce": 0.88, "Dairy": 0.07, "Snacks": 0.05}
    decision = bridge.decide(scores, {"title": "Fresh Alphonso Mango 1kg"})
    assert decision["action"] == "auto_tag"
    assert decision["category"] == "Fresh Produce"
    assert np.isclose(decision["confidence"], 0.88)


def test_bridge_human_review_moderate_confidence():
    """Test 12: Moderate confidence between 0.40 and 0.70 routes to human_review."""
    bridge = OnboardingBridge(auto_tag_confidence=0.70, review_queue_confidence=0.40, min_top1_margin=0.15)
    scores = {"Fresh Produce": 0.58, "Dairy": 0.25, "Snacks": 0.17}
    decision = bridge.decide(scores, {"title": "Organic Sprouted Grain"})
    assert decision["action"] == "human_review"
    assert decision["category"] == "Fresh Produce"


def test_bridge_reject_below_floor():
    """Test 13: Scores below review floor (0.40) are rejected."""
    bridge = OnboardingBridge(review_queue_confidence=0.40)
    scores = {"Fresh Produce": 0.31, "Dairy": 0.29, "Snacks": 0.25, "Beverages": 0.15}
    decision = bridge.decide(scores, {"title": "Ambiguous Mystery Item"})
    assert decision["action"] == "reject"


def test_bridge_reject_forbidden_category():
    """Test 14: Forbidden categories are rejected even when high confidence."""
    bridge = OnboardingBridge(
        auto_tag_confidence=0.70,
        forbidden_categories=["alcohol", "tobacco"]
    )
    scores = {"Alcohol": 0.96, "Beverages": 0.04}
    decision = bridge.decide(scores, {"title": "Craft Malt Beverage"})
    assert decision["action"] == "reject"
    assert any("forbidden" in log.lower() for log in decision["audit"])


def test_bridge_escalate_on_tight_margin():
    """Test 15: High confidence with tight margin (< min_top1_margin) escalates to Category Head."""
    bridge = OnboardingBridge(
        auto_tag_confidence=0.70,
        min_top1_margin=0.15
    )
    # Top-1 is 0.72, top-2 is 0.65 -> margin = 0.07 < 0.15
    scores = {"Dairy & Breakfast": 0.72, "Fresh Produce": 0.65}
    decision = bridge.decide(scores, {"title": "Vegan Coconut Yogurt"})
    assert decision["action"] == "escalate"
    assert any("escalated" in log.lower() for log in decision["audit"])


def test_bridge_batch_decide_consistency():
    """Test 16: Batch decision method matches individual decisions."""
    bridge = OnboardingBridge()
    batch_scores = [
        {"Produce": 0.90, "Dairy": 0.10},
        {"Dairy": 0.55, "Produce": 0.30},
        {"Snacks": 0.20, "Dairy": 0.20}
    ]
    batch_res = bridge.batch_decide(batch_scores)
    assert len(batch_res) == 3
    assert batch_res[0]["action"] == "auto_tag"
    assert batch_res[1]["action"] == "human_review"
    assert batch_res[2]["action"] == "reject"


# -----------------------------------------------------------------------------
# Test Runner
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_recall_at_1_perfect_ranking,
        test_recall_at_5_cutoff_boundary,
        test_recall_at_k_empty_queries,
        test_mrr_first_position_hits,
        test_mrr_decay_gradient,
        test_mrr_zero_hits,
        test_compute_retrieval_metrics_consistency,
        test_zero_shot_top1_accuracy_perfect,
        test_zero_shot_top1_accuracy_imperfect,
        test_zero_shot_topk_accuracy_superset,
        test_bridge_auto_tag_high_confidence,
        test_bridge_human_review_moderate_confidence,
        test_bridge_reject_below_floor,
        test_bridge_reject_forbidden_category,
        test_bridge_escalate_on_tight_margin,
        test_bridge_batch_decide_consistency,
    ]

    print(f"Running {len(tests)} CLIP-Catalog unit tests...\n")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  [PASS] {t.__name__}")
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")

    print(f"\n=======================================================")
    print(f" Results: {passed}/{len(tests)} tests passed successfully.")
    print(f"=======================================================")
    if passed < len(tests):
        sys.exit(1)

