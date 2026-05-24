import json

import numpy as np

from src.evaluation.metrics import evaluate_predictions, recommend_thresholds, write_metrics


def test_evaluation_metrics_and_threshold_outputs(tmp_path):
    targets = np.array([[1, 0], [0, 0], [0, 1]])
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.9]])
    metrics = evaluate_predictions(targets, probabilities, ["forward", "stop"], [0.5, 0.5])
    recommendations = recommend_thresholds(targets, probabilities, ["forward", "stop"])
    write_metrics(tmp_path, metrics, recommendations)
    assert metrics["empty_target_false_activation_rate"] == 1.0
    assert metrics["exact_match_rate"] == 2 / 3
    assert set(recommendations) == {"forward", "stop"}
    assert json.loads((tmp_path / "metrics.json").read_text())["macro_f1"] == metrics["macro_f1"]
    assert (tmp_path / "threshold_recommendations.csv").exists()


def test_threshold_recommendation_for_absent_class_defaults_to_safe_midpoint():
    targets = np.array([[1, 0], [0, 0], [1, 0]])
    probabilities = np.array([[0.9, 0.9], [0.2, 0.8], [0.8, 0.7]])
    recommendations = recommend_thresholds(targets, probabilities, ["present", "absent"])
    assert recommendations["absent"] == 0.5
