from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _safe_div(num: float, denom: float) -> float:
    return float(num / denom) if denom else 0.0


def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = float(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = float(np.logical_and(y_true == 1, y_pred == 0).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, scores))
    except Exception:
        return None


def evaluate_predictions(
    targets: np.ndarray,
    probabilities: np.ndarray,
    classes: Sequence[str],
    thresholds: Sequence[float],
) -> dict[str, object]:
    y_true = np.asarray(targets).astype(int)
    probs = np.asarray(probabilities).astype(float)
    threshold_arr = np.asarray(thresholds).astype(float)
    y_pred = (probs >= threshold_arr).astype(int)
    micro_p, micro_r, micro_f1 = _precision_recall_f1(y_true, y_pred)
    per_class = {}
    per_class_f1 = []
    for idx, label in enumerate(classes):
        p, r, f1 = _precision_recall_f1(y_true[:, idx], y_pred[:, idx])
        per_class_f1.append(f1)
        per_class[label] = {
            "auroc": _auroc(y_true[:, idx], probs[:, idx]),
            "precision": p,
            "recall": r,
            "f1": f1,
            "threshold": float(threshold_arr[idx]),
        }
    empty_rows = y_true.sum(axis=1) == 0
    false_activation_rate = float(y_pred[empty_rows].any(axis=1).mean()) if empty_rows.any() else 0.0
    return {
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": float(np.mean(per_class_f1)) if per_class_f1 else 0.0,
        "exact_match_rate": float((y_true == y_pred).all(axis=1).mean()) if len(y_true) else 0.0,
        "empty_target_false_activation_rate": false_activation_rate,
        "per_class": per_class,
    }


def recommend_thresholds(targets: np.ndarray, probabilities: np.ndarray, classes: Sequence[str]) -> dict[str, float]:
    recommendations = {}
    grid = np.linspace(0.05, 0.95, 19)
    for idx, label in enumerate(classes):
        if targets[:, idx].sum() == 0:
            recommendations[label] = 0.5
            continue
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in grid:
            _, _, f1 = _precision_recall_f1(targets[:, idx].astype(int), (probabilities[:, idx] >= threshold).astype(int))
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
        recommendations[label] = best_threshold
    return recommendations


def write_metrics(output_dir: str | Path, metrics: dict[str, object], recommendations: dict[str, float]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output / "threshold_recommendations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "threshold"])
        writer.writeheader()
        for label, threshold in recommendations.items():
            writer.writerow({"class": label, "threshold": threshold})
