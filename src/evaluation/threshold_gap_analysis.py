from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.config import classes_from_config
from src.data import DuckDBSpeechCommandsDataset
from src.evaluation.metrics import _auroc, _precision_recall_f1
from src.models import build_model
from src.train import collate_batch


@dataclass(frozen=True)
class ROCCurve:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auroc: float | None


@dataclass(frozen=True)
class InferenceResult:
    samples: pd.DataFrame
    targets: np.ndarray
    probabilities: np.ndarray
    classes: list[str]


def download_clearml_model(model_id: str, output_dir: str | Path | None = None) -> Path:
    from clearml import Model

    model = Model(model_id=model_id)
    local_path = model.get_local_copy(extract_archive=True, force_download=False, local_path=str(output_dir) if output_dir else None)
    path = Path(local_path)
    return find_checkpoint_file(path) if path.is_dir() else path


def download_clearml_dataset(dataset_id: str, output_dir: str | Path | None = None) -> Path:
    from clearml import Dataset

    del output_dir
    dataset = Dataset.get(dataset_id=dataset_id)
    local_path = dataset.get_local_copy()
    return Path(local_path)


def find_checkpoint_file(model_path: str | Path) -> Path:
    path = Path(model_path)
    if path.is_file():
        return path
    candidates = sorted([*path.rglob("*.pt"), *path.rglob("*.pth"), *path.rglob("*.ckpt")])
    if not candidates:
        raise FileNotFoundError(f"No PyTorch checkpoint found under {path}")
    return candidates[0]


def find_duckdb_file(dataset_dir: str | Path) -> Path:
    root = Path(dataset_dir)
    candidates = sorted([*root.rglob("*.duckdb"), *root.rglob("*.db")])
    if not candidates:
        raise FileNotFoundError(f"No DuckDB file found under {root}")
    return candidates[0]


def class_roc_curve(y_true: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> ROCCurve:
    target = np.asarray(y_true).astype(int)
    probability = np.asarray(scores).astype(float)
    if len(np.unique(target)) < 2:
        tpr_end = 1.0 if target.size and int(target[0]) == 1 else 0.0
        return ROCCurve(
            fpr=np.array([0.0, 1.0]),
            tpr=np.array([0.0, tpr_end]),
            thresholds=np.array([1.0, 0.0]),
            auroc=None,
        )
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(target, probability)
    finite = np.where(np.isinf(thresholds), 1.0, np.clip(thresholds, 0.0, 1.0))
    return ROCCurve(fpr=fpr.astype(float), tpr=tpr.astype(float), thresholds=finite.astype(float), auroc=_auroc(target, probability))


def threshold_markers(curve: ROCCurve, count: int = 11) -> list[dict[str, float | int]]:
    marker_thresholds = np.linspace(0.0, 1.0, count)
    markers = []
    for threshold in marker_thresholds:
        index = int(np.abs(curve.thresholds - threshold).argmin())
        markers.append(
            {
                "threshold": float(round(threshold, 6)),
                "fpr": float(curve.fpr[index]),
                "tpr": float(curve.tpr[index]),
                "roc_index": index,
            }
        )
    return markers


def confusion_labels(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_index: int,
    threshold: float,
) -> np.ndarray:
    y_true = np.asarray(targets)[:, class_index].astype(int)
    y_pred = (np.asarray(probabilities)[:, class_index] >= threshold).astype(int)
    labels = np.empty(y_true.shape, dtype=object)
    labels[(y_true == 1) & (y_pred == 1)] = "TP"
    labels[(y_true == 0) & (y_pred == 1)] = "FP"
    labels[(y_true == 0) & (y_pred == 0)] = "TN"
    labels[(y_true == 1) & (y_pred == 0)] = "FN"
    return labels


def class_metrics_at_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_index: int,
    threshold: float,
) -> dict[str, float | int | None]:
    y_true = np.asarray(targets)[:, class_index].astype(int)
    scores = np.asarray(probabilities)[:, class_index].astype(float)
    y_pred = (scores >= threshold).astype(int)
    precision, recall, f1 = _precision_recall_f1(y_true, y_pred)
    labels = confusion_labels(targets, probabilities, class_index=class_index, threshold=threshold)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": _auroc(y_true, scores),
        "tp": int((labels == "TP").sum()),
        "fp": int((labels == "FP").sum()),
        "tn": int((labels == "TN").sum()),
        "fn": int((labels == "FN").sum()),
    }


def samples_for_confusion(
    result: InferenceResult,
    *,
    class_index: int,
    threshold: float,
    confusion: str,
) -> pd.DataFrame:
    labels = confusion_labels(result.targets, result.probabilities, class_index=class_index, threshold=threshold)
    samples = result.samples.copy()
    samples["confusion"] = labels
    samples["probability"] = result.probabilities[:, class_index]
    samples["target"] = result.targets[:, class_index]
    return samples[samples["confusion"] == confusion].reset_index(drop=True)


def run_inference(
    *,
    checkpoint_path: str | Path,
    database_path: str | Path,
    table_name: str = "test",
    batch_size: int | None = None,
    device: str | None = None,
) -> InferenceResult:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = _config_from_checkpoint(checkpoint)
    classes = _classes_from_checkpoint(config, checkpoint)
    dataset = _dataset_from_config(config, Path(database_path), table_name, classes)
    trainer = config.get("trainer", {})
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size or trainer.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(trainer.get("num_workers", 0)),
        collate_fn=collate_batch,
    )
    requested_device = device or trainer.get("device", "cpu")
    torch_device = torch.device(requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu")
    model_cfg = config.get("model", {})
    model = build_model(num_classes=len(classes), pretrained=False, architecture=model_cfg.get("architecture", "mobilenet_v3_small"))
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(torch_device)
    model.eval()

    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input"].to(torch_device))
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            batch_targets = batch["target"].cpu().numpy()
            probs.append(batch_probs)
            targets.append(batch_targets)
            for idx, path in enumerate(batch["path"]):
                rows.append(
                    {
                        "sample_index": offset + idx,
                        "audio_path": path,
                        "metadata": dict(batch["metadata"][idx]),
                        "targets": batch_targets[idx],
                        "probabilities": batch_probs[idx],
                        "classes": classes,
                    }
                )
            offset += len(batch["path"])
    probabilities = np.concatenate(probs, axis=0) if probs else np.empty((0, len(classes)), dtype=float)
    target_array = np.concatenate(targets, axis=0) if targets else np.empty((0, len(classes)), dtype=float)
    return InferenceResult(samples=pd.DataFrame(rows), targets=target_array, probabilities=probabilities, classes=classes)


def _config_from_checkpoint(checkpoint: object) -> DictConfig:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        return OmegaConf.create(checkpoint["config"])
    return OmegaConf.create(
        {
            "data_source": {
                "audio_column": "audio_path",
                "labels_column": "labels",
                "text_column": "text",
                "background_flag_column": "is_background",
                "metadata_columns": ["speaker_id", "session_id"],
            },
            "dataset": {
                "unknown_label_policy": "error",
                "sample_rate": 16000,
                "duration_seconds": 1.0,
                "n_fft": 400,
                "hop_length": 160,
                "image_size": 224,
            },
            "model": {"architecture": "mobilenet_v3_small"},
            "trainer": {"batch_size": 32, "num_workers": 0, "device": "cpu"},
            "experiment": {"classes": []},
        }
    )


def _classes_from_checkpoint(config: DictConfig, checkpoint: object) -> list[str]:
    if isinstance(checkpoint, dict) and "classes" in checkpoint:
        return list(checkpoint["classes"])
    return classes_from_config(config)


def _dataset_from_config(config: DictConfig, database_path: Path, table_name: str, classes: list[str]) -> DuckDBSpeechCommandsDataset:
    data_source = config.get("data_source", {})
    dataset = config.get("dataset", {})
    return DuckDBSpeechCommandsDataset(
        database_path=str(database_path),
        table_name=table_name,
        audio_column=data_source.get("audio_column", "audio_path"),
        labels_column=data_source.get("labels_column", "labels"),
        text_column=data_source.get("text_column"),
        background_flag_column=data_source.get("background_flag_column"),
        metadata_columns=list(data_source.get("metadata_columns", [])),
        classes=classes,
        unknown_label_policy=dataset.get("unknown_label_policy", "error"),
        sample_rate=int(dataset.get("sample_rate", 16000)),
        duration_seconds=float(dataset.get("duration_seconds", 1.0)),
        n_fft=int(dataset.get("n_fft", 400)),
        hop_length=int(dataset.get("hop_length", 160)),
        image_size=int(dataset.get("image_size", 224)),
    )
