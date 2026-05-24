from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.config import classes_from_config, thresholds_from_config
from src.evaluation.metrics import evaluate_predictions, recommend_thresholds, write_metrics
from src.models import build_model
from src.train import build_dataset, collate_batch


def classes_from_checkpoint(config: DictConfig, checkpoint: object) -> list[str]:
    if isinstance(checkpoint, dict) and "classes" in checkpoint:
        return list(checkpoint["classes"])
    return classes_from_config(config)


def evaluate(config: DictConfig) -> dict[str, object]:
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    classes = classes_from_checkpoint(config, checkpoint)
    thresholds = thresholds_from_config(config, classes)
    dataset = build_dataset(config, "test", classes=classes)
    loader = DataLoader(
        dataset,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        num_workers=config.trainer.num_workers,
        collate_fn=collate_batch,
    )
    model = build_model(num_classes=len(classes), pretrained=False, architecture=config.model.architecture)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    probs = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input"])
            probs.append(torch.sigmoid(logits).numpy())
            targets.append(batch["target"].numpy())
    probabilities = np.concatenate(probs, axis=0)
    target_array = np.concatenate(targets, axis=0)
    metrics = evaluate_predictions(target_array, probabilities, classes, thresholds)
    recommendations = recommend_thresholds(target_array, probabilities, classes)
    write_metrics(config.paths.evaluation_dir, metrics, recommendations)
    return metrics


@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(config: DictConfig) -> None:
    if not config.checkpoint_path:
        raise ValueError("checkpoint_path is required")
    evaluate(config)


if __name__ == "__main__":
    main()
