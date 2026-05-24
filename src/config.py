from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def classes_from_config(config: DictConfig) -> list[str]:
    return list(OmegaConf.to_container(config.experiment.classes, resolve=True))


def thresholds_from_config(config: DictConfig, classes: list[str]) -> list[float]:
    value = config.thresholds.values
    if isinstance(value, (float, int)):
        return [float(value)] * len(classes)
    values = list(value)
    if len(values) != len(classes):
        raise ValueError("threshold count must match class count")
    return [float(item) for item in values]

