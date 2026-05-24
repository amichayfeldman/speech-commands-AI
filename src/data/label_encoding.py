from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


def parse_labels(value: object) -> list[str]:
    """Parse labels stored as a list, JSON list, stringified list, or scalar string."""
    if value is None:
        return []
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return parse_labels(value.tolist())
    if isinstance(value, (list, tuple, set)):
        labels: list[str] = []
        for item in value:
            labels.extend(parse_labels(item))
        return labels
    try:
        import pandas as pd

        if pd.isna(value):
            return []
    except Exception:
        pass
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
            except Exception:
                continue
            if isinstance(parsed, (list, tuple, set)):
                return [str(item) for item in parsed if str(item)]
            if parsed in (None, ""):
                return []
            return [str(parsed)]
        return [stripped]
    return [str(value)]


@dataclass(frozen=True)
class LabelEncoder:
    classes: Sequence[str]
    unknown_policy: str = "error"

    def __post_init__(self) -> None:
        if self.unknown_policy not in {"error", "ignore"}:
            raise ValueError("unknown_policy must be 'error' or 'ignore'")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("classes must be unique")

    @property
    def class_to_index(self) -> dict[str, int]:
        return {label: idx for idx, label in enumerate(self.classes)}

    def encode(self, labels: object) -> torch.Tensor:
        parsed = parse_labels(labels)
        target = torch.zeros(len(self.classes), dtype=torch.float32)
        class_to_index = self.class_to_index
        unknown = [label for label in parsed if label not in class_to_index]
        if unknown and self.unknown_policy == "error":
            raise KeyError(f"Unknown labels: {unknown}")
        for label in parsed:
            if label in class_to_index:
                target[class_to_index[label]] = 1.0
        return target

    def decode(self, target: Iterable[float]) -> list[str]:
        return [label for label, active in zip(self.classes, target) if float(active) > 0.0]
