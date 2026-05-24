from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ThresholdDecoder:
    classes: Sequence[str]
    thresholds: Sequence[float] | float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.thresholds, (float, int)):
            object.__setattr__(self, "_thresholds", [float(self.thresholds)] * len(self.classes))
        else:
            values = [float(value) for value in self.thresholds]
            if len(values) != len(self.classes):
                raise ValueError("thresholds length must match classes length")
            object.__setattr__(self, "_thresholds", values)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def decode(self, logits: torch.Tensor) -> list[list[str]]:
        probs = self.probabilities(logits)
        if probs.ndim == 1:
            probs = probs.unsqueeze(0)
        thresholds = torch.tensor(self._thresholds, dtype=probs.dtype, device=probs.device)
        active = probs >= thresholds
        return [[label for label, is_active in zip(self.classes, row.tolist()) if is_active] for row in active]

