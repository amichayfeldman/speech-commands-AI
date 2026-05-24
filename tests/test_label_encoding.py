import pytest
import numpy as np
import torch

from src.data.label_encoding import LabelEncoder, parse_labels


def test_parse_labels_variants():
    assert parse_labels(["move_forward", "stop"]) == ["move_forward", "stop"]
    assert parse_labels('["move_forward", "stop"]') == ["move_forward", "stop"]
    assert parse_labels(np.array(["move_forward", "stop"], dtype=object)) == ["move_forward", "stop"]
    assert parse_labels("[]") == []
    assert parse_labels("") == []
    assert parse_labels("stop") == ["stop"]


def test_label_encoder_multi_hot_empty_and_unknown():
    encoder = LabelEncoder(["move_forward", "stop"])
    assert torch.equal(encoder.encode("move_forward"), torch.tensor([1.0, 0.0]))
    assert torch.equal(encoder.encode(["move_forward", "stop"]), torch.tensor([1.0, 1.0]))
    assert torch.equal(encoder.encode([]), torch.tensor([0.0, 0.0]))
    with pytest.raises(KeyError):
        encoder.encode(["unknown"])
    ignore_encoder = LabelEncoder(["move_forward"], unknown_policy="ignore")
    assert torch.equal(ignore_encoder.encode(["unknown"]), torch.tensor([0.0]))
