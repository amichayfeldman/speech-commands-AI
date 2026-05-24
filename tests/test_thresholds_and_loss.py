import torch

from src.losses import MultiLabelBCELoss
from src.thresholds import ThresholdDecoder


def test_bce_loss_matches_torch():
    logits = torch.tensor([[0.0, 2.0], [-1.0, 1.0]])
    targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    expected = torch.nn.BCEWithLogitsLoss()(logits, targets)
    actual = MultiLabelBCELoss()(logits, targets)
    assert torch.allclose(actual, expected)


def test_threshold_decoder_returns_zero_one_or_many_labels():
    decoder = ThresholdDecoder(["a", "b"], thresholds=[0.5, 0.8])
    logits = torch.tensor([[0.0, 2.0], [2.0, 2.0], [-5.0, -5.0]])
    assert decoder.decode(logits) == [["a", "b"], ["a", "b"], []]

