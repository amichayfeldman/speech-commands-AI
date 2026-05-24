from __future__ import annotations

import torch


class CompactSpeechCNN(torch.nn.Module):
    def __init__(self, num_classes: int, width: int = 32) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, width, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(width),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(width * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(width * 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Linear(width * 4, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.features(inputs)
        return self.classifier(torch.flatten(x, 1))


def build_model(num_classes: int, pretrained: bool = False, architecture: str = "mobilenet_v3_small") -> torch.nn.Module:
    """Build a raw-logit multi-label model.

    Uses torchvision MobileNetV3 when available and falls back to a compact CNN in minimal
    environments. Both return logits only.
    """
    if architecture != "mobilenet_v3_small":
        raise ValueError("Only mobilenet_v3_small is supported")
    try:
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        first = model.features[0][0]
        replacement = torch.nn.Conv2d(1, first.out_channels, first.kernel_size, first.stride, first.padding, bias=first.bias is not None)
        with torch.no_grad():
            replacement.weight.copy_(first.weight.mean(dim=1, keepdim=True))
            if first.bias is not None:
                replacement.bias.copy_(first.bias)
        model.features[0][0] = replacement
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    except Exception:
        return CompactSpeechCNN(num_classes=num_classes)

