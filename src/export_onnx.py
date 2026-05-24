from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from src.config import classes_from_config, thresholds_from_config
from src.models import build_model


def export(config: DictConfig) -> Path:
    if not config.checkpoint_path:
        raise ValueError("checkpoint_path is required")
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    classes = list(checkpoint["classes"]) if isinstance(checkpoint, dict) and "classes" in checkpoint else classes_from_config(config)
    model = build_model(num_classes=len(classes), pretrained=False, architecture=config.model.architecture)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    output_path = Path(config.export.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(*config.export.input_shape)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=config.export.opset_version,
        dynamic_axes=None,
    )
    with torch.no_grad():
        torch_logits = model(dummy).detach().numpy()
    parity = {"validated": False}
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        ort_logits = session.run(["logits"], {"input": dummy.numpy()})[0]
        max_abs_diff = float(abs(torch_logits - ort_logits).max())
        parity = {"validated": max_abs_diff <= config.export.parity_atol, "max_abs_diff": max_abs_diff}
        if not parity["validated"]:
            raise AssertionError(f"ONNX parity failed: max_abs_diff={max_abs_diff}")
    except ImportError:
        parity = {"validated": False, "reason": "onnxruntime is not installed"}
    metadata = {
        "classes": classes,
        "thresholds": dict(zip(classes, thresholds_from_config(config, classes))),
        "input_shape": list(config.export.input_shape),
        "parity": parity,
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


@hydra.main(config_path="../configs", config_name="export_onnx", version_base=None)
def main(config: DictConfig) -> None:
    export(config)


if __name__ == "__main__":
    main()
