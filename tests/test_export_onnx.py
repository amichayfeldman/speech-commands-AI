import pytest
import torch
from hydra import compose, initialize_config_dir

from src.export_onnx import export
from src.models import build_model


def test_export_writes_onnx_and_metadata(tmp_path, project_root):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    checkpoint_path = tmp_path / "model.pt"
    model = build_model(num_classes=5, pretrained=False)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    with initialize_config_dir(config_dir=str(project_root / "configs"), version_base=None):
        cfg = compose(
            config_name="export_onnx",
            overrides=[
                f"checkpoint_path={checkpoint_path}",
                f"export.output_path={tmp_path / 'model.onnx'}",
                "export.input_shape=[1,1,224,224]",
            ],
        )
    output = export(cfg)
    assert output.exists()
    assert output.with_suffix(".metadata.json").exists()

