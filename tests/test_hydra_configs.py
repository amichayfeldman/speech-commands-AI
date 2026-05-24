from hydra import compose, initialize_config_dir


def test_hydra_configs_compose(project_root):
    config_dir = str(project_root / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        train_cfg = compose(config_name="train", overrides=["experiment=robot_commands"])
        eval_cfg = compose(config_name="evaluate", overrides=["checkpoint_path=/tmp/model.pt"])
        export_cfg = compose(config_name="export_onnx", overrides=["checkpoint_path=/tmp/model.pt", "export.output_path=/tmp/model.onnx"])
    assert train_cfg.data_source.train_table == "train"
    assert eval_cfg.checkpoint_path == "/tmp/model.pt"
    assert export_cfg.export.input_shape == [1, 1, 224, 224]
    assert train_cfg.dataset.label_policy == "oov_background_as_empty_target"

