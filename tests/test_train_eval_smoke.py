import math
import wave
from pathlib import Path

import duckdb
import numpy as np
from hydra import compose, initialize_config_dir

from src.evaluate import evaluate
from src.models.mobilenet_v3 import CompactSpeechCNN
from src.train import train


def _write_wav(path: Path, frequency: float, sample_rate: int = 16000) -> None:
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = (0.1 * np.sin(2 * math.pi * frequency * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio.tobytes())


def test_tiny_train_eval_smoke(tmp_path, project_root, monkeypatch):
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    db_path = tmp_path / "data.duckdb"
    _write_wav(wav_a, 440)
    _write_wav(wav_b, 660)
    with duckdb.connect(str(db_path)) as con:
        for table in ("train", "val", "test"):
            con.execute(f"CREATE TABLE {table} (audio_path VARCHAR, labels VARCHAR, text VARCHAR, is_background BOOLEAN)")
            con.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?)", [str(wav_a), '["move_forward"]', "forward", False])
            con.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?)", [str(wav_b), "[]", "background", True])

    def tiny_model(num_classes, pretrained=False, architecture="mobilenet_v3_small"):
        del pretrained, architecture
        return CompactSpeechCNN(num_classes=num_classes, width=2)

    monkeypatch.setattr("src.train.build_model", tiny_model)
    monkeypatch.setattr("src.evaluate.build_model", tiny_model)
    overrides = [
        f"data_source.database_path={db_path}",
        f"paths.output_dir={tmp_path / 'checkpoints'}",
        f"paths.evaluation_dir={tmp_path / 'evaluation'}",
        "trainer.batch_size=1",
        "trainer.epochs=1",
        "dataset.image_size=32",
    ]
    with initialize_config_dir(config_dir=str(project_root / "configs"), version_base=None):
        train_cfg = compose(config_name="train", overrides=overrides)
    checkpoint = train(train_cfg)
    with initialize_config_dir(config_dir=str(project_root / "configs"), version_base=None):
        eval_cfg = compose(config_name="evaluate", overrides=overrides + [f"checkpoint_path={checkpoint}"])
    evaluate(eval_cfg)
    assert checkpoint.exists()
    assert (tmp_path / "evaluation" / "metrics.json").exists()
    assert (tmp_path / "evaluation" / "threshold_recommendations.csv").exists()

