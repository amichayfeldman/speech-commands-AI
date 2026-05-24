import math
import wave
from pathlib import Path

import duckdb
import numpy as np

from src.data import DuckDBSpeechCommandsDataset


def _write_wav(path: Path, sample_rate: int = 16000) -> None:
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = (0.1 * np.sin(2 * math.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio.tobytes())


def test_duckdb_dataset_loads_rows_and_encodes_targets(tmp_path):
    wav_path = tmp_path / "sample.wav"
    db_path = tmp_path / "data.duckdb"
    _write_wav(wav_path)
    with duckdb.connect(str(db_path)) as con:
        con.execute("CREATE TABLE train (audio_path VARCHAR, labels VARCHAR, text VARCHAR, is_background BOOLEAN)")
        con.execute("INSERT INTO train VALUES (?, ?, ?, ?)", [str(wav_path), '["move_forward", "stop"]', "go stop", False])
        con.execute("INSERT INTO train VALUES (?, ?, ?, ?)", [str(wav_path), '["move_forward"]', "noise", True])
        con.execute("INSERT INTO train VALUES (?, ?, ?, ?)", [str(wav_path), '["unknown"]', "oov", None])

    dataset = DuckDBSpeechCommandsDataset(
        database_path=str(db_path),
        table_name="train",
        audio_column="audio_path",
        labels_column="labels",
        text_column="text",
        background_flag_column="is_background",
        classes=["move_forward", "stop"],
        image_size=64,
    )

    sample = dataset[0]
    assert sample["input"].shape == (1, 64, 64)
    assert sample["target"].tolist() == [1.0, 1.0]
    assert sample["labels"] == ["move_forward", "stop"]
    assert sample["metadata"]["text"] == "go stop"
    assert dataset[1]["target"].tolist() == [0.0, 0.0]
    assert dataset[2]["target"].tolist() == [0.0, 0.0]
