import math
import wave
from pathlib import Path

import duckdb
import numpy as np
import torch

from src.models.mobilenet_v3 import CompactSpeechCNN


def _write_wav(path: Path, frequency: float, sample_rate: int = 16000) -> None:
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = (0.1 * np.sin(2 * math.pi * frequency * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio.tobytes())


def test_confusion_labels_match_binary_truth_table():
    from src.evaluation.threshold_gap_analysis import confusion_labels

    targets = np.array([[1], [0], [0], [1]])
    probabilities = np.array([[0.8], [0.7], [0.2], [0.1]])

    assert confusion_labels(targets, probabilities, class_index=0, threshold=0.5).tolist() == ["TP", "FP", "TN", "FN"]


def test_roc_curve_handles_single_value_targets():
    from src.evaluation.threshold_gap_analysis import class_roc_curve

    curve = class_roc_curve(np.array([0, 0, 0]), np.array([0.1, 0.3, 0.2]))

    assert curve.auroc is None
    assert curve.fpr.tolist() == [0.0, 1.0]
    assert curve.tpr.tolist() == [0.0, 0.0]
    assert curve.thresholds.tolist() == [1.0, 0.0]


def test_threshold_markers_use_valid_roc_points():
    from src.evaluation.threshold_gap_analysis import class_roc_curve, threshold_markers

    curve = class_roc_curve(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
    markers = threshold_markers(curve, count=5)
    point_set = set(zip(curve.fpr.tolist(), curve.tpr.tolist()))

    assert [marker["threshold"] for marker in markers] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert all((marker["fpr"], marker["tpr"]) in point_set for marker in markers)
    assert all(0 <= marker["roc_index"] < len(curve.thresholds) for marker in markers)


def test_find_duckdb_file_discovers_nested_dataset_bundle(tmp_path):
    from src.evaluation.threshold_gap_analysis import find_duckdb_file

    bundle = tmp_path / "dataset" / "nested"
    bundle.mkdir(parents=True)
    expected = bundle / "samples.duckdb"
    expected.write_bytes(b"not a real database for discovery")

    assert find_duckdb_file(tmp_path / "dataset") == expected


def test_inference_helper_returns_per_sample_analysis(tmp_path, monkeypatch):
    from src.evaluation import threshold_gap_analysis

    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    db_path = tmp_path / "data.duckdb"
    _write_wav(wav_a, 440)
    _write_wav(wav_b, 660)
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            "CREATE TABLE test (audio_path VARCHAR, labels VARCHAR, text VARCHAR, speaker_id VARCHAR, session_id VARCHAR, is_background BOOLEAN)"
        )
        con.execute("INSERT INTO test VALUES (?, ?, ?, ?, ?, ?)", [str(wav_a), '["move_forward"]', "forward", "spk1", "s1", False])
        con.execute("INSERT INTO test VALUES (?, ?, ?, ?, ?, ?)", [str(wav_b), "[]", "background", "spk2", "s1", True])

    def tiny_model(num_classes, pretrained=False, architecture="mobilenet_v3_small"):
        del pretrained, architecture
        return CompactSpeechCNN(num_classes=num_classes, width=2)

    monkeypatch.setattr(threshold_gap_analysis, "build_model", tiny_model)
    model = tiny_model(num_classes=1)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": ["move_forward"],
            "config": {
                "data_source": {
                    "audio_column": "audio_path",
                    "labels_column": "labels",
                    "text_column": "text",
                    "background_flag_column": "is_background",
                    "metadata_columns": ["speaker_id", "session_id"],
                },
                "dataset": {
                    "unknown_label_policy": "error",
                    "sample_rate": 16000,
                    "duration_seconds": 1.0,
                    "n_fft": 400,
                    "hop_length": 160,
                    "image_size": 32,
                },
                "model": {"architecture": "mobilenet_v3_small"},
                "trainer": {"batch_size": 1, "num_workers": 0, "device": "cpu"},
            },
        },
        checkpoint_path,
    )

    result = threshold_gap_analysis.run_inference(
        checkpoint_path=checkpoint_path,
        database_path=db_path,
        table_name="test",
    )

    assert result.classes == ["move_forward"]
    assert result.probabilities.shape == (2, 1)
    assert result.targets.tolist() == [[1.0], [0.0]]
    assert result.samples["audio_path"].tolist() == [str(wav_a), str(wav_b)]
    assert result.samples.loc[0, "metadata"]["speaker_id"] == "spk1"
