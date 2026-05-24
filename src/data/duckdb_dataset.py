from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import duckdb
import pandas as pd
from torch.utils.data import Dataset

from src.data.audio_features import load_wav_mono, log_spectrogram
from src.data.label_encoding import LabelEncoder, parse_labels


@dataclass
class DuckDBSpeechCommandsDataset(Dataset):
    database_path: str
    table_name: str
    audio_column: str
    labels_column: str
    classes: list[str]
    text_column: str | None = None
    background_flag_column: str | None = None
    metadata_columns: list[str] = field(default_factory=list)
    unknown_label_policy: str = "error"
    sample_rate: int = 16000
    duration_seconds: float = 1.0
    n_fft: int = 400
    hop_length: int = 160
    image_size: int = 224

    def __post_init__(self) -> None:
        self.encoder = LabelEncoder(self.classes, self.unknown_label_policy)
        with duckdb.connect(self.database_path, read_only=True) as con:
            self.dataframe = con.execute(f"SELECT * FROM {self.table_name}").fetchdf()
        required = {self.audio_column, self.labels_column}
        missing = required - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"Missing DuckDB columns in {self.table_name}: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]
        path = str(row[self.audio_column])
        raw_labels = [] if self._is_background(row) else parse_labels(row[self.labels_column])
        target = self.encoder.encode(raw_labels)
        audio = load_wav_mono(path, self.sample_rate)
        features = log_spectrogram(
            audio,
            sample_rate=self.sample_rate,
            duration_seconds=self.duration_seconds,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            image_size=self.image_size,
        )
        metadata = self._metadata(row)
        return {
            "input": features,
            "target": target,
            "labels": self.encoder.decode(target),
            "path": path,
            "metadata": metadata,
        }

    def _is_background(self, row: pd.Series) -> bool:
        if not self.background_flag_column:
            return False
        if self.background_flag_column not in row:
            return False
        value = row[self.background_flag_column]
        if pd.isna(value):
            return False
        return bool(value)

    def _metadata(self, row: pd.Series) -> dict[str, Any]:
        keys = list(self.metadata_columns)
        if self.text_column:
            keys.append(self.text_column)
        return {key: row[key] for key in keys if key in row and not pd.isna(row[key])}
