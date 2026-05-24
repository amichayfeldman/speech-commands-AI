from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.config import classes_from_config
from src.data import DuckDBSpeechCommandsDataset
from src.losses import MultiLabelBCELoss
from src.models import build_model


def build_dataset(config: DictConfig, split: str, classes: list[str] | None = None) -> DuckDBSpeechCommandsDataset:
    table_name = getattr(config.data_source, f"{split}_table")
    if classes is None:
        classes = classes_from_config(config)
    return DuckDBSpeechCommandsDataset(
        database_path=config.data_source.database_path,
        table_name=table_name,
        audio_column=config.data_source.audio_column,
        labels_column=config.data_source.labels_column,
        text_column=config.data_source.get("text_column"),
        background_flag_column=config.data_source.get("background_flag_column"),
        metadata_columns=list(config.data_source.get("metadata_columns", [])),
        classes=classes,
        unknown_label_policy=config.dataset.unknown_label_policy,
        sample_rate=config.dataset.sample_rate,
        duration_seconds=config.dataset.duration_seconds,
        n_fft=config.dataset.n_fft,
        hop_length=config.dataset.hop_length,
        image_size=config.dataset.image_size,
    )


def collate_batch(batch: list[dict]) -> dict:
    return {
        "input": torch.stack([item["input"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "labels": [item["labels"] for item in batch],
        "path": [item["path"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def train(config: DictConfig) -> Path:
    classes = classes_from_config(config)
    device = torch.device(config.trainer.device if torch.cuda.is_available() or config.trainer.device == "cpu" else "cpu")
    train_dataset = build_dataset(config, "train")
    val_dataset = build_dataset(config, "val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=config.trainer.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        num_workers=config.trainer.num_workers,
        collate_fn=collate_batch,
    )
    model = build_model(num_classes=len(classes), pretrained=config.model.pretrained, architecture=config.model.architecture).to(device)
    criterion = MultiLabelBCELoss(reduction=config.loss.reduction)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.trainer.learning_rate, weight_decay=config.trainer.weight_decay)
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for _epoch in range(config.trainer.epochs):
        model.train()
        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                _ = criterion(model(batch["input"].to(device)), batch["target"].to(device))

    checkpoint_path = output_dir / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "classes": classes, "config": OmegaConf.to_container(config, resolve=True)}, checkpoint_path)
    return checkpoint_path


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(config: DictConfig) -> None:
    train(config)


if __name__ == "__main__":
    main()
