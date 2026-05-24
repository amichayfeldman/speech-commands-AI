# speech-commands-AI

Multi-label edge speech command training for robot control.

This repository trains a compact multi-label speech model for edge deployment, evaluates thresholded command activations, and exports an ONNX artifact with class and threshold metadata.

The model treats only supported robot commands as classes. Out-of-vocabulary speech, silence, background audio, and irrelevant utterances are not modeled as a special class. They map to an empty label list, which becomes an all-zero multi-hot target vector.

## Why Multi-Label

Robot-control audio can contain overlapping or composed commands. A multi-label model lets each supported command activate independently instead of forcing every clip into exactly one class. This also makes the background/OOV policy explicit: no supported command means no active output.

## Commands

Training:

```bash
python -m src.train experiment=robot_commands
```

Evaluation:

```bash
python -m src.evaluate checkpoint_path=/path/to/model.pt
```

ONNX export:

```bash
python -m src.export_onnx checkpoint_path=/path/to/model.pt export.output_path=artifacts/model.onnx
```

## DuckDB Table Contract

The DuckDB data source is configured in `configs/data_source/duckdb.yaml`.

Required columns:

- `audio_path`: path to a 16-bit PCM WAV file.
- `labels`: a list, JSON list, stringified list, or scalar string containing zero or more supported command labels.

Optional columns:

- `text`: source transcript or synthetic composition text.
- `is_background`: when true, labels are forced to `[]`.
- `speaker_id`, `session_id`, or other configured metadata columns.

Label policy:

- Supported robot commands are the only model classes.
- OOV, silence, and irrelevant speech are stored as `[]` or flagged as background.
- Unknown labels raise an error by default so data issues are visible.

## Configuration

Hydra entrypoints live at:

- `configs/train.yaml`
- `configs/evaluate.yaml`
- `configs/export_onnx.yaml`

Config groups define data source, dataset preprocessing, model, loss, trainer, logger, paths, thresholds, and experiment vocabulary. Local logging is the default; ClearML is present only as an optional disabled config.

## Outputs

Evaluation writes:

- `metrics.json`
- `threshold_recommendations.csv`

ONNX export writes:

- `model.onnx`
- `model.metadata.json` containing the class map, thresholds, input shape, and ONNXRuntime parity result when ONNXRuntime is installed.
