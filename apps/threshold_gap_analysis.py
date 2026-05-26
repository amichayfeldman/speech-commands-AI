from __future__ import annotations

from pathlib import Path
import tempfile

import plotly.graph_objects as go
import streamlit as st

from src.data.audio_features import load_wav_mono, log_spectrogram
from src.evaluation.threshold_gap_analysis import (
    InferenceResult,
    class_metrics_at_threshold,
    class_roc_curve,
    confusion_labels,
    download_clearml_dataset,
    download_clearml_model,
    find_duckdb_file,
    run_inference,
    samples_for_confusion,
    threshold_markers,
)


st.set_page_config(page_title="Threshold Tuning And Gap Analysis", layout="wide")


def _init_thresholds(result: InferenceResult) -> None:
    current = st.session_state.get("thresholds")
    if current and set(current) == set(result.classes):
        return
    st.session_state.thresholds = {label: 0.5 for label in result.classes}


def _download_and_run(model_id: str, dataset_id: str, table_name: str) -> InferenceResult:
    workdir = Path(st.session_state.setdefault("analysis_workdir", tempfile.mkdtemp(prefix="threshold-gap-")))
    progress = st.sidebar.progress(0)
    status = st.sidebar.empty()
    status.write("Downloading ClearML model...")
    checkpoint_path = download_clearml_model(model_id, workdir / "model")
    progress.progress(25)
    status.write("Downloading ClearML dataset...")
    dataset_dir = download_clearml_dataset(dataset_id, workdir / "dataset")
    progress.progress(50)
    status.write("Finding DuckDB bundle...")
    database_path = find_duckdb_file(dataset_dir)
    progress.progress(65)
    status.write("Running inference...")
    result = run_inference(checkpoint_path=checkpoint_path, database_path=database_path, table_name=table_name)
    progress.progress(100)
    status.write("Inference complete.")
    return result


def _metric_cols(metrics: dict[str, float | int | None]) -> None:
    cols = st.columns(8)
    cols[0].metric("Precision", f"{metrics['precision']:.3f}")
    cols[1].metric("Recall", f"{metrics['recall']:.3f}")
    cols[2].metric("F1", f"{metrics['f1']:.3f}")
    auroc = metrics["auroc"]
    cols[3].metric("AUROC", "n/a" if auroc is None else f"{auroc:.3f}")
    cols[4].metric("TP", str(metrics["tp"]))
    cols[5].metric("FP", str(metrics["fp"]))
    cols[6].metric("TN", str(metrics["tn"]))
    cols[7].metric("FN", str(metrics["fn"]))


def _roc_tab(result: InferenceResult) -> None:
    selected_class = st.selectbox("Class", result.classes, key="roc_class")
    class_index = result.classes.index(selected_class)
    threshold = st.slider(
        "Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.thresholds[selected_class]),
        step=0.01,
        key=f"threshold_{selected_class}",
    )
    st.session_state.thresholds[selected_class] = threshold

    curve = class_roc_curve(result.targets[:, class_index], result.probabilities[:, class_index])
    markers = threshold_markers(curve, count=11)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve.fpr, y=curve.tpr, mode="lines", name="ROC"))
    fig.add_trace(
        go.Scatter(
            x=[marker["fpr"] for marker in markers],
            y=[marker["tpr"] for marker in markers],
            mode="markers+text",
            text=[f"{marker['threshold']:.2f}" for marker in markers],
            textposition="top center",
            name="Threshold markers",
        )
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "#888"})
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=520)
    st.plotly_chart(fig, use_container_width=True)
    _metric_cols(class_metrics_at_threshold(result.targets, result.probabilities, class_index=class_index, threshold=threshold))


def _gap_tab(result: InferenceResult) -> None:
    selected_class = st.selectbox("Class", result.classes, key="gap_class")
    class_index = result.classes.index(selected_class)
    threshold = float(st.session_state.thresholds[selected_class])
    error_type = st.selectbox("Type", ["FN", "FP", "TP", "TN"])
    matches = samples_for_confusion(result, class_index=class_index, threshold=threshold, confusion=error_type)
    st.caption(f"{len(matches)} matching samples")
    if matches.empty:
        return

    sample_pos = st.slider("Sample", min_value=0, max_value=len(matches) - 1, value=0)
    sample = matches.iloc[sample_pos]
    sample_index = int(sample["sample_index"])
    probability = float(result.probabilities[sample_index, class_index])
    target = int(result.targets[sample_index, class_index])
    predicted = int(probability >= threshold)

    cols = st.columns(4)
    cols[0].metric("Probability", f"{probability:.3f}")
    cols[1].metric("Threshold", f"{threshold:.2f}")
    cols[2].metric("Prediction", str(predicted))
    cols[3].metric("Ground Truth", str(target))

    audio_path = Path(sample["audio_path"])
    st.audio(str(audio_path))
    audio = load_wav_mono(audio_path, sample_rate=16000)
    spectrogram = log_spectrogram(audio, sample_rate=16000, duration_seconds=1.0, n_fft=400, hop_length=160, image_size=224)
    st.image(spectrogram.squeeze(0).numpy(), caption="Log spectrogram", clamp=True)

    st.code(str(audio_path), language=None)
    metadata = sample.get("metadata", {})
    if metadata:
        st.json(metadata)
    st.caption(f"Confusion: {confusion_labels(result.targets, result.probabilities, class_index=class_index, threshold=threshold)[sample_index]}")


with st.sidebar:
    st.header("Inputs")
    model_id = st.text_input("ClearML checkpoint/model hash ID")
    dataset_id = st.text_input("ClearML dataset hash ID")
    table_name = st.selectbox("Split/table", ["test", "val", "train"], index=0)
    run = st.button("Run", type="primary", disabled=not model_id or not dataset_id)

if run:
    try:
        st.session_state.analysis_result = _download_and_run(model_id, dataset_id, table_name)
        _init_thresholds(st.session_state.analysis_result)
    except Exception as exc:
        st.error(str(exc))

result = st.session_state.get("analysis_result")
if result is None:
    st.info("Enter ClearML IDs in the sidebar and run inference.")
else:
    _init_thresholds(result)
    roc_tab, gap_tab = st.tabs(["ROC Threshold Tuning", "Gap Analysis"])
    with roc_tab:
        _roc_tab(result)
    with gap_tab:
        _gap_tab(result)
