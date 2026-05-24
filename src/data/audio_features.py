from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_wav_mono(path: str | Path, sample_rate: int) -> torch.Tensor:
    """Load a PCM WAV file as mono float32 audio.

    The project keeps audio I/O intentionally small for tests and portability. Production
    runs can pre-normalize files to the configured sample rate before inserting paths into DuckDB.
    """
    with wave.open(str(path), "rb") as wav:
        source_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported by the built-in loader: {path}")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    tensor = torch.from_numpy(audio)
    if source_rate != sample_rate:
        new_length = max(1, int(round(tensor.numel() * sample_rate / source_rate)))
        tensor = F.interpolate(tensor.view(1, 1, -1), size=new_length, mode="linear", align_corners=False).view(-1)
    return tensor


def fixed_length(audio: torch.Tensor, num_samples: int) -> torch.Tensor:
    if audio.numel() >= num_samples:
        return audio[:num_samples]
    return F.pad(audio, (0, num_samples - audio.numel()))


def log_spectrogram(
    audio: torch.Tensor,
    *,
    sample_rate: int,
    duration_seconds: float,
    n_fft: int,
    hop_length: int,
    image_size: int,
) -> torch.Tensor:
    audio = fixed_length(audio.float(), int(round(duration_seconds * sample_rate)))
    window = torch.hann_window(n_fft)
    spec = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    power = spec.abs().pow(2.0)
    log_power = torch.log1p(power)
    denom = log_power.max() - log_power.min()
    if float(denom) > 0.0:
        log_power = (log_power - log_power.min()) / denom
    else:
        log_power = torch.zeros_like(log_power)
    return F.interpolate(log_power.unsqueeze(0).unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
