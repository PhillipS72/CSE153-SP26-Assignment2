from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


LOGGER = logging.getLogger("librosa_audio")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def find_audio_files(input_dir: Path, pattern: str) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No audio files matched {input_dir / pattern}")
    return files


def load_audio(audio_path: Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    return audio.astype(np.float32, copy=False)


def fix_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    if target_length < 0:
        raise ValueError("target_length must be >= 0")
    if len(audio) == target_length:
        return audio.astype(np.float32, copy=False)
    return librosa.util.fix_length(audio.astype(np.float32, copy=False), size=target_length)


def write_audio(audio_path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(audio_path, audio, sample_rate, subtype="PCM_16")


def duration_samples(audio: np.ndarray) -> int:
    return int(audio.shape[0])
