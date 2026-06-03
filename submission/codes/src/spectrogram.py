from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import librosa
import librosa.display

from librosa_audio_utils import configure_logging, find_audio_files, load_audio


LOGGER = logging.getLogger("spectrogram")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate spectrogram images for a folder of audio files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/audio/processed"),
        help="Folder containing audio files to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/spectrograms"),
        help="Folder where PNG spectrograms will be written.",
    )
    parser.add_argument(
        "--pattern",
        default="*.wav",
        help="Glob pattern for audio files inside --input-dir.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=32000,
        help="Sample rate used to load and visualize the audio.",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=2048,
        help="FFT window size used for the spectrogram.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Hop length used for the spectrogram.",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=20.0,
        help="Lowest frequency shown in the mel spectrogram.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Highest frequency shown in the mel spectrogram. Defaults to Nyquist.",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colormap used for the image.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional display cap in seconds for the rendered spectrogram.",
    )
    parser.add_argument(
        "--trim-db",
        type=float,
        default=35.0,
        help="Top dB threshold used to trim leading/trailing silence before plotting.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N audio files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug-level progress information.",
    )
    return parser


def render_spectrogram(
    audio_path: Path,
    output_path: Path,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    fmin: float,
    fmax: float | None,
    cmap: str,
    max_seconds: float | None,
    trim_db: float,
) -> None:
    audio = load_audio(audio_path, sample_rate)
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=trim_db)
    if len(trimmed_audio) > 0:
        audio = trimmed_audio

    if max_seconds is not None:
        max_samples = int(round(max_seconds * sample_rate))
        audio = audio[:max_samples]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=1.0)

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    image = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=fmin,
        fmax=fmax,
        cmap=cmap,
        ax=ax,
    )
    ax.set_title(audio_path.stem)
    fig.colorbar(image, ax=ax, format="%+2.0f dB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    audio_files = find_audio_files(args.input_dir, args.pattern)
    if args.limit is not None:
        audio_files = audio_files[: args.limit]

    if not audio_files:
        raise FileNotFoundError(f"No audio files matched {args.input_dir / args.pattern}")

    for audio_path in audio_files:
        output_path = args.output_dir / f"{audio_path.stem}.png"
        LOGGER.info("Rendering %s -> %s", audio_path, output_path)
        render_spectrogram(
            audio_path=audio_path,
            output_path=output_path,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            fmin=args.fmin,
            fmax=args.fmax,
            cmap=args.cmap,
            max_seconds=args.max_seconds,
            trim_db=args.trim_db,
        )

    LOGGER.info("Wrote %d spectrograms to %s", len(audio_files), args.output_dir)


if __name__ == "__main__":
    main()