from __future__ import annotations

import argparse
import os
import shutil
import stat
from pathlib import Path
from typing import Callable

import librosa
import numpy as np

from librosa_audio_utils import configure_logging, duration_samples, find_audio_files, fix_length, load_audio, write_audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild processed and augmented audio with librosa at a fixed 32 kHz sample rate and shared length."
    )
    parser.add_argument(
        "--flat-source-dir",
        type=Path,
        default=Path("data/raw/flat"),
        help="Source folder for the flat WAV files.",
    )
    parser.add_argument(
        "--dropbox-source-dir",
        type=Path,
        default=Path("data/raw/dropbox"),
        help="Source folder for the dropbox MP3 files.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("data/audio"),
        help="Root folder containing processed and augmented audio outputs.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=32000,
        help="Target output sample rate in Hz.",
    )
    parser.add_argument(
        "--pitch-semitones",
        type=float,
        default=2.0,
        help="Pitch shift amount in semitones for the pitch_shift augmentation.",
    )
    parser.add_argument(
        "--speed-up-rate",
        type=float,
        default=1.25,
        help="Playback rate for the speed_up augmentation.",
    )
    parser.add_argument(
        "--noise-db",
        type=float,
        default=-24.0,
        help="Relative noise level in dB for the background_noise augmentation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output directories to be deleted and rebuilt.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug-level progress information.",
    )
    return parser


def collect_sources(flat_source_dir: Path, dropbox_source_dir: Path) -> tuple[list[Path], list[Path]]:
    flat_files = find_audio_files(flat_source_dir, "*.wav")
    dropbox_files = find_audio_files(dropbox_source_dir, "*.mp3")
    return flat_files, dropbox_files


def compute_target_length(
    flat_files: list[Path],
    dropbox_files: list[Path],
    sample_rate: int,
) -> int:
    max_length = 0
    for audio_path in [*flat_files, *dropbox_files]:
        audio = load_audio(audio_path, sample_rate)
        max_length = max(max_length, duration_samples(audio))
    return max_length


def wipe_output_tree(audio_root: Path) -> None:
    for relative_path in [
        "processed",
        "augmented",
    ]:
        target = audio_root / relative_path
        if not target.exists():
            continue

        def handle_remove_readonly(func: Callable[[str], None], path: str, exc_info: object) -> None:
            del exc_info
            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(target, onerror=handle_remove_readonly)


def build_processed_outputs(
    flat_files: list[Path],
    dropbox_files: list[Path],
    audio_root: Path,
    sample_rate: int,
    target_length: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    flat_output_dir = audio_root / "processed" / "flat"
    dropbox_output_dir = audio_root / "processed" / "dropbox"
    flat_output_dir.mkdir(parents=True, exist_ok=True)
    dropbox_output_dir.mkdir(parents=True, exist_ok=True)

    flat_audio: list[np.ndarray] = []
    dropbox_audio: list[np.ndarray] = []

    for index, audio_path in enumerate(flat_files, start=1):
        audio = fix_length(load_audio(audio_path, sample_rate), target_length)
        flat_audio.append(audio)
        write_audio(flat_output_dir / f"{index:03d}.wav", audio, sample_rate)

    for index, audio_path in enumerate(dropbox_files, start=1):
        audio = fix_length(load_audio(audio_path, sample_rate), target_length)
        dropbox_audio.append(audio)
        write_audio(dropbox_output_dir / f"{index:03d}.wav", audio, sample_rate)

    return flat_audio, dropbox_audio


def add_background_noise(audio: np.ndarray, noise_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(audio)).astype(np.float32)
    signal_rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    signal_rms = max(signal_rms, 1e-4)
    noise_rms = signal_rms * (10.0 ** (noise_db / 20.0))
    noise = noise / max(float(np.sqrt(np.mean(noise**2))), 1e-8) * noise_rms
    mixed = audio + noise
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def pitch_shift_audio(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones)
    return shifted.astype(np.float32, copy=False)


def time_stretch_audio(audio: np.ndarray, rate: float) -> np.ndarray:
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    return stretched.astype(np.float32, copy=False)


def build_augmentations(
    audio_sets: dict[str, list[np.ndarray]],
    audio_root: Path,
    sample_rate: int,
    target_length: int,
    pitch_semitones: float,
    speed_up_rate: float,
    noise_db: float,
) -> None:
    for augmentation_name in ["background_noise", "pitch_shift", "speed_up"]:
        (audio_root / "augmented" / augmentation_name / "flat").mkdir(parents=True, exist_ok=True)
        (audio_root / "augmented" / augmentation_name / "dropbox").mkdir(parents=True, exist_ok=True)

    for group_name, audio_list in audio_sets.items():
        for index, audio in enumerate(audio_list, start=1):
            noisy = fix_length(add_background_noise(audio, noise_db=noise_db, seed=index), target_length)
            shifted = fix_length(pitch_shift_audio(audio, sample_rate, pitch_semitones), target_length)
            sped_up = fix_length(time_stretch_audio(audio, rate=speed_up_rate), target_length)

            write_audio(audio_root / "augmented" / "background_noise" / group_name / f"{index:03d}.wav", noisy, sample_rate)
            write_audio(audio_root / "augmented" / "pitch_shift" / group_name / f"{index:03d}.wav", shifted, sample_rate)
            write_audio(audio_root / "augmented" / "speed_up" / group_name / f"{index:03d}.wav", sped_up, sample_rate)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    flat_files, dropbox_files = collect_sources(args.flat_source_dir, args.dropbox_source_dir)
    if not flat_files:
        raise FileNotFoundError(f"No flat WAV files found in {args.flat_source_dir}")
    if not dropbox_files:
        raise FileNotFoundError(f"No dropbox MP3 files found in {args.dropbox_source_dir}")

    target_length = compute_target_length(flat_files, dropbox_files, args.sample_rate)
    print(f"Target sample rate: {args.sample_rate} Hz")
    print(f"Target length: {target_length} samples ({target_length / args.sample_rate:.3f} s)")

    if args.audio_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.audio_root} already exists. Pass --overwrite to rebuild the audio tree."
            )
        wipe_output_tree(args.audio_root)

    flat_audio, dropbox_audio = build_processed_outputs(
        flat_files=flat_files,
        dropbox_files=dropbox_files,
        audio_root=args.audio_root,
        sample_rate=args.sample_rate,
        target_length=target_length,
    )

    build_augmentations(
        audio_sets={"flat": flat_audio, "dropbox": dropbox_audio},
        audio_root=args.audio_root,
        sample_rate=args.sample_rate,
        target_length=target_length,
        pitch_semitones=args.pitch_semitones,
        speed_up_rate=args.speed_up_rate,
        noise_db=args.noise_db,
    )

    print(f"Rebuilt processed and augmented audio under {args.audio_root}")


if __name__ == "__main__":
    main()