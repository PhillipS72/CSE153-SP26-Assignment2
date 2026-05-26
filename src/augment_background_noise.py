from __future__ import annotations

import argparse
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import WhiteNoise

from audio_augmentation_utils import configure_ffmpeg, configure_logging, find_audio_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add background noise to WAV files from the flat processed dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/audio/processed/flat"),
        help="Folder containing the source WAV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/audio/augmented/background_noise/flat"),
        help="Folder where noisy WAV files will be written.",
    )
    parser.add_argument("--pattern", default="*.wav", help="Glob pattern for input files.")
    parser.add_argument(
        "--noise-gain-db",
        type=float,
        default=-24.0,
        help="Gain to apply to the generated noise before mixing.",
    )
    parser.add_argument(
        "--mix-gain-db",
        type=float,
        default=-3.0,
        help="Gain applied to the original audio before overlaying noise.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug-level logging.",
    )
    return parser


def add_background_noise(input_path: Path, output_path: Path, noise_gain_db: float, mix_gain_db: float, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        return False

    audio = AudioSegment.from_file(input_path)
    noise = WhiteNoise().to_audio_segment(duration=len(audio), volume=noise_gain_db)
    noisy_audio = audio.apply_gain(mix_gain_db).overlay(noise)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    noisy_audio.export(output_path, format="wav")
    return True


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    configure_ffmpeg()

    input_files = find_audio_files(args.input_dir, args.pattern)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for input_path in input_files:
        output_path = args.output_dir / input_path.name
        if add_background_noise(input_path, output_path, args.noise_gain_db, args.mix_gain_db, args.overwrite):
            written += 1

    print(f"Finished: {written}/{len(input_files)} files written to {args.output_dir}")


if __name__ == "__main__":
    main()