from __future__ import annotations

import argparse
from pathlib import Path

from pydub import AudioSegment

from audio_augmentation_utils import configure_ffmpeg, configure_logging, find_audio_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create slower WAV copies of the flat processed dataset."
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
        default=Path("data/audio/augmented/slow_down/flat"),
        help="Folder where slowed-down WAV files will be written.",
    )
    parser.add_argument("--pattern", default="*.wav", help="Glob pattern for input files.")
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=0.85,
        help="Playback speed multiplier. Values below 1.0 make audio slower.",
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


def slow_down(audio: AudioSegment, speed_factor: float) -> AudioSegment:
    if speed_factor <= 0:
        raise ValueError("speed_factor must be greater than 0")
    slowed = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * speed_factor)})
    return slowed.set_frame_rate(audio.frame_rate)


def augment_file(input_path: Path, output_path: Path, speed_factor: float, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        return False

    audio = AudioSegment.from_file(input_path)
    slowed = slow_down(audio, speed_factor)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    slowed.export(output_path, format="wav")
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
        output_path = args.output_dir / f"{input_path.stem}.wav"
        if augment_file(input_path, output_path, args.speed_factor, args.overwrite):
            written += 1

    print(f"Finished: {written}/{len(input_files)} files written to {args.output_dir}")


if __name__ == "__main__":
    main()