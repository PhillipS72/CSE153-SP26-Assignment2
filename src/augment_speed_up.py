from __future__ import annotations

import argparse
from pathlib import Path

import librosa

from librosa_audio_utils import configure_logging, find_audio_files, fix_length, load_audio, write_audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create sped-up WAV copies of a folder of audio files."
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
        default=Path("data/audio/augmented/speed_up/flat"),
        help="Folder where sped-up WAV files will be written.",
    )
    parser.add_argument(
        "--pattern",
        default="*.wav",
        help="Glob pattern used to select input files.",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=1.25,
        help="Playback speed multiplier. Values above 1.0 make audio shorter and faster.",
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    if args.speed_factor <= 0:
        raise ValueError("--speed-factor must be greater than 0")

    input_files = find_audio_files(args.input_dir, args.pattern)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for input_path in input_files:
        output_path = args.output_dir / f"{input_path.stem}.wav"
        if output_path.exists() and not args.overwrite:
            continue

        audio = load_audio(input_path, sample_rate=32000)
        augmented = librosa.effects.time_stretch(audio, rate=args.speed_factor)
        augmented = fix_length(augmented, len(audio))
        write_audio(output_path, augmented, 32000)
        if args.verbose:
            print(f"Wrote {output_path}")
            written += 1

    print(f"Finished: {written}/{len(input_files)} files written.")


if __name__ == "__main__":
    main()