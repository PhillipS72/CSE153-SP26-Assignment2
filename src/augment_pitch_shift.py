from __future__ import annotations

import argparse
from pathlib import Path

from pydub import AudioSegment

from audio_augmentation_utils import configure_ffmpeg, configure_logging, find_audio_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pitch-shift WAV files from the flat processed dataset without changing duration too much."
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
        default=Path("data/audio/augmented/pitch_shift/flat"),
        help="Folder where pitch-shifted WAV files will be written.",
    )
    parser.add_argument("--pattern", default="*.wav", help="Glob pattern for input files.")
    parser.add_argument(
        "--semitones",
        type=float,
        default=2.0,
        help="Pitch shift amount in semitones. Positive values raise pitch; negative values lower it.",
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


def pitch_shift(audio: AudioSegment, semitones: float) -> AudioSegment:
    factor = 2 ** (semitones / 12.0)
    shifted = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * factor)})
    return shifted.set_frame_rate(audio.frame_rate)


def augment_file(input_path: Path, output_path: Path, semitones: float, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        return False

    audio = AudioSegment.from_file(input_path)
    shifted = pitch_shift(audio, semitones)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shifted.export(output_path, format="wav")
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
        if augment_file(input_path, output_path, args.semitones, args.overwrite):
            written += 1

    print(f"Finished: {written}/{len(input_files)} files written to {args.output_dir}")


if __name__ == "__main__":
    main()