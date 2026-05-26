from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pydub import AudioSegment
from pydub.effects import speedup
from pydub.utils import which


LOGGER = logging.getLogger("augment_speed_up")


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


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def configure_ffmpeg() -> None:
    if which("ffmpeg") is not None or which("avconv") is not None:
        return

    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - import guard for CLI users
        raise SystemExit(
            "Missing dependency: ffmpeg. Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
    LOGGER.info("Using ffmpeg bundled by imageio-ffmpeg: %s", AudioSegment.converter)


def find_audio_files(input_dir: Path, pattern: str) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No audio files matched {input_dir / pattern}")
    return files


def augment_file(
    input_path: Path,
    output_path: Path,
    speed_factor: float,
    overwrite: bool,
) -> bool:
    if output_path.exists() and not overwrite:
        LOGGER.info("Skipping existing file: %s", output_path.name)
        return False

    audio = AudioSegment.from_file(input_path)
    augmented = speedup(audio, playback_speed=speed_factor)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented.export(output_path, format="wav")
    LOGGER.info("Wrote %s", output_path)
    return True


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    configure_ffmpeg()

    if args.speed_factor <= 0:
        raise ValueError("--speed-factor must be greater than 0")

    input_files = find_audio_files(args.input_dir, args.pattern)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for input_path in input_files:
        output_path = args.output_dir / input_path.name
        if augment_file(input_path, output_path, args.speed_factor, args.overwrite):
            written += 1

    LOGGER.info("Finished: %d/%d files written.", written, len(input_files))


if __name__ == "__main__":
    main()