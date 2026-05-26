from __future__ import annotations

import logging
from pathlib import Path

from pydub import AudioSegment
from pydub.utils import which


LOGGER = logging.getLogger("audio_augmentation")


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