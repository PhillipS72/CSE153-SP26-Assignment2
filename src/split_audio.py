from __future__ import annotations

import argparse
import csv
import html
import logging
import math
import os
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable


LOGGER = logging.getLogger("split_audio")


@dataclass(frozen=True)
class Segment:
    index: int
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class ClipRecord:
    index: int
    output_path: Path
    start_ms: int
    end_ms: int
    duration_ms: int
    dbfs: float
    repeat_group: int | None = None
    repeat_index: int | None = None
    repeat_delta_ms: int | None = None
    repeat_status: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split a long station-jingle MP3/WAV into separate WAV clips using "
            "silence/pause detection."
        )
    )
    parser.add_argument("input_audio", type=Path, help="Long MP3/WAV file to split.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/split_audio"),
        help="Directory where jingle_001.wav, jingle_002.wav, ... will be written.",
    )
    parser.add_argument(
        "--prefix",
        default="jingle",
        help="Filename prefix for exported clips.",
    )
    parser.add_argument(
        "--silence-thresh-dbfs",
        type=float,
        default=None,
        help=(
            "Absolute silence threshold in dBFS. More negative values are stricter. "
            "If omitted, the script uses input dBFS plus --relative-silence-db."
        ),
    )
    parser.add_argument(
        "--relative-silence-db",
        type=float,
        default=-16.0,
        help=(
            "Silence threshold relative to the whole-file average dBFS when "
            "--silence-thresh-dbfs is omitted. Default: average dBFS - 16 dB."
        ),
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=250,
        help="Minimum pause length that can separate two clips.",
    )
    parser.add_argument(
        "--keep-silence-ms",
        type=int,
        default=150,
        help="Milliseconds of pause to keep at the start/end of each exported clip.",
    )
    parser.add_argument(
        "--min-clip-ms",
        type=int,
        default=1200,
        help="Drop detected clips shorter than this duration.",
    )
    parser.add_argument(
        "--merge-gap-ms",
        type=int,
        default=250,
        help="Merge detected regions separated by gaps shorter than this duration.",
    )
    parser.add_argument(
        "--seek-step-ms",
        type=int,
        default=5,
        help="Analysis step size in milliseconds. Smaller is more precise but slower.",
    )
    parser.add_argument(
        "--fade-ms",
        type=int,
        default=10,
        help="Apply a tiny fade-in/out to exported clips to avoid clicks.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Optional sample rate for exported WAV files, e.g. 44100.",
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Export clips as mono WAV files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV path for detected clip timing metadata. Defaults to output-dir/manifest.csv.",
    )
    parser.add_argument(
        "--review-html",
        nargs="?",
        const="review.html",
        default=None,
        help=(
            "Optional HTML review sheet with audio controls. If no path is provided, "
            "writes output-dir/review.html."
        ),
    )
    parser.add_argument(
        "--expected-repeats",
        type=int,
        default=2,
        help=(
            "Expected number of consecutive repeats per melody for diagnostics. "
            "Use 0 to disable repeat checks."
        ),
    )
    parser.add_argument(
        "--repeat-tolerance-ms",
        type=int,
        default=500,
        help="Warn when repeated clips in a group differ by more than this many ms.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect and log segments without exporting audio clips.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output WAV files with the same names.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug-level segmentation details.",
    )
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logging.getLogger("pydub.converter").setLevel(logging.WARNING)


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_audio.exists():
        raise FileNotFoundError(f"Input audio file not found: {args.input_audio}")

    positive_int_fields = [
        "min_silence_ms",
        "keep_silence_ms",
        "min_clip_ms",
        "merge_gap_ms",
        "seek_step_ms",
        "fade_ms",
        "expected_repeats",
        "repeat_tolerance_ms",
    ]
    for field in positive_int_fields:
        value = getattr(args, field)
        if value < 0:
            raise ValueError(f"--{field.replace('_', '-')} must be >= 0")

    if args.seek_step_ms == 0:
        raise ValueError("--seek-step-ms must be > 0")


def load_audio_tools() -> tuple[Any, Callable[..., list[list[int]]], bool]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Couldn't find ffmpeg or avconv.*",
                category=RuntimeWarning,
            )
            from pydub import AudioSegment
            from pydub.silence import detect_nonsilent
            from pydub.utils import which
    except ImportError as exc:  # pragma: no cover - import guard for CLI users
        raise SystemExit(
            "Missing dependency: pydub. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    ffmpeg_available = configure_ffmpeg(AudioSegment, which)
    return AudioSegment, detect_nonsilent, ffmpeg_available


def configure_ffmpeg(
    audio_segment_class: Any,
    which_fn: Callable[[str], str | None],
) -> bool:
    if which_fn("ffmpeg") is not None or which_fn("avconv") is not None:
        return True

    try:
        import imageio_ffmpeg
    except ImportError:
        return False

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    audio_segment_class.converter = ffmpeg_path
    LOGGER.info("Using ffmpeg bundled by imageio-ffmpeg: %s", ffmpeg_path)
    return True


def load_audio_file(audio_segment_class: Any, input_audio: Path) -> Any:
    audio_format = input_audio.suffix.lower().lstrip(".") or None
    codec_by_format = {
        "aac": "aac",
        "flac": "flac",
        "m4a": "aac",
        "mp3": "mp3",
        "oga": "vorbis",
        "ogg": "vorbis",
        "opus": "opus",
    }
    codec = codec_by_format.get(audio_format)
    LOGGER.debug("Loading with format=%s codec=%s", audio_format, codec)
    return audio_segment_class.from_file(
        str(input_audio),
        format=audio_format,
        codec=codec,
    )


def resolve_output_path(path_arg: str | Path | None, output_dir: Path) -> Path | None:
    if path_arg is None:
        return None

    path = Path(path_arg)
    if path.is_absolute():
        return path
    return output_dir / path


def resolve_silence_threshold(
    audio: Any,
    silence_thresh_dbfs: float | None,
    relative_silence_db: float,
) -> float:
    if silence_thresh_dbfs is not None:
        return silence_thresh_dbfs

    if math.isfinite(audio.dBFS):
        return audio.dBFS + relative_silence_db

    LOGGER.warning("Input audio average dBFS is -inf; falling back to -40 dBFS.")
    return -40.0


def detect_segments(
    audio: Any,
    detect_nonsilent_fn: Callable[..., list[list[int]]],
    silence_thresh_dbfs: float,
    min_silence_ms: int,
    keep_silence_ms: int,
    min_clip_ms: int,
    merge_gap_ms: int,
    seek_step_ms: int,
) -> list[Segment]:
    raw_ranges = detect_nonsilent_fn(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_dbfs,
        seek_step=seek_step_ms,
    )
    LOGGER.debug("Raw nonsilent ranges: %s", raw_ranges)

    raw_segments = [
        Segment(
            index=0,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        for start_ms, end_ms in raw_ranges
    ]
    merged = merge_close_segments(raw_segments, merge_gap_ms)
    expanded = [
        Segment(
            index=0,
            start_ms=max(0, segment.start_ms - keep_silence_ms),
            end_ms=min(len(audio), segment.end_ms + keep_silence_ms),
        )
        for segment in merged
    ]

    kept: list[Segment] = []
    dropped = 0
    for segment in expanded:
        if segment.duration_ms >= min_clip_ms:
            kept.append(
                Segment(
                    index=len(kept) + 1,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                )
            )
        else:
            dropped += 1

    if dropped:
        LOGGER.info("Dropped %d clips shorter than %d ms.", dropped, min_clip_ms)

    return kept


def merge_close_segments(segments: Iterable[Segment], merge_gap_ms: int) -> list[Segment]:
    merged: list[Segment] = []
    for segment in sorted(segments, key=lambda item: item.start_ms):
        if not merged:
            merged.append(segment)
            continue

        previous = merged[-1]
        gap_ms = segment.start_ms - previous.end_ms
        if gap_ms <= merge_gap_ms:
            merged[-1] = Segment(
                index=0,
                start_ms=previous.start_ms,
                end_ms=max(previous.end_ms, segment.end_ms),
            )
        else:
            merged.append(segment)
    return merged


def ensure_output_dir(output_dir: Path, prefix: str, overwrite: bool, dry_run: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob(f"{prefix}_*.wav"))
    if existing and not overwrite and not dry_run:
        examples = ", ".join(path.name for path in existing[:3])
        raise FileExistsError(
            f"{output_dir} already contains {prefix}_*.wav files ({examples}). "
            "Use --overwrite to replace matching filenames, or choose another --output-dir."
        )


def export_clips(
    audio: Any,
    segments: list[Segment],
    output_dir: Path,
    prefix: str,
    fade_ms: int,
    sample_rate: int | None,
    mono: bool,
    dry_run: bool,
) -> list[ClipRecord]:
    records: list[ClipRecord] = []

    for segment in segments:
        clip = audio[segment.start_ms : segment.end_ms]

        if fade_ms:
            fade_duration = min(fade_ms, max(0, clip.duration_seconds * 1000 / 2))
            clip = clip.fade_in(int(fade_duration)).fade_out(int(fade_duration))
        if sample_rate:
            clip = clip.set_frame_rate(sample_rate)
        if mono:
            clip = clip.set_channels(1)

        output_path = output_dir / f"{prefix}_{segment.index:03d}.wav"
        if not dry_run:
            clip.export(output_path, format="wav")

        records.append(
            ClipRecord(
                index=segment.index,
                output_path=output_path,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                duration_ms=segment.duration_ms,
                dbfs=clip.dBFS,
            )
        )
        LOGGER.debug(
            "Clip %03d: %s - %s (%s)",
            segment.index,
            format_time(segment.start_ms),
            format_time(segment.end_ms),
            format_duration(segment.duration_ms),
        )

    return records


def annotate_repeat_groups(
    records: list[ClipRecord],
    expected_repeats: int,
    repeat_tolerance_ms: int,
) -> list[ClipRecord]:
    if expected_repeats <= 1:
        return records

    annotated: list[ClipRecord] = []
    flagged_groups = 0

    for group_start in range(0, len(records), expected_repeats):
        group = records[group_start : group_start + expected_repeats]
        group_number = group_start // expected_repeats + 1
        first_duration = group[0].duration_ms
        is_complete = len(group) == expected_repeats
        max_delta = max(abs(record.duration_ms - first_duration) for record in group)

        if not is_complete:
            status = "incomplete"
            flagged_groups += 1
        elif max_delta > repeat_tolerance_ms:
            status = "check"
            flagged_groups += 1
        else:
            status = "ok"

        for repeat_index, record in enumerate(group, start=1):
            annotated.append(
                replace(
                    record,
                    repeat_group=group_number,
                    repeat_index=repeat_index,
                    repeat_delta_ms=record.duration_ms - first_duration,
                    repeat_status=status,
                )
            )

        if status == "check":
            durations = ", ".join(f"{record.duration_ms / 1000:.3f}s" for record in group)
            LOGGER.warning(
                "Repeat group %03d duration mismatch: %s (tolerance %d ms).",
                group_number,
                durations,
                repeat_tolerance_ms,
            )
        elif status == "incomplete":
            LOGGER.warning(
                "Repeat group %03d has only %d/%d clip(s).",
                group_number,
                len(group),
                expected_repeats,
            )

    LOGGER.info(
        "Repeat diagnostics: %d group(s) of %d, %d flagged.",
        math.ceil(len(records) / expected_repeats),
        expected_repeats,
        flagged_groups,
    )
    return annotated


def write_manifest(records: list[ClipRecord], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "index",
                "repeat_group",
                "repeat_index",
                "repeat_delta_ms",
                "repeat_status",
                "output_path",
                "start_ms",
                "end_ms",
                "start_s",
                "end_s",
                "duration_ms",
                "duration_s",
                "clip_dbfs",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "index": record.index,
                    "repeat_group": record.repeat_group or "",
                    "repeat_index": record.repeat_index or "",
                    "repeat_delta_ms": (
                        "" if record.repeat_delta_ms is None else record.repeat_delta_ms
                    ),
                    "repeat_status": record.repeat_status,
                    "output_path": str(record.output_path),
                    "start_ms": record.start_ms,
                    "end_ms": record.end_ms,
                    "start_s": f"{record.start_ms / 1000:.3f}",
                    "end_s": f"{record.end_ms / 1000:.3f}",
                    "duration_ms": record.duration_ms,
                    "duration_s": f"{record.duration_ms / 1000:.3f}",
                    "clip_dbfs": format_dbfs(record.dbfs),
                }
            )


def write_review_html(records: list[ClipRecord], review_path: Path) -> None:
    review_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        relative_audio_path = os.path.relpath(record.output_path, start=review_path.parent)
        audio_src = Path(relative_audio_path).as_posix()
        rows.append(
            "      <tr>"
            f"<td>{record.index:03d}</td>"
            f"<td>{record.repeat_group or ''}</td>"
            f"<td>{record.repeat_index or ''}</td>"
            f"<td>{html.escape(record.repeat_status)}</td>"
            f"<td>{format_time(record.start_ms)}</td>"
            f"<td>{format_duration(record.duration_ms)}</td>"
            f"<td>{format_dbfs(record.dbfs)}</td>"
            f'<td><audio controls src="{html.escape(audio_src)}"></audio></td>'
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Jingle Split Review</title>
  <style>
    body {{
      color: #1f2328;
      font-family: Arial, sans-serif;
      margin: 24px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
    }}
    th, td {{
      border-bottom: 1px solid #d0d7de;
      padding: 8px;
      text-align: left;
      vertical-align: middle;
    }}
    th {{
      background: #f6f8fa;
      position: sticky;
      top: 0;
    }}
    audio {{
      width: 320px;
      max-width: 100%;
    }}
  </style>
</head>
<body>
  <h1>Jingle Split Review</h1>
  <table>
    <thead>
      <tr>
        <th>Clip</th>
        <th>Group</th>
        <th>Repeat</th>
        <th>Status</th>
        <th>Start</th>
        <th>Duration</th>
        <th>dBFS</th>
        <th>Preview</th>
      </tr>
    </thead>
    <tbody>
{chr(10).join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    review_path.write_text(html_text, encoding="utf-8")


def format_time(ms: int) -> str:
    total_seconds = ms / 1000
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:06.3f}"


def format_duration(ms: int) -> str:
    return f"{ms / 1000:.3f}s"


def format_dbfs(dbfs: float) -> str:
    if not math.isfinite(dbfs):
        return "-inf"
    return f"{dbfs:.2f}"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    validate_args(args)

    AudioSegment, detect_nonsilent_fn, ffmpeg_available = load_audio_tools()

    if not ffmpeg_available:
        LOGGER.warning(
            "ffmpeg was not found on PATH and imageio-ffmpeg is not installed. "
            "WAV may work, but MP3 loading usually needs ffmpeg."
        )

    output_dir = args.output_dir
    manifest_path = args.manifest or output_dir / "manifest.csv"
    review_path = resolve_output_path(args.review_html, output_dir)

    ensure_output_dir(output_dir, args.prefix, args.overwrite, args.dry_run)

    LOGGER.info("Loading %s", args.input_audio)
    audio = load_audio_file(AudioSegment, args.input_audio)
    LOGGER.info(
        "Loaded %.2f minutes, %d Hz, %d channel(s), average %s dBFS.",
        len(audio) / 60_000,
        audio.frame_rate,
        audio.channels,
        format_dbfs(audio.dBFS),
    )

    silence_thresh_dbfs = resolve_silence_threshold(
        audio,
        args.silence_thresh_dbfs,
        args.relative_silence_db,
    )
    LOGGER.info(
        "Detecting pauses: threshold %s dBFS, min silence %d ms, keep %d ms.",
        format_dbfs(silence_thresh_dbfs),
        args.min_silence_ms,
        args.keep_silence_ms,
    )

    segments = detect_segments(
        audio=audio,
        detect_nonsilent_fn=detect_nonsilent_fn,
        silence_thresh_dbfs=silence_thresh_dbfs,
        min_silence_ms=args.min_silence_ms,
        keep_silence_ms=args.keep_silence_ms,
        min_clip_ms=args.min_clip_ms,
        merge_gap_ms=args.merge_gap_ms,
        seek_step_ms=args.seek_step_ms,
    )

    if not segments:
        LOGGER.warning("No clips detected. Try a less negative threshold or shorter min silence.")
        return

    records = export_clips(
        audio=audio,
        segments=segments,
        output_dir=output_dir,
        prefix=args.prefix,
        fade_ms=args.fade_ms,
        sample_rate=args.sample_rate,
        mono=args.mono,
        dry_run=args.dry_run,
    )
    records = annotate_repeat_groups(
        records,
        expected_repeats=args.expected_repeats,
        repeat_tolerance_ms=args.repeat_tolerance_ms,
    )

    write_manifest(records, manifest_path)
    LOGGER.info("Wrote manifest: %s", manifest_path)

    if review_path:
        if args.dry_run:
            LOGGER.warning("Skipping review HTML in --dry-run mode because clips were not exported.")
        else:
            write_review_html(records, review_path)
            LOGGER.info("Wrote review HTML: %s", review_path)

    if args.dry_run:
        LOGGER.info("Dry run complete: detected %d clips.", len(records))
    else:
        LOGGER.info("Exported %d clips to %s", len(records), output_dir)


if __name__ == "__main__":
    main()
