from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LOGGER = logging.getLogger("transcribe_basic_pitch")


@dataclass(frozen=True)
class TranscriptionRecord:
    input_audio: Path
    midi_path: Path
    note_events_path: Path | None
    note_count: int
    duration_s: float
    min_pitch: int | None
    max_pitch: int | None
    mean_pitch: float | None
    mean_velocity: float | None
    status: str
    error: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-transcribe split WAV jingles to MIDI with Spotify Basic Pitch."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/split_audio"),
        help="Folder containing split WAV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/midi"),
        help="Folder where MIDI files and metadata will be written.",
    )
    parser.add_argument(
        "--pattern",
        default="*.wav",
        help="Glob pattern for audio files inside --input-dir.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Summary CSV path. Defaults to output-dir/transcription_summary.csv.",
    )
    parser.add_argument(
        "--pair-summary",
        type=Path,
        default=None,
        help="Repeat-pair comparison CSV path. Defaults to output-dir/repeat_pair_summary.csv.",
    )
    parser.add_argument(
        "--save-note-events",
        action="store_true",
        default=True,
        help="Save one CSV of Basic Pitch note events per audio file.",
    )
    parser.add_argument(
        "--no-note-events",
        action="store_false",
        dest="save_note_events",
        help="Do not save per-file note-event CSVs.",
    )
    parser.add_argument(
        "--note-events-dir",
        type=Path,
        default=None,
        help="Folder for note-event CSVs. Defaults to output-dir/note_events.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing MIDI and note-event CSV files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Transcribe only the first N files. Useful for a quick smoke test.",
    )
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.5,
        help="Basic Pitch onset threshold. Higher values produce fewer notes.",
    )
    parser.add_argument(
        "--frame-threshold",
        type=float,
        default=0.3,
        help="Basic Pitch frame threshold. Higher values shorten/remove weak notes.",
    )
    parser.add_argument(
        "--minimum-note-length-ms",
        type=float,
        default=127.7,
        help="Drop notes shorter than this length during Basic Pitch post-processing.",
    )
    parser.add_argument(
        "--minimum-frequency",
        type=float,
        default=None,
        help="Optional lower frequency bound in Hz.",
    )
    parser.add_argument(
        "--maximum-frequency",
        type=float,
        default=None,
        help="Optional upper frequency bound in Hz.",
    )
    parser.add_argument(
        "--multiple-pitch-bends",
        action="store_true",
        help="Allow overlapping pitch bends in MIDI output.",
    )
    parser.add_argument(
        "--disable-melodia-trick",
        action="store_true",
        help="Disable Basic Pitch's melodia post-processing trick.",
    )
    parser.add_argument(
        "--midi-tempo",
        type=float,
        default=120.0,
        help="Tempo written into generated MIDI files.",
    )
    parser.add_argument(
        "--include-pitch-bends",
        action="store_true",
        help="Include Basic Pitch pitch-bend arrays in note-event CSVs.",
    )
    parser.add_argument(
        "--expected-repeats",
        type=int,
        default=2,
        help="Expected consecutive repeats per melody for transcription diagnostics.",
    )
    parser.add_argument(
        "--pair-note-tolerance",
        type=int,
        default=12,
        help="Flag repeat pairs whose note counts differ by more than this amount.",
    )
    parser.add_argument(
        "--pair-duration-tolerance-s",
        type=float,
        default=0.35,
        help="Flag repeat pairs whose transcribed durations differ by more than this amount.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug-level information.",
    )
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def load_basic_pitch() -> tuple[Any, Any, Any]:
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model, predict
    except ImportError as exc:  # pragma: no cover - import guard for CLI users
        raise SystemExit(
            "Missing dependency: basic-pitch. Install it in a Python 3.11 "
            "environment with `pip install -r requirements-basic-pitch.txt`."
        ) from exc

    return ICASSP_2022_MODEL_PATH, Model, predict


def find_audio_files(input_dir: Path, pattern: str, limit: int | None) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    audio_files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if limit is not None:
        audio_files = audio_files[:limit]

    if not audio_files:
        raise FileNotFoundError(f"No audio files matched {input_dir / pattern}")

    return audio_files


def ensure_output_paths(
    output_dir: Path,
    note_events_dir: Path | None,
    save_note_events: bool,
) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_note_events_dir = None
    if save_note_events:
        resolved_note_events_dir = note_events_dir or output_dir / "note_events"
        resolved_note_events_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, resolved_note_events_dir


def transcribe_file(
    audio_path: Path,
    output_dir: Path,
    note_events_dir: Path | None,
    predict_fn: Any,
    model: Any,
    args: argparse.Namespace,
) -> TranscriptionRecord:
    midi_path = output_dir / f"{audio_path.stem}.mid"
    note_events_path = (
        note_events_dir / f"{audio_path.stem}_notes.csv" if note_events_dir else None
    )

    if midi_path.exists() and not args.overwrite:
        if note_events_path and note_events_path.exists():
            stats = summarize_note_events(read_note_events(note_events_path))
            status = "skipped_exists"
        else:
            stats = {
                "note_count": 0,
                "duration_s": 0.0,
                "min_pitch": None,
                "max_pitch": None,
                "mean_pitch": None,
                "mean_velocity": None,
            }
            status = "skipped_exists_no_notes"

        return TranscriptionRecord(
            input_audio=audio_path,
            midi_path=midi_path,
            note_events_path=note_events_path,
            status=status,
            **stats,
        )

    LOGGER.info("Transcribing %s", audio_path.name)
    try:
        _, midi_data, note_events = predict_fn(
            audio_path,
            model,
            onset_threshold=args.onset_threshold,
            frame_threshold=args.frame_threshold,
            minimum_note_length=args.minimum_note_length_ms,
            minimum_frequency=args.minimum_frequency,
            maximum_frequency=args.maximum_frequency,
            multiple_pitch_bends=args.multiple_pitch_bends,
            melodia_trick=not args.disable_melodia_trick,
            midi_tempo=args.midi_tempo,
        )
        midi_data.write(str(midi_path))

        sorted_note_events = sort_note_events(note_events)

        if note_events_path:
            write_note_events(
                sorted_note_events,
                note_events_path,
                include_pitch_bends=args.include_pitch_bends,
            )

        stats = summarize_note_events(sorted_note_events)
        return TranscriptionRecord(
            input_audio=audio_path,
            midi_path=midi_path,
            note_events_path=note_events_path,
            status="ok",
            **stats,
        )
    except Exception as exc:  # pragma: no cover - useful for batch robustness
        LOGGER.exception("Failed to transcribe %s", audio_path)
        return TranscriptionRecord(
            input_audio=audio_path,
            midi_path=midi_path,
            note_events_path=note_events_path,
            note_count=0,
            duration_s=0.0,
            min_pitch=None,
            max_pitch=None,
            mean_pitch=None,
            mean_velocity=None,
            status="error",
            error=str(exc),
        )


def sort_note_events(note_events: Iterable[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return sorted(
        note_events,
        key=lambda event: (float(event[0]), float(event[1]), int(event[2])),
    )


def write_note_events(
    note_events: Iterable[tuple[Any, ...]],
    output_path: Path,
    include_pitch_bends: bool,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        fieldnames = ["start_s", "end_s", "duration_s", "pitch_midi", "velocity"]
        if include_pitch_bends:
            fieldnames.append("pitch_bend")
        writer.writerow(fieldnames)

        for event in note_events:
            start_s, end_s, pitch_midi, amplitude, pitch_bends = event
            velocity = int(round(127 * float(amplitude)))
            row = [
                f"{float(start_s):.6f}",
                f"{float(end_s):.6f}",
                f"{float(end_s) - float(start_s):.6f}",
                int(pitch_midi),
                velocity,
            ]
            if include_pitch_bends:
                row.append(" ".join(str(int(value)) for value in pitch_bends or []))
            writer.writerow(row)


def summarize_note_events(note_events: list[tuple[Any, ...]]) -> dict[str, Any]:
    if not note_events:
        return {
            "note_count": 0,
            "duration_s": 0.0,
            "min_pitch": None,
            "max_pitch": None,
            "mean_pitch": None,
            "mean_velocity": None,
        }

    pitches = [int(event[2]) for event in note_events]
    velocities = [127 * float(event[3]) for event in note_events]
    duration_s = max(float(event[1]) for event in note_events)
    return {
        "note_count": len(note_events),
        "duration_s": duration_s,
        "min_pitch": min(pitches),
        "max_pitch": max(pitches),
        "mean_pitch": sum(pitches) / len(pitches),
        "mean_velocity": sum(velocities) / len(velocities),
    }


def read_note_events(note_events_path: Path) -> list[tuple[Any, ...]]:
    events: list[tuple[Any, ...]] = []
    with note_events_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            velocity = float(row["velocity"]) / 127
            events.append(
                (
                    float(row["start_s"]),
                    float(row["end_s"]),
                    int(row["pitch_midi"]),
                    velocity,
                    [],
                )
            )
    return events


def write_summary(records: list[TranscriptionRecord], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "input_audio",
                "midi_path",
                "note_events_path",
                "status",
                "note_count",
                "duration_s",
                "min_pitch",
                "max_pitch",
                "mean_pitch",
                "mean_velocity",
                "error",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "input_audio": str(record.input_audio),
                    "midi_path": str(record.midi_path),
                    "note_events_path": (
                        "" if record.note_events_path is None else str(record.note_events_path)
                    ),
                    "status": record.status,
                    "note_count": record.note_count,
                    "duration_s": f"{record.duration_s:.3f}",
                    "min_pitch": "" if record.min_pitch is None else record.min_pitch,
                    "max_pitch": "" if record.max_pitch is None else record.max_pitch,
                    "mean_pitch": (
                        "" if record.mean_pitch is None else f"{record.mean_pitch:.2f}"
                    ),
                    "mean_velocity": (
                        "" if record.mean_velocity is None else f"{record.mean_velocity:.2f}"
                    ),
                    "error": record.error,
                }
            )


def write_pair_summary(
    records: list[TranscriptionRecord],
    pair_summary_path: Path,
    expected_repeats: int,
    note_tolerance: int,
    duration_tolerance_s: float,
) -> None:
    if expected_repeats <= 1:
        return

    pair_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with pair_summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "group",
                "status",
                "files",
                "note_counts",
                "note_count_delta",
                "durations_s",
                "duration_delta_s",
                "pitch_ranges",
                "mean_pitches",
                "mean_pitch_delta",
            ],
        )
        writer.writeheader()

        for group_start in range(0, len(records), expected_repeats):
            group = records[group_start : group_start + expected_repeats]
            group_number = group_start // expected_repeats + 1

            if len(group) != expected_repeats:
                writer.writerow(
                    {
                        "group": group_number,
                        "status": "incomplete",
                        "files": " ".join(record.input_audio.name for record in group),
                    }
                )
                continue

            note_counts = [record.note_count for record in group]
            durations = [record.duration_s for record in group]
            mean_pitches = [
                record.mean_pitch for record in group if record.mean_pitch is not None
            ]
            note_count_delta = max(note_counts) - min(note_counts)
            duration_delta_s = max(durations) - min(durations)
            mean_pitch_delta = (
                max(mean_pitches) - min(mean_pitches) if len(mean_pitches) == len(group) else None
            )
            status = (
                "check"
                if note_count_delta > note_tolerance
                or duration_delta_s > duration_tolerance_s
                else "ok"
            )

            writer.writerow(
                {
                    "group": group_number,
                    "status": status,
                    "files": " ".join(record.input_audio.name for record in group),
                    "note_counts": " ".join(str(value) for value in note_counts),
                    "note_count_delta": note_count_delta,
                    "durations_s": " ".join(f"{value:.3f}" for value in durations),
                    "duration_delta_s": f"{duration_delta_s:.3f}",
                    "pitch_ranges": " ".join(
                        format_pitch_range(record) for record in group
                    ),
                    "mean_pitches": " ".join(
                        "" if record.mean_pitch is None else f"{record.mean_pitch:.2f}"
                        for record in group
                    ),
                    "mean_pitch_delta": (
                        "" if mean_pitch_delta is None else f"{mean_pitch_delta:.2f}"
                    ),
                }
            )


def format_pitch_range(record: TranscriptionRecord) -> str:
    if record.min_pitch is None or record.max_pitch is None:
        return ""
    return f"{record.min_pitch}-{record.max_pitch}"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    audio_files = find_audio_files(args.input_dir, args.pattern, args.limit)
    output_dir, note_events_dir = ensure_output_paths(
        args.output_dir,
        args.note_events_dir,
        args.save_note_events,
    )
    summary_path = args.summary or output_dir / "transcription_summary.csv"
    pair_summary_path = args.pair_summary or output_dir / "repeat_pair_summary.csv"

    model_path, model_cls, predict_fn = load_basic_pitch()
    LOGGER.info("Loading Basic Pitch model: %s", model_path)
    model = model_cls(model_path)

    records = [
        transcribe_file(
            audio_path=audio_path,
            output_dir=output_dir,
            note_events_dir=note_events_dir,
            predict_fn=predict_fn,
            model=model,
            args=args,
        )
        for audio_path in audio_files
    ]
    write_summary(records, summary_path)
    write_pair_summary(
        records,
        pair_summary_path,
        expected_repeats=args.expected_repeats,
        note_tolerance=args.pair_note_tolerance,
        duration_tolerance_s=args.pair_duration_tolerance_s,
    )

    ok_count = sum(record.status == "ok" for record in records)
    LOGGER.info("Finished: %d/%d files transcribed.", ok_count, len(records))
    LOGGER.info("Wrote summary: %s", summary_path)
    if args.expected_repeats > 1:
        LOGGER.info("Wrote repeat-pair summary: %s", pair_summary_path)


if __name__ == "__main__":
    main()
