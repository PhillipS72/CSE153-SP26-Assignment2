from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import seaborn as sns

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mido

sns.set_theme(style="whitegrid", palette="pastel")


LOGGER = logging.getLogger("pitch_histogram")

PITCH_CLASS_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass(frozen=True)
class PitchHistogramRecord:
    source_group: str
    midi_file: Path
    note_count: int
    min_pitch: int | None
    max_pitch: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create pitch histograms from MIDI files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/midi"),
        help="Folder containing MIDI files. Subfolders are processed as groups.",
    )
    parser.add_argument(
        "--pattern",
        default="*.mid",
        help="Glob pattern for MIDI files inside each folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pitch_histograms"),
        help="Folder where histogram PNGs and CSV summaries will be written.",
    )
    parser.add_argument(
        "--group-by-subdir",
        action="store_true",
        default=True,
        help="Create one histogram per immediate subfolder under --input-dir.",
    )
    parser.add_argument(
        "--no-group-by-subdir",
        action="store_false",
        dest="group_by_subdir",
        help="Treat --input-dir itself as the only group.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug-level progress information.",
    )
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def find_midi_files(folder: Path, pattern: str) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.glob(pattern) if path.is_file())


def collect_groups(input_dir: Path, pattern: str, group_by_subdir: bool) -> dict[str, list[Path]]:
    if group_by_subdir:
        groups = {
            child.name: find_midi_files(child, pattern)
            for child in sorted(path for path in input_dir.iterdir() if path.is_dir())
        }
        return {name: paths for name, paths in groups.items() if paths}

    files = find_midi_files(input_dir, pattern)
    return {input_dir.name: files} if files else {}


def extract_note_pitches(midi_path: Path) -> list[int]:
    midi = mido.MidiFile(midi_path)
    pitches: list[int] = []
    for track in midi.tracks:
        for message in track:
            if message.type == "note_on" and message.velocity > 0:
                pitches.append(int(message.note))
    return pitches


def summarize_group(group_name: str, midi_files: list[Path]) -> tuple[list[int], list[PitchHistogramRecord]]:
    all_pitches: list[int] = []
    records: list[PitchHistogramRecord] = []

    for midi_file in midi_files:
        pitches = extract_note_pitches(midi_file)
        all_pitches.extend(pitches)
        records.append(
            PitchHistogramRecord(
                source_group=group_name,
                midi_file=midi_file,
                note_count=len(pitches),
                min_pitch=min(pitches) if pitches else None,
                max_pitch=max(pitches) if pitches else None,
            )
        )

    return all_pitches, records


def plot_histograms(group_name: str, pitches: list[int], output_dir: Path) -> None:
    if not pitches:
        raise ValueError(f"No note pitches found for group {group_name}")

    pitch_counts = Counter(pitches)
    pitch_class_counts = Counter(pitch % 12 for pitch in pitches)

    min_pitch = min(pitch_counts)
    max_pitch = max(pitch_counts)
    bins = list(range(min_pitch, max_pitch + 2))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

    sns.histplot(pitches, bins=bins, color="#4C72B0", edgecolor="black", ax=axes[0])
    axes[0].set_title(f"Pitch Histogram - {group_name}")
    axes[0].set_xlabel("MIDI note number")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks(range(min_pitch, max_pitch + 1, max(1, (max_pitch - min_pitch) // 12 or 1)))

    class_labels = PITCH_CLASS_LABELS
    class_values = [pitch_class_counts[i] for i in range(12)]
    sns.barplot(x=class_labels, y=class_values, color="#55A868", edgecolor="black", ax=axes[1])
    axes[1].set_title(f"Pitch Class Histogram - {group_name}")
    axes[1].set_xlabel("Pitch class")
    axes[1].set_ylabel("Count")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{group_name}_pitch_histogram.png", dpi=200)
    plt.close(fig)


def write_summary_csv(records: list[PitchHistogramRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["source_group", "midi_file", "note_count", "min_pitch", "max_pitch"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_group": record.source_group,
                    "midi_file": str(record.midi_file),
                    "note_count": record.note_count,
                    "min_pitch": "" if record.min_pitch is None else record.min_pitch,
                    "max_pitch": "" if record.max_pitch is None else record.max_pitch,
                }
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    groups = collect_groups(args.input_dir, args.pattern, args.group_by_subdir)
    if not groups:
        raise FileNotFoundError(f"No MIDI files found under {args.input_dir}")

    all_records: list[PitchHistogramRecord] = []
    for group_name, midi_files in groups.items():
        LOGGER.info("Processing %s (%d files)", group_name, len(midi_files))
        pitches, records = summarize_group(group_name, midi_files)
        plot_histograms(group_name, pitches, args.output_dir)
        all_records.extend(records)

    write_summary_csv(all_records, args.output_dir / "pitch_histogram_summary.csv")
    LOGGER.info("Wrote histograms and summary to %s", args.output_dir)


if __name__ == "__main__":
    main()