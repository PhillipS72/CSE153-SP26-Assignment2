from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import librosa

from librosa_audio_utils import configure_logging, find_audio_files


@dataclass(frozen=True)
class MelodyScanRecord:
    file_number: str
    audio_file: str
    duration_s: float
    onset_count: int
    onset_density: float
    rms_mean: float
    flagged: bool
    reason: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan Dropbox source MP3s for short or sparse melodies using librosa onset analysis."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/dropbox"),
        help="Folder containing the Dropbox source MP3 files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.mp3",
        help="Glob pattern for source audio files.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=32000,
        help="Sample rate to use while analyzing the MP3 files.",
    )
    parser.add_argument(
        "--short-seconds-threshold",
        type=float,
        default=8.0,
        help="Flag files shorter than this many seconds.",
    )
    parser.add_argument(
        "--low-onset-threshold",
        type=int,
        default=6,
        help="Flag files with no more than this many detected onsets.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/raw/dropbox/sparse_melodies.csv"),
        help="CSV file where the scan results will be written.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress while scanning files.",
    )
    return parser


def analyze_file(
    audio_path: Path,
    sample_rate: int,
    short_seconds_threshold: float,
    low_onset_threshold: int,
) -> MelodyScanRecord:
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration_s = len(audio) / sample_rate
    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=sample_rate,
        units="frames",
        backtrack=True,
    )
    onset_count = int(len(onset_frames))
    onset_density = onset_count / duration_s if duration_s > 0 else 0.0

    rms = librosa.feature.rms(y=audio)[0]
    rms_mean = float(rms.mean()) if rms.size else 0.0

    reasons: list[str] = []
    if duration_s <= short_seconds_threshold:
        reasons.append(f"shorter_than_{short_seconds_threshold:.1f}s")
    if onset_count <= low_onset_threshold:
        reasons.append(f"onsets_at_or_below_{low_onset_threshold}")

    return MelodyScanRecord(
        file_number=audio_path.stem,
        audio_file=audio_path.name,
        duration_s=duration_s,
        onset_count=onset_count,
        onset_density=onset_density,
        rms_mean=rms_mean,
        flagged=bool(reasons),
        reason=";".join(reasons),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    audio_files = find_audio_files(args.input_dir, args.pattern)
    records: list[MelodyScanRecord] = []

    for audio_path in audio_files:
        if args.verbose:
            print(f"Scanning {audio_path.name}...")
        record = analyze_file(
            audio_path=audio_path,
            sample_rate=args.sample_rate,
            short_seconds_threshold=args.short_seconds_threshold,
            low_onset_threshold=args.low_onset_threshold,
        )
        records.append(record)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file_number",
                "audio_file",
                "duration_s",
                "onset_count",
                "onset_density",
                "rms_mean",
                "flagged",
                "reason",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "file_number": record.file_number,
                    "audio_file": record.audio_file,
                    "duration_s": f"{record.duration_s:.3f}",
                    "onset_count": record.onset_count,
                    "onset_density": f"{record.onset_density:.3f}",
                    "rms_mean": f"{record.rms_mean:.6f}",
                    "flagged": str(record.flagged).lower(),
                    "reason": record.reason,
                }
            )

    flagged_records = sorted(
        (record for record in records if record.flagged),
        key=lambda record: (record.duration_s, record.onset_count, record.file_number),
    )

    print(f"Wrote scan results to {args.output_csv}")
    print("Flagged candidates:")
    for record in flagged_records[:25]:
        print(
            f"{record.audio_file}: duration={record.duration_s:.3f}s, onsets={record.onset_count}, density={record.onset_density:.3f}, reason={record.reason}"
        )


if __name__ == "__main__":
    main()