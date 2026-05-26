# Japanese Train Jingle Generation

Final project repo for a music ML pipeline that learns from Japanese train
station departure jingles and generates new symbolic and audio examples.

## Project Structure

```text
.
|-- src/
|   |-- main.py
|   `-- split_audio.py          # Long-audio -> per-jingle WAV clips
|-- data/
|   |-- raw/                    # Put original long MP3/WAV files here
|   |-- split_audio/            # Generated jingle_001.wav, ...
|   |-- midi/                   # Basic Pitch MIDI transcriptions
|   `-- processed/              # Future features, MIDI, tokens, metadata
|-- requirements.txt
|-- requirements-basic-pitch.txt
`-- README.md
```

Generated data folders are ignored by git except for `.gitkeep` placeholders.

## Setup

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

The script can use the bundled ffmpeg binary from `imageio-ffmpeg`, which is in
`requirements.txt`. A system `ffmpeg` on your PATH also works.

## Split A Long Jingle Compilation

Put the source audio in `data/raw/`, then run:

```bash
python src/split_audio.py data/raw/long_jingles.mp3 --output-dir data/split_audio --review-html
```

This creates files like:

```text
data/split_audio/jingle_001.wav
data/split_audio/jingle_002.wav
data/split_audio/manifest.csv
data/split_audio/review.html
```

For a first pass on station-jingle compilations, try:

```bash
python src/split_audio.py data/raw/long_jingles.mp3 ^
  --output-dir data/split_audio ^
  --min-silence-ms 250 ^
  --relative-silence-db -18 ^
  --keep-silence-ms 150 ^
  --min-clip-ms 1200 ^
  --merge-gap-ms 250 ^
  --review-html ^
  --expected-repeats 2
```

On macOS/Linux, replace the Windows line-continuation `^` with `\`.

## Tuning Segmentation

Main parameters:

- `--min-silence-ms`: how long a pause must be to count as a split. Increase
  this if melodies are split in the middle; decrease it if adjacent jingles are
  stuck together.
- `--silence-thresh-dbfs`: absolute threshold, often around `-35` to `-45` for
  clean audio. More negative means stricter silence detection.
- `--relative-silence-db`: automatic threshold based on the input average dBFS.
  The default uses average volume minus 16 dB.
- `--keep-silence-ms`: padding kept around each jingle so note tails are not
  cut off.
- `--merge-gap-ms`: merges tiny gaps that are probably internal rests.
- `--min-clip-ms`: removes clicks, announcements, or fragments that are too
  short to be useful.
- `--expected-repeats`: adds diagnostics for the Flat score pattern where each
  jingle plays twice in a row. The manifest and review page mark each pair.

Useful debugging workflow:

```bash
python src/split_audio.py data/raw/long_jingles.mp3 --dry-run --verbose
python src/split_audio.py data/raw/long_jingles.mp3 --review-html --overwrite
```

Open `data/split_audio/review.html` in a browser to audition every split clip.
Use `data/split_audio/manifest.csv` to find the original timestamps for bad
splits.

## Next Pipeline Steps

After splitting:

1. Run WAV-to-MIDI transcription with Spotify Basic Pitch on `data/split_audio/`.
2. Store MIDI/tokenized symbolic data in `data/processed/`.
3. Train a symbolic model for short melody generation.
4. Train or fine-tune an audio/spectrogram model for continuous generation.
5. Evaluate against simple baselines such as random n-gram symbolic generation
   and pitch/rhythm distribution matching.

## Basic Pitch WAV To MIDI

Spotify Basic Pitch is a Python automatic music transcription tool. Its README
currently lists Python 3.7 through 3.11 as compatible, so use a Python 3.11
environment for this step.

On Windows:

```bash
py -3.11 -m venv .venv-basic-pitch
.venv-basic-pitch\Scripts\activate
python -m pip install -r requirements-basic-pitch.txt
```

Then transcribe all split WAVs:

```bash
python src/transcribe_basic_pitch.py --input-dir data/split_audio --output-dir data/midi
```

Useful smoke test:

```bash
python src/transcribe_basic_pitch.py --input-dir data/split_audio --output-dir data/midi --limit 3 --overwrite
```

Outputs:

```text
data/midi/jingle_001.mid
data/midi/jingle_002.mid
data/midi/note_events/jingle_001_notes.csv
data/midi/transcription_summary.csv
data/midi/repeat_pair_summary.csv
```

For cleaner symbolic training data, inspect `transcription_summary.csv` and the
per-file note-event CSVs. Because each jingle appears twice in a row, also
inspect `repeat_pair_summary.csv`; pairs with status `check` deserve listening
or manual note review. Useful cleanup knobs:

- Increase `--onset-threshold` to remove extra weak notes.
- Increase `--frame-threshold` to shorten/remove sustained low-confidence notes.
- Increase `--minimum-note-length-ms` to remove tiny ornaments or artifacts.
- Use `--minimum-frequency` and `--maximum-frequency` to restrict the piano
  range if Basic Pitch invents very low or very high notes.
- Keep pitch bends out of symbolic training at first. The note-event CSV omits
  them by default; add `--include-pitch-bends` only if you decide to model them.
