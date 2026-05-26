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
|   `-- processed/              # Future features, MIDI, tokens, metadata
|-- requirements.txt
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
