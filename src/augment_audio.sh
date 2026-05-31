source="flat"

python3 src/augment_pitch_shift.py \
 --input-dir data/audio/processed/$source \
 --output-dir data/audio/augmented/pitch_shift_down_4/$source \
 --semitones -4.0

python3 src/augment_pitch_shift.py \
 --input-dir data/audio/processed/$source \
 --output-dir data/audio/augmented/pitch_shift_down_8/$source \
 --semitones -8.0

python3 src/augment_speed_up.py \
 --input-dir data/audio/processed/$source \
 --output-dir data/audio/augmented/speed_up_125/$source \
 --speed-factor 1.25 --overwrite