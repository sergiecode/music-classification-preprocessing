# Music Classification Data

This directory contains the data for the music classification preprocessing pipeline.

## Directory Structure

- **raw_audio/**: Place your original audio files here
  - Supported formats: .wav, .mp3, .flac, .m4a, .ogg, .aiff
  - Organize by genre/class if needed (e.g., raw_audio/rock/, raw_audio/jazz/)

- **processed/**: Output directory for processed data
  - **features/**: Extracted musical features in JSON format
  - **spectrograms/**: Generated mel-spectrograms in NumPy format
  - **logs/**: Processing logs and statistics

- **examples/**: Sample audio files for testing and demonstration
  - Small audio files for testing the pipeline
  - Use these to verify everything works before processing your full dataset

## Getting Started

1. **Add your audio files** to the `raw_audio/` directory
2. **Run the preprocessing pipeline** using the CLI or Python scripts
3. **Find processed data** in the `processed/` directory

## Example Usage

```bash
# Process all files in raw_audio directory
python -m src.cli preprocess-dataset data/raw_audio/ --output data/processed/ --features --spectrograms

# Process specific files
python -m src.cli extract-features data/raw_audio/song.mp3 --output data/processed/features/

# Generate spectrograms
python -m src.cli generate-spectrograms data/raw_audio/ --output data/processed/spectrograms/
```

## Data Organization Tips

- **By Genre**: Organize audio files in subdirectories by genre for classification tasks
- **By Artist**: Group by artist for artist identification tasks  
- **By Mood**: Organize by mood/emotion for emotion recognition tasks
- **Flat Structure**: Keep all files in one directory for general feature extraction

## File Naming

The pipeline preserves original filenames and adds appropriate suffixes:
- Features: `original_filename_features.json`
- Spectrograms: `original_filename_spectrogram.npy`
- Images: `original_filename_spectrogram.png`
