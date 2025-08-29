# Music Classification Preprocessing Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive audio preprocessing pipeline designed specifically for music classification tasks. This repository serves as the foundation for building machine learning models that can classify music by genre, mood, instrument, or other musical characteristics.

## 🎵 What This Project Does

This preprocessing pipeline transforms raw audio files into machine learning-ready features through:

- **Audio Loading & Normalization**: Robust loading of various audio formats with proper sampling rate handling
- **Feature Extraction**: Extraction of musical features including:
  - Tempo and beat tracking
  - Key and tonal features
  - Spectral features (MFCCs, chroma, spectral contrast)
  - Rhythmic features
- **Mel-Spectrogram Generation**: Time-frequency representations optimized for deep learning models
- **Data Pipeline**: Scalable processing of large audio datasets
- **Visualization Tools**: Audio waveform and spectrogram visualization utilities

This repository is the first part of a three-part music classification system:
1. **music-preprocessing** (this repo) - Data preprocessing and feature extraction
2. **music-classification-model** (future) - Model training and evaluation
3. **music-classification-api** (future) - Production API for real-time classification

## 🎼 How Audio Preprocessing Works

### Audio Loading Process
```
Raw Audio Files → Standardized Format → Feature Extraction → ML-Ready Data
```

1. **Audio Normalization**: 
   - Convert to mono (single channel)
   - Standardize sample rate (22,050 Hz default)
   - Normalize amplitude levels

2. **Feature Extraction Pipeline**:
   - **Temporal Features**: Tempo, beat positions, onset detection
   - **Spectral Features**: MFCCs (13 coefficients), spectral centroid, rolloff, zero-crossing rate
   - **Harmonic Features**: Chroma features, key estimation, tonal centroid
   - **Rhythmic Features**: Tempogram, beat histogram

3. **Mel-Spectrogram Generation**:
   - Convert audio to time-frequency representation
   - Apply mel-scale for perceptual relevance
   - Generate spectrograms suitable for CNN input

### Feature Categories

| Feature Type | Description | Use Case |
|--------------|-------------|----------|
| **MFCCs** | Mel-frequency cepstral coefficients | Genre classification, timbre analysis |
| **Chroma** | Pitch class profiles | Key detection, harmonic analysis |
| **Spectral** | Frequency domain characteristics | Instrument recognition, audio texture |
| **Tempo** | Beat and rhythm information | Dance music classification, mood detection |
| **Mel-Spectrograms** | Time-frequency images | Deep learning input, visual pattern recognition |

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Windows PowerShell (for Windows users)
- At least 4GB RAM recommended for processing large audio files

### Step 1: Clone the Repository
```powershell
git clone https://github.com/yourusername/music-preprocessing.git
cd music-preprocessing
```

### Step 2: Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Verify Installation
```powershell
python -c "import librosa; print('Librosa version:', librosa.__version__)"
```

## 📖 Usage Examples

### Basic Feature Extraction

```python
from src.audio_loader import AudioLoader
from src.feature_extractor import FeatureExtractor

# Load an audio file
loader = AudioLoader()
audio_data, sample_rate = loader.load_audio("path/to/your/song.mp3")

# Extract features
extractor = FeatureExtractor()
features = extractor.extract_all_features(audio_data, sample_rate)

print(f"Extracted features: {list(features.keys())}")
```

### Generate Mel-Spectrograms

```python
from src.spectrogram_generator import SpectrogramGenerator

# Create mel-spectrogram
spec_gen = SpectrogramGenerator()
mel_spec = spec_gen.generate_mel_spectrogram(
    audio_data, 
    sample_rate,
    n_mels=128,
    hop_length=512
)

# Save as image for visualization
spec_gen.save_spectrogram_image(mel_spec, "output/spectrogram.png")
```

### Batch Processing

```python
from src.batch_processor import BatchProcessor

# Process entire directory
processor = BatchProcessor("data/raw_audio", "data/processed")
processor.process_directory(
    extract_features=True,
    generate_spectrograms=True,
    save_format="both"  # both features and spectrograms
)
```

### Command Line Usage

```powershell
# Extract features from a single file
python -m src.cli extract-features "data/raw_audio/song.mp3" --output "data/features/"

# Generate mel-spectrograms for all files in a directory
python -m src.cli generate-spectrograms "data/raw_audio/" --output "data/spectrograms/" --n-mels 128

# Full preprocessing pipeline
python -m src.cli preprocess-dataset "data/raw_audio/" --output "data/processed/" --features --spectrograms
```

## 📁 Project Structure

```
music-preprocessing/
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── audio_loader.py          # Audio file loading utilities
│   ├── feature_extractor.py     # Musical feature extraction
│   ├── spectrogram_generator.py # Mel-spectrogram generation
│   ├── batch_processor.py       # Batch processing utilities
│   ├── cli.py                   # Command-line interface
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       ├── audio_utils.py       # Audio processing helpers
│       └── visualization.py     # Plotting and visualization
│
├── data/                        # Data directory
│   ├── raw_audio/              # Original audio files
│   ├── processed/              # Processed features and spectrograms
│   └── examples/               # Sample audio files for testing
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_audio_exploration.ipynb      # Audio data exploration
│   ├── 02_feature_analysis.ipynb       # Feature extraction analysis
│   └── 03_spectrogram_visualization.ipynb # Spectrogram visualization
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_audio_loader.py
│   ├── test_feature_extractor.py
│   └── test_spectrogram_generator.py
│
├── .github/                     # GitHub configuration
│   └── copilot-instructions.md
│
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── README.md                    # This file
└── LICENSE                      # MIT License
```

## 🎯 Example Workflows

### 1. Music Genre Classification Prep
```powershell
# Organize your music by genre in data/raw_audio/
# data/raw_audio/rock/
# data/raw_audio/jazz/
# data/raw_audio/classical/

python -m src.cli preprocess-dataset "data/raw_audio/" --output "data/genre_features/" --features
```

### 2. Deep Learning Dataset Creation
```powershell
# Generate mel-spectrograms for CNN training
python -m src.cli generate-spectrograms "data/raw_audio/" --output "data/spectrograms/" --n-mels 128 --fixed-length 3.0
```

### 3. Feature Analysis and Visualization
```powershell
# Open the analysis notebook
jupyter notebook notebooks/02_feature_analysis.ipynb
```

## 🔬 Technical Details

### Audio Processing Parameters
- **Sample Rate**: 22,050 Hz (configurable)
- **Audio Format**: Mono conversion applied
- **Normalization**: Peak normalization to [-1, 1]

### Feature Extraction Settings
- **MFCCs**: 13 coefficients, 2048 FFT window
- **Chroma**: 12 pitch classes
- **Mel-Spectrograms**: 128 mel bands, 512 hop length
- **Tempo**: Beat tracking with dynamic programming

### Performance Considerations
- **Memory Usage**: ~50MB per minute of audio (approximate)
- **Processing Speed**: ~10-15 files per minute (depends on hardware)
- **Batch Processing**: Recommended for datasets >100 files

## 🛠️ Configuration

Create a `config.yaml` file to customize processing parameters:

```yaml
audio:
  sample_rate: 22050
  duration: null  # null for full length, or seconds for fixed length
  
features:
  mfccs: 13
  chroma: 12
  spectral_contrast_bands: 6
  
spectrograms:
  n_mels: 128
  hop_length: 512
  win_length: 2048
```

## 🧪 Testing

Run the test suite to verify your installation:

```powershell
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_feature_extractor.py -v

# Run with coverage
pytest --cov=src tests/
```

## 🔮 Future Integration

This preprocessing pipeline is designed to integrate seamlessly with:

1. **Music Classification Model Repository**:
   - Processed features will feed directly into training scripts
   - Standardized data format ensures compatibility
   - Feature configurations match model input requirements

2. **Music Classification API Repository**:
   - Same preprocessing pipeline will be used for real-time inference
   - Docker containers will include this preprocessing code
   - API endpoints will utilize the same feature extraction methods

## 📊 Example Output

After processing, you'll have structured data ready for machine learning:

```
data/processed/
├── features/
│   ├── song1_features.json
│   └── song2_features.json
└── spectrograms/
    ├── song1_spectrogram.npy
    └── song2_spectrogram.npy
```

Each feature file contains:
```json
{
  "filename": "song1.mp3",
  "duration": 180.5,
  "sample_rate": 22050,
  "tempo": 120.0,
  "key": "C major",
  "mfccs": [13 coefficient arrays],
  "chroma": [12 pitch class arrays],
  "spectral_features": {...},
  "metadata": {...}
}
```

## 🤝 Contributing

This project is part of Sergie Code's educational content for AI tools for musicians. Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request



## 👨‍💻 About

Created by **Sergie Code** - Software Engineer and YouTube educator specializing in AI tools for musicians.

- 📸 Instagram: https://www.instagram.com/sergiecode

- 🧑🏼‍💻 LinkedIn: https://www.linkedin.com/in/sergiecode/

- 📽️Youtube: https://www.youtube.com/@SergieCode

- 😺 Github: https://github.com/sergiecode

- 👤 Facebook: https://www.facebook.com/sergiecodeok

- 🎞️ Tiktok: https://www.tiktok.com/@sergiecode

- 🕊️Twitter: https://twitter.com/sergiecode

- 🧵Threads: https://www.threads.net/@sergiecode

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/sergiecode/music-preprocessing/issues) page
2. Review the example notebooks in the `notebooks/` directory
3. Join the discussion in the YouTube comments
4. Create a new issue with a detailed description

## Testing

The project includes a comprehensive test suite with 80+ tests covering all major functionality.

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_audio_loader.py -v
python -m pytest tests/test_feature_extractor.py -v
python -m pytest tests/test_spectrogram_generator.py -v
python -m pytest tests/test_batch_processor.py -v
python -m pytest tests/test_cli.py -v

# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

- **AudioLoader**: 100% test coverage - All functionality tested
- **FeatureExtractor**: 95+ test coverage - Comprehensive feature testing  
- **SpectrogramGenerator**: 90+ test coverage - All spectrogram operations tested
- **BatchProcessor**: 100% test coverage - Batch operations fully tested
- **CLI**: 100% test coverage - All CLI commands tested
- **Integration**: Complete pipeline testing

### Manual Testing

Test the application with real audio files:

```bash
# Create a test audio file (or use your own)
python -c "
import numpy as np
import soundfile as sf
from pathlib import Path

test_dir = Path('data/test')
test_dir.mkdir(parents=True, exist_ok=True)

duration = 3.0
sample_rate = 22050
t = np.linspace(0, duration, int(duration * sample_rate))
audio = 0.5 * np.sin(2 * np.pi * 440 * t)

test_file = test_dir / 'test_audio.wav'
sf.write(str(test_file), audio, sample_rate)
print(f'Created: {test_file}')
"

# Test feature extraction
python src/cli.py extract-features data/test/test_audio.wav --output data/output --summary

# Test spectrogram generation  
python src/cli.py generate-spectrograms data/test/test_audio.wav --output data/output --save-image

# Test file info
python src/cli.py info data/test/test_audio.wav

# Test complete pipeline
python src/cli.py preprocess-dataset data/test --output data/processed --features --spectrograms --manifest test_dataset --stats
```

---

**Next Steps**: After setting up this preprocessing pipeline, you'll be ready to move on to the model training repository where these features will be used to train classification models, followed by the API repository for production deployment.
