# Music Classification Pipeline Integration Guide

**Author**: Sergie Code - Software Engineer & YouTube Programming Educator  
**Project**: AI Tools for Musicians  
**Date**: August 29, 2025  
**Purpose**: Integration specifications for music-classification-model and music-classification-api repositories

---

## 🎯 Project Overview

This integration guide provides specifications for connecting the **music-classification-preprocessing** repository with two additional repositories:

1. **music-classification-model**: PyTorch-based CNN/RNN models for automatic music tagging
2. **music-classification-api**: Lightweight REST API for real-time music classification

## 📊 Data Flow Architecture

```mermaid
graph LR
    A[Audio Files] --> B[Preprocessing Pipeline]
    B --> C[Features & Spectrograms]
    C --> D[Model Training]
    D --> E[Trained Models]
    E --> F[REST API]
    F --> G[Real-time Predictions]
```

## 🔧 Preprocessing Repository Output Specifications

### Feature Output Format
The preprocessing pipeline generates standardized outputs that must be consumed by the model repository:

#### 1. Feature Files (JSON Format)
**Location**: `data/processed/features/`  
**Format**: `{filename}_features.json`

```json
{
  "filename": "audio_file.wav",
  "duration": 180.5,
  "sample_rate": 22050,
  "processing_info": {
    "librosa_version": "0.10.1",
    "processed_at": "2025-08-29T12:00:00Z"
  },
  
  // Temporal Features (9 features)
  "tempo": 120.5,
  "onset_rate": 2.3,
  "zero_crossing_rate_mean": 0.15,
  "zero_crossing_rate_std": 0.08,
  
  // Spectral Features (46 features)
  "spectral_centroid_mean": 2048.5,
  "spectral_centroid_std": 512.3,
  "spectral_bandwidth_mean": 1024.7,
  "spectral_rolloff_mean": 4096.2,
  "mfcc_1_mean": -125.8,
  "mfcc_1_std": 45.2,
  // ... mfcc_1 through mfcc_13 (mean/std)
  
  // Harmonic Features (38 features)
  "chroma_1_mean": 0.25,
  "chroma_1_std": 0.12,
  // ... chroma_1 through chroma_12 (mean/std)
  "key_clarity": 0.75,
  "tonal_centroid_1": 0.85,
  
  // Rhythmic Features (4 features)
  "rhythm_complexity": 0.65,
  "beat_strength": 0.78,
  
  // Statistical Features (7 features)
  "energy_mean": 0.45,
  "rms_energy_mean": 0.35
}
```

#### 2. Spectrogram Files (NumPy Format)
**Location**: `data/processed/spectrograms/`  
**Format**: `{filename}_spectrogram.npy`

```python
# Spectrogram specifications
spectrogram_shape = (128, variable_time_frames)  # Default n_mels=128
data_type = np.float32
scale = "mel"  # Mel-scale frequency bins
power = 2.0   # Power spectrogram
hop_length = 512  # Time resolution
sample_rate = 22050  # Audio sample rate

# Load spectrogram
import numpy as np
spectrogram = np.load("filename_spectrogram.npy")
# Shape: (n_mels, time_frames)
```

#### 3. Dataset Manifest (JSON Format)
**Location**: `data/processed/{dataset_name}_manifest.json`

```json
{
  "dataset_name": "music_dataset",
  "created_by": "music-classification-preprocessing",
  "total_files": 1000,
  "total_errors": 5,
  "processing_date": "2025-08-29T12:00:00Z",
  "statistics": {
    "total_duration_hours": 45.5,
    "average_duration_seconds": 164.2,
    "sample_rates": [22050, 44100],
    "file_formats": ["wav", "mp3", "flac"]
  },
  "files": [
    {
      "filename": "song1.wav",
      "duration": 180.5,
      "sample_rate": 22050,
      "features_file": "data/processed/features/song1_features.json",
      "spectrogram_file": "data/processed/spectrograms/song1_spectrogram.npy",
      "file_size_mb": 15.2,
      "genre": "rock",  // If available
      "mood": "energetic",  // If available
      "bpm": 120  // If available
    }
  ]
}
```

## 🤖 Model Repository Specifications (music-classification-model)

### Technology Stack Requirements
- **Framework**: PyTorch 2.0+
- **Python**: 3.9+
- **Dependencies**: torchvision, torchaudio, scikit-learn, numpy, pandas
- **GPU Support**: CUDA-compatible for training acceleration

### Model Architecture Recommendations

#### 1. Spectrogram-based CNN Model
```python
# Expected input shape from preprocessing
input_shape = (1, 128, variable_length)  # (channels, mel_bins, time_frames)

# Model should handle variable length inputs through:
# - Adaptive pooling
# - Attention mechanisms
# - Fixed-length chunking
```

#### 2. Feature-based Models
```python
# Feature vector specifications
feature_vector_size = 103  # Total features from preprocessing
feature_categories = {
    "temporal": 9,
    "spectral": 46, 
    "harmonic": 38,
    "rhythmic": 4,
    "statistical": 7
}
```

### Dataset Loading Integration

#### DataLoader Implementation
```python
import torch
from torch.utils.data import Dataset
import json
import numpy as np

class MusicDataset(Dataset):
    def __init__(self, manifest_file, use_spectrograms=True, use_features=True):
        """
        Args:
            manifest_file: Path to preprocessing manifest JSON
            use_spectrograms: Whether to load spectrogram data
            use_features: Whether to load feature data
        """
        with open(manifest_file, 'r') as f:
            self.manifest = json.load(f)
        
        self.files = self.manifest['files']
        self.use_spectrograms = use_spectrograms
        self.use_features = use_features
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_info = self.files[idx]
        
        data = {}
        
        if self.use_spectrograms:
            # Load spectrogram from preprocessing
            spec_path = file_info['spectrogram_file']
            spectrogram = np.load(spec_path)
            data['spectrogram'] = torch.FloatTensor(spectrogram)
        
        if self.use_features:
            # Load features from preprocessing
            features_path = file_info['features_file']
            with open(features_path, 'r') as f:
                features_dict = json.load(f)
            
            # Extract numeric features (exclude metadata)
            feature_vector = self._extract_feature_vector(features_dict)
            data['features'] = torch.FloatTensor(feature_vector)
        
        # Labels (if available in manifest)
        if 'genre' in file_info:
            data['genre'] = file_info['genre']
        if 'mood' in file_info:
            data['mood'] = file_info['mood']
        if 'bpm' in file_info:
            data['bpm'] = file_info['bpm']
        
        return data
```

### Model Training Pipeline

#### 1. Multi-task Learning Architecture
```python
class MusicClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Spectrogram CNN branch
        self.spectrogram_cnn = SpectrogramCNN()
        
        # Feature MLP branch  
        self.feature_mlp = FeatureMLP(input_size=103)
        
        # Fusion layer
        self.fusion = nn.Linear(cnn_features + mlp_features, 512)
        
        # Task-specific heads
        self.genre_head = nn.Linear(512, num_genres)
        self.mood_head = nn.Linear(512, num_moods)
        self.bpm_head = nn.Linear(512, 1)  # Regression
        self.key_head = nn.Linear(512, 12)  # 12 musical keys
    
    def forward(self, spectrogram, features):
        # Process both input types
        spec_features = self.spectrogram_cnn(spectrogram)
        feat_features = self.feature_mlp(features)
        
        # Fuse features
        combined = torch.cat([spec_features, feat_features], dim=1)
        fused = self.fusion(combined)
        
        # Multi-task outputs
        return {
            'genre': self.genre_head(fused),
            'mood': self.mood_head(fused),
            'bpm': self.bpm_head(fused),
            'key': self.key_head(fused)
        }
```

#### 2. Training Configuration
```python
# Expected preprocessing integration
def load_preprocessed_data(data_dir):
    """
    Load data processed by music-classification-preprocessing
    
    Args:
        data_dir: Directory containing processed data
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    
    # Find manifest files
    manifest_files = list(Path(data_dir).glob("*_manifest.json"))
    
    # Create datasets
    datasets = []
    for manifest_file in manifest_files:
        dataset = MusicDataset(manifest_file)
        datasets.append(dataset)
    
    return datasets
```

### Model Export Specifications

#### 1. Model Serialization
```python
# Required model export format for API integration
def export_model(model, export_path):
    """Export trained model for API serving"""
    
    # Save complete model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model.__class__.__name__,
        'feature_size': 103,
        'spectrogram_shape': (128, None),  # Variable length
        'class_mappings': {
            'genres': ['rock', 'pop', 'jazz', 'classical', ...],
            'moods': ['happy', 'sad', 'energetic', 'calm', ...],
            'keys': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        },
        'preprocessing_config': {
            'sample_rate': 22050,
            'n_mels': 128,
            'hop_length': 512,
            'feature_count': 103
        }
    }, export_path)
```

## 🌐 API Repository Specifications (music-classification-api)

### Technology Stack Requirements
- **Framework**: FastAPI or Flask
- **Python**: 3.9+
- **Dependencies**: torch, librosa, numpy, uvicorn (for FastAPI)
- **Deployment**: Docker, AWS/Azure/GCP compatible

### API Integration Points

#### 1. Preprocessing Integration
```python
# API should integrate preprocessing components
import sys
sys.path.append('path/to/music-classification-preprocessing/src')

from audio_loader import AudioLoader
from feature_extractor import FeatureExtractor
from spectrogram_generator import SpectrogramGenerator

class PreprocessingService:
    def __init__(self):
        self.audio_loader = AudioLoader()
        self.feature_extractor = FeatureExtractor()
        self.spectrogram_generator = SpectrogramGenerator()
    
    def process_audio_file(self, file_path):
        """Process uploaded audio file for prediction"""
        
        # Load audio
        audio_data, sample_rate = self.audio_loader.load_audio(file_path)
        
        # Extract features
        features = self.feature_extractor.extract_all_features(audio_data, sample_rate)
        
        # Generate spectrogram
        spectrogram = self.spectrogram_generator.generate_mel_spectrogram(audio_data, sample_rate)
        
        return {
            'features': self._extract_feature_vector(features),
            'spectrogram': spectrogram,
            'metadata': {
                'duration': features['duration'],
                'sample_rate': features['sample_rate']
            }
        }
```

#### 2. API Endpoints Structure
```python
from fastapi import FastAPI, UploadFile, File
import torch

app = FastAPI(title="Music Classification API")

@app.post("/classify")
async def classify_music(file: UploadFile = File(...)):
    """
    Classify uploaded music file
    
    Returns:
        {
            "genre": {"label": "rock", "confidence": 0.85},
            "mood": {"label": "energetic", "confidence": 0.78},
            "bpm": {"value": 120.5, "confidence": 0.82},
            "key": {"label": "C", "confidence": 0.71},
            "processing_time": 2.3
        }
    """
    
    # Process audio using preprocessing pipeline
    processed_data = preprocessing_service.process_audio_file(file)
    
    # Run model inference
    predictions = model.predict(
        features=processed_data['features'],
        spectrogram=processed_data['spectrogram']
    )
    
    return predictions

@app.post("/batch_classify")
async def batch_classify(files: List[UploadFile]):
    """Batch classification for multiple files"""
    pass

@app.get("/health")
async def health_check():
    """API health check"""
    return {"status": "healthy", "preprocessing": "ready", "model": "loaded"}
```

### Docker Integration
```dockerfile
# API Dockerfile should include preprocessing dependencies
FROM python:3.9-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy preprocessing repository
COPY music-classification-preprocessing/ /app/preprocessing/
COPY music-classification-model/ /app/model/
COPY music-classification-api/ /app/api/

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set environment variables
ENV PYTHONPATH="/app/preprocessing/src:/app/model/src:/app/api/src"

# Run API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔄 Development Workflow

### 1. Data Preparation (Current Repository)
```bash
# Preprocess music dataset
python src/cli.py preprocess-dataset music_data/ --output processed_data/ --features --spectrograms --manifest training_data

# Verify processing
python validate.py
```

### 2. Model Training (music-classification-model)
```bash
# Clone and setup model repository
git clone music-classification-model
cd music-classification-model

# Install dependencies
pip install -r requirements.txt

# Train model using preprocessed data
python train.py --data ../music-classification-preprocessing/processed_data/training_data_manifest.json

# Export trained model
python export_model.py --model_path models/best_model.pth --output api_model.pth
```

### 3. API Deployment (music-classification-api)
```bash
# Clone and setup API repository
git clone music-classification-api
cd music-classification-api

# Copy preprocessing components
cp -r ../music-classification-preprocessing/src/ preprocessing/

# Copy trained model
cp ../music-classification-model/api_model.pth models/

# Run API
uvicorn main:app --reload
```

## 📋 Testing Integration

### End-to-End Testing Pipeline
```python
def test_full_pipeline():
    """Test complete pipeline from preprocessing to API prediction"""
    
    # 1. Test preprocessing
    from music_classification_preprocessing import process_audio
    features, spectrogram = process_audio("test_song.wav")
    
    # 2. Test model prediction
    from music_classification_model import MusicClassifier
    model = MusicClassifier.load("trained_model.pth")
    predictions = model.predict(features, spectrogram)
    
    # 3. Test API endpoint
    import requests
    with open("test_song.wav", "rb") as f:
        response = requests.post("http://localhost:8000/classify", files={"file": f})
    
    assert response.status_code == 200
    assert "genre" in response.json()
```

## 🚀 Production Deployment

### Environment Setup
```bash
# Production environment variables
export PREPROCESSING_PATH="/app/music-classification-preprocessing"
export MODEL_PATH="/app/models/production_model.pth"
export API_PORT=8000
export BATCH_SIZE=32
export MAX_FILE_SIZE=50MB
```

### Performance Considerations
- **Processing Time**: Target <5 seconds per file
- **Memory Usage**: Optimize for 2GB RAM containers
- **Throughput**: Support 100+ requests/hour
- **File Formats**: Support MP3, WAV, FLAC, M4A

## 📝 Integration Checklist

### For Model Repository
- [ ] Import preprocessing components
- [ ] Implement MusicDataset class
- [ ] Create multi-task model architecture
- [ ] Add training pipeline
- [ ] Implement model export functionality
- [ ] Add evaluation metrics
- [ ] Create model documentation

### For API Repository  
- [ ] Integrate preprocessing pipeline
- [ ] Load trained models
- [ ] Implement classification endpoints
- [ ] Add file upload handling
- [ ] Create batch processing endpoints
- [ ] Add error handling
- [ ] Implement health checks
- [ ] Add API documentation
- [ ] Create Docker configuration
- [ ] Add performance monitoring

## 🎓 Educational Content Integration

As this is part of Sergie Code's educational content for YouTube:

### Documentation Requirements
- Clear, step-by-step tutorials
- Code examples with explanations
- Common troubleshooting guides
- Performance optimization tips

### Demo Scripts
- End-to-end pipeline demonstration
- Real-time classification examples
- Batch processing workflows
- Model comparison scripts

---

**This integration guide ensures seamless connectivity between all three repositories in the music classification pipeline, enabling AI agents to create fully compatible model training and API serving repositories.**
