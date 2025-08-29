# Quick Setup Context for AI Agents

## Repository 1: music-classification-model
**Goal**: PyTorch-based CNN/RNN models for automatic music tagging

### Key Requirements:
- Use PyTorch 2.0+
- Consume data from music-classification-preprocessing repository
- Implement multi-task learning (genre, mood, BPM, key prediction)
- Handle both spectrograms (128×variable) and feature vectors (103 features)
- Export models compatible with API serving

### Critical Integration:
- Input: 103-feature vectors + mel-spectrograms from preprocessing
- Output: Trained models with class mappings for API
- Dataset: Load from manifest JSON files
- Architecture: CNN for spectrograms + MLP for features + fusion layer

## Repository 2: music-classification-api  
**Goal**: Lightweight REST API for real-time music classification

### Key Requirements:
- Use FastAPI or Flask
- Integrate preprocessing pipeline for real-time audio processing
- Load and serve trained models from model repository
- Handle file uploads (MP3, WAV, FLAC)
- Return JSON predictions with confidence scores

### Critical Integration:
- Import: AudioLoader, FeatureExtractor, SpectrogramGenerator from preprocessing
- Load: Trained models from model repository
- Process: Real-time audio → features/spectrograms → predictions
- Return: {"genre": {"label": "rock", "confidence": 0.85}, ...}

## Author Context:
- **Name**: Sergie Code
- **Role**: Software Engineer & YouTube Programming Educator  
- **Project**: AI Tools for Musicians
- **Platform**: Windows (PowerShell)
- **Style**: Educational, well-documented, production-ready code

## File Structure Reference:
```
music-classification-preprocessing/
├── src/
│   ├── audio_loader.py (103 features extracted)
│   ├── feature_extractor.py (Multi-class capabilities)
│   ├── spectrogram_generator.py (128 mel-bins)
│   └── cli.py (Working CLI interface)
├── data/processed/ (Output location)
└── INTEGRATION_GUIDE.md (Complete specifications)
```

Use INTEGRATION_GUIDE.md for detailed technical specifications.
