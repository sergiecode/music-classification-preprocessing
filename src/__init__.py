"""
Music Classification Preprocessing Pipeline

A comprehensive audio preprocessing pipeline for music classification tasks.
Provides tools for audio loading, feature extraction, and mel-spectrogram generation.
"""

from .audio_loader import AudioLoader
from .feature_extractor import FeatureExtractor
from .spectrogram_generator import SpectrogramGenerator
from .batch_processor import BatchProcessor

__version__ = "1.0.0"
__author__ = "Sergie Code"
__email__ = "sergieCode@example.com"

__all__ = [
    "AudioLoader",
    "FeatureExtractor", 
    "SpectrogramGenerator",
    "BatchProcessor"
]
