"""
Utility functions for music preprocessing pipeline.
"""

from .audio_utils import (
    normalize_audio,
    trim_silence,
    apply_pre_emphasis,
    split_audio_chunks,
    compute_spectral_features,
    detect_tempo_changes,
    estimate_key_signature,
    compute_audio_fingerprint,
    validate_audio_quality,
    preprocess_audio_standard
)

from .visualization import (
    plot_waveform,
    plot_spectrogram,
    plot_feature_comparison,
    plot_feature_distribution,
    plot_chroma_features,
    plot_tempo_analysis,
    create_feature_correlation_matrix,
    create_audio_analysis_report
)

__all__ = [
    # Audio utilities
    "normalize_audio",
    "trim_silence", 
    "apply_pre_emphasis",
    "split_audio_chunks",
    "compute_spectral_features",
    "detect_tempo_changes",
    "estimate_key_signature",
    "compute_audio_fingerprint",
    "validate_audio_quality",
    "preprocess_audio_standard",
    
    # Visualization utilities
    "plot_waveform",
    "plot_spectrogram",
    "plot_feature_comparison",
    "plot_feature_distribution",
    "plot_chroma_features",
    "plot_tempo_analysis",
    "create_feature_correlation_matrix",
    "create_audio_analysis_report"
]
