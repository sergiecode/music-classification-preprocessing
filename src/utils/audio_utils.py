"""
Audio processing utility functions.

This module provides helper functions for common audio processing tasks
used throughout the music preprocessing pipeline.
"""

import numpy as np
import librosa
from typing import Tuple, Union, Optional
from pathlib import Path


def normalize_audio(audio: np.ndarray, method: str = "peak") -> np.ndarray:
    """
    Normalize audio using different methods.
    
    Args:
        audio: Input audio array
        method: Normalization method ('peak', 'rms', or 'lufs')
        
    Returns:
        Normalized audio array
    """
    if method == "peak":
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    elif method == "rms":
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            return audio / rms
        return audio
    
    elif method == "lufs":
        # Simplified LUFS-style normalization
        # This is a basic implementation, not true LUFS
        target_lufs = -23.0  # EBU R128 recommendation
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms > 0:
            target_rms = 10**(target_lufs / 20)
            return audio * (target_rms / current_rms)
        return audio
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def trim_silence(audio: np.ndarray, 
                sample_rate: int,
                threshold_db: float = -40.0,
                frame_length: int = 2048,
                hop_length: int = 512) -> np.ndarray:
    """
    Trim silence from the beginning and end of audio.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate of audio
        threshold_db: Silence threshold in decibels
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis
        
    Returns:
        Trimmed audio array
    """
    try:
        trimmed_audio, _ = librosa.effects.trim(
            audio,
            top_db=-threshold_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        return trimmed_audio
    except Exception:
        # Return original audio if trimming fails
        return audio


def apply_pre_emphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    Apply pre-emphasis filter to audio.
    
    Args:
        audio: Input audio array
        coeff: Pre-emphasis coefficient
        
    Returns:
        Pre-emphasized audio array
    """
    return np.append(audio[0], audio[1:] - coeff * audio[:-1])


def split_audio_chunks(audio: np.ndarray,
                      sample_rate: int,
                      chunk_duration: float,
                      overlap: float = 0.0) -> list:
    """
    Split audio into fixed-length chunks.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate of audio
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks (0.0 to 1.0)
        
    Returns:
        List of audio chunks
    """
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap * chunk_samples)
    step_size = chunk_samples - overlap_samples
    
    chunks = []
    start = 0
    
    while start + chunk_samples <= len(audio):
        chunk = audio[start:start + chunk_samples]
        chunks.append(chunk)
        start += step_size
    
    # Handle remaining audio if it's significant
    if start < len(audio) and (len(audio) - start) > chunk_samples * 0.5:
        # Pad the last chunk to full length
        last_chunk = audio[start:]
        padding = chunk_samples - len(last_chunk)
        last_chunk = np.pad(last_chunk, (0, padding), mode='constant')
        chunks.append(last_chunk)
    
    return chunks


def compute_spectral_features(audio: np.ndarray,
                            sample_rate: int,
                            n_fft: int = 2048,
                            hop_length: int = 512) -> dict:
    """
    Compute basic spectral features from audio.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate of audio
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        Dictionary of spectral features
    """
    # Compute STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    
    # Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(
        S=magnitude, sr=sample_rate, hop_length=hop_length
    )[0]
    
    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        S=magnitude, sr=sample_rate, hop_length=hop_length
    )[0]
    
    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        S=magnitude, sr=sample_rate, hop_length=hop_length
    )[0]
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
    
    features = {
        'spectral_centroid_mean': float(np.mean(spectral_centroids)),
        'spectral_centroid_std': float(np.std(spectral_centroids)),
        'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
        'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
        'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
        'spectral_rolloff_std': float(np.std(spectral_rolloff)),
        'zcr_mean': float(np.mean(zcr)),
        'zcr_std': float(np.std(zcr))
    }
    
    return features


def detect_tempo_changes(audio: np.ndarray, 
                        sample_rate: int,
                        hop_length: int = 512) -> dict:
    """
    Detect tempo changes throughout the audio.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate of audio
        hop_length: Hop length for analysis
        
    Returns:
        Dictionary with tempo analysis results
    """
    try:
        # Compute tempogram
        onset_envelope = librosa.onset.onset_strength(
            y=audio, sr=sample_rate, hop_length=hop_length
        )
        
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            hop_length=hop_length
        )
        
        # Global tempo
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            hop_length=hop_length
        )
        
        # Local tempo variation
        tempo_variance = np.var(tempogram, axis=1)
        
        results = {
            'global_tempo': float(tempo),
            'beat_count': int(len(beats)),
            'tempo_stability': float(1.0 / (1.0 + np.mean(tempo_variance))),
            'tempo_variance_mean': float(np.mean(tempo_variance)),
            'onset_density': float(len(beats) / (len(audio) / sample_rate))
        }
        
        return results
        
    except Exception:
        return {
            'global_tempo': 0.0,
            'beat_count': 0,
            'tempo_stability': 0.0,
            'tempo_variance_mean': 0.0,
            'onset_density': 0.0
        }


def estimate_key_signature(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Estimate the key signature of the audio.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate of audio
        
    Returns:
        Dictionary with key estimation results
    """
    try:
        # Extract chroma features
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
        
        # Average chroma across time
        chroma_mean = np.mean(chroma, axis=1)
        
        # Key names
        major_keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        minor_keys = ['Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm']
        
        # Simple key estimation based on chroma profile
        # This is a basic implementation
        estimated_root = np.argmax(chroma_mean)
        
        # Heuristic for major/minor detection
        # Check if the third and sixth are prominent
        third_major = (estimated_root + 4) % 12
        third_minor = (estimated_root + 3) % 12
        
        major_strength = chroma_mean[third_major]
        minor_strength = chroma_mean[third_minor]
        
        if major_strength > minor_strength:
            estimated_key = major_keys[estimated_root]
            mode = "major"
        else:
            estimated_key = minor_keys[estimated_root]
            mode = "minor"
        
        # Confidence based on how much the root dominates
        confidence = float(chroma_mean[estimated_root] / np.sum(chroma_mean))
        
        results = {
            'estimated_key': estimated_key,
            'mode': mode,
            'confidence': confidence,
            'root_note': major_keys[estimated_root],
            'chroma_profile': chroma_mean.tolist()
        }
        
        return results
        
    except Exception:
        return {
            'estimated_key': 'C',
            'mode': 'major',
            'confidence': 0.0,
            'root_note': 'C',
            'chroma_profile': [0.0] * 12
        }


def compute_audio_fingerprint(audio: np.ndarray, 
                            sample_rate: int,
                            n_features: int = 64) -> np.ndarray:
    """
    Compute a compact audio fingerprint for similarity comparison.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate of audio
        n_features: Number of features in the fingerprint
        
    Returns:
        Audio fingerprint array
    """
    try:
        # Extract various features for fingerprint
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        
        # Compute means and standard deviations
        features = []
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.mean(spectral_contrast, axis=1))
        
        # Pad or truncate to desired size
        features = np.array(features)
        if len(features) > n_features:
            features = features[:n_features]
        elif len(features) < n_features:
            features = np.pad(features, (0, n_features - len(features)), mode='constant')
        
        # Normalize fingerprint
        if np.std(features) > 0:
            features = (features - np.mean(features)) / np.std(features)
        
        return features
        
    except Exception:
        return np.zeros(n_features)


def validate_audio_quality(audio: np.ndarray, 
                          sample_rate: int,
                          min_duration: float = 1.0,
                          max_silence_ratio: float = 0.8) -> dict:
    """
    Validate audio quality and detect potential issues.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate of audio
        min_duration: Minimum required duration in seconds
        max_silence_ratio: Maximum allowed silence ratio
        
    Returns:
        Dictionary with validation results
    """
    duration = len(audio) / sample_rate
    
    # Check duration
    duration_ok = duration >= min_duration
    
    # Check for clipping
    clipping_ratio = np.sum(np.abs(audio) >= 0.99) / len(audio)
    clipping_ok = clipping_ratio < 0.01  # Less than 1% clipping
    
    # Check for silence
    silence_threshold = 0.01
    silence_ratio = np.sum(np.abs(audio) < silence_threshold) / len(audio)
    silence_ok = silence_ratio < max_silence_ratio
    
    # Check dynamic range
    dynamic_range = np.max(audio) - np.min(audio)
    dynamic_range_ok = dynamic_range > 0.1
    
    # Check for DC bias
    dc_bias = np.abs(np.mean(audio))
    dc_bias_ok = dc_bias < 0.1
    
    # Overall quality score
    checks = [duration_ok, clipping_ok, silence_ok, dynamic_range_ok, dc_bias_ok]
    quality_score = sum(checks) / len(checks)
    
    results = {
        'duration_ok': duration_ok,
        'duration': duration,
        'clipping_ok': clipping_ok,
        'clipping_ratio': float(clipping_ratio),
        'silence_ok': silence_ok,
        'silence_ratio': float(silence_ratio),
        'dynamic_range_ok': dynamic_range_ok,
        'dynamic_range': float(dynamic_range),
        'dc_bias_ok': dc_bias_ok,
        'dc_bias': float(dc_bias),
        'quality_score': float(quality_score),
        'overall_ok': quality_score >= 0.8
    }
    
    return results


# Convenience function for common preprocessing
def preprocess_audio_standard(audio: np.ndarray,
                             sample_rate: int,
                             target_sr: int = 22050,
                             normalize_method: str = "peak",
                             trim_silence_flag: bool = True) -> Tuple[np.ndarray, int]:
    """
    Apply standard preprocessing steps to audio.
    
    Args:
        audio: Input audio array
        sample_rate: Original sample rate
        target_sr: Target sample rate
        normalize_method: Normalization method
        trim_silence_flag: Whether to trim silence
        
    Returns:
        Tuple of (processed_audio, target_sample_rate)
    """
    # Resample if necessary
    if sample_rate != target_sr:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
    
    # Trim silence
    if trim_silence_flag:
        audio = trim_silence(audio, target_sr)
    
    # Normalize
    audio = normalize_audio(audio, method=normalize_method)
    
    return audio, target_sr
