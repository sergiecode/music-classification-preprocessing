"""
Musical feature extraction utilities for music classification preprocessing.

This module provides comprehensive feature extraction including tempo,
key detection, spectral features, and harmonic content analysis.
"""

import librosa
import numpy as np
from typing import Dict, Any
from scipy import stats


class FeatureExtractor:
    """
    Comprehensive musical feature extractor for classification tasks.
    
    Extracts temporal, spectral, harmonic, and rhythmic features
    suitable for machine learning models.
    """
    
    def __init__(self, 
                 n_mfcc: int = 13,
                 n_chroma: int = 12,
                 n_contrast: int = 6):
        """
        Initialize FeatureExtractor with parameters.
        
        Args:
            n_mfcc: Number of MFCC coefficients to extract
            n_chroma: Number of chroma features to extract
            n_contrast: Number of spectral contrast bands
        """
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.n_contrast = n_contrast
    
    def extract_all_features(self, 
                           audio_data: np.ndarray, 
                           sample_rate: int) -> Dict[str, Any]:
        """
        Extract all available features from audio data.
        
        Args:
            audio_data: Audio time series
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        # Basic audio properties
        features['duration'] = len(audio_data) / sample_rate
        features['sample_rate'] = sample_rate
        
        # Temporal features
        features.update(self.extract_temporal_features(audio_data, sample_rate))
        
        # Spectral features
        features.update(self.extract_spectral_features(audio_data, sample_rate))
        
        # Harmonic features
        features.update(self.extract_harmonic_features(audio_data, sample_rate))
        
        # Rhythmic features
        features.update(self.extract_rhythmic_features(audio_data, sample_rate))
        
        # Statistical summaries
        features.update(self.extract_statistical_features(audio_data, sample_rate))
        
        return features
    
    def extract_temporal_features(self, 
                                audio_data: np.ndarray, 
                                sample_rate: int) -> Dict[str, Any]:
        """
        Extract temporal features including tempo and onset detection.
        
        Args:
            audio_data: Audio time series
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        try:
            # Tempo estimation
            tempo, beats = librosa.beat.beat_track(
                y=audio_data, 
                sr=sample_rate,
                hop_length=512
            )
            features['tempo'] = float(tempo)
            features['beat_count'] = len(beats)
            
            # Onset detection
            onsets = librosa.onset.onset_detect(
                y=audio_data,
                sr=sample_rate,
                hop_length=512
            )
            features['onset_count'] = len(onsets)
            features['onset_density'] = len(onsets) / (len(audio_data) / sample_rate)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=512)
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
        except Exception as e:
            print(f"Warning: Error in temporal feature extraction: {e}")
            features.update({
                'tempo': 0.0,
                'beat_count': 0,
                'onset_count': 0,
                'onset_density': 0.0,
                'zcr_mean': 0.0,
                'zcr_std': 0.0
            })
        
        return features
    
    def extract_spectral_features(self, 
                                audio_data: np.ndarray, 
                                sample_rate: int) -> Dict[str, Any]:
        """
        Extract spectral features including MFCCs and spectral characteristics.
        
        Args:
            audio_data: Audio time series
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary of spectral features
        """
        features = {}
        
        try:
            # MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=self.n_mfcc,
                hop_length=512
            )
            
            # Statistical summaries of MFCCs
            for i in range(self.n_mfcc):
                features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data,
                sr=sample_rate,
                hop_length=512
            )
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data,
                sr=sample_rate,
                hop_length=512
            )
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data,
                sr=sample_rate,
                hop_length=512
            )
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(
                y=audio_data,
                sr=sample_rate,
                n_bands=self.n_contrast,
                hop_length=512
            )
            
            for i in range(self.n_contrast + 1):  # +1 for the sub-band
                features[f'spectral_contrast_{i+1}_mean'] = float(np.mean(spectral_contrast[i]))
                features[f'spectral_contrast_{i+1}_std'] = float(np.std(spectral_contrast[i]))
                
        except Exception as e:
            print(f"Warning: Error in spectral feature extraction: {e}")
            # Set default values for failed extraction
            for i in range(self.n_mfcc):
                features[f'mfcc_{i+1}_mean'] = 0.0
                features[f'mfcc_{i+1}_std'] = 0.0
            
            features.update({
                'spectral_centroid_mean': 0.0,
                'spectral_centroid_std': 0.0,
                'spectral_rolloff_mean': 0.0,
                'spectral_rolloff_std': 0.0,
                'spectral_bandwidth_mean': 0.0,
                'spectral_bandwidth_std': 0.0
            })
            
            for i in range(self.n_contrast + 1):
                features[f'spectral_contrast_{i+1}_mean'] = 0.0
                features[f'spectral_contrast_{i+1}_std'] = 0.0
        
        return features
    
    def extract_harmonic_features(self, 
                                audio_data: np.ndarray, 
                                sample_rate: int) -> Dict[str, Any]:
        """
        Extract harmonic features including chroma and key estimation.
        
        Args:
            audio_data: Audio time series
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary of harmonic features
        """
        features = {}
        
        try:
            # Chroma features
            chroma = librosa.feature.chroma_stft(
                y=audio_data,
                sr=sample_rate,
                hop_length=512
            )
            
            # Statistical summaries of chroma
            for i in range(self.n_chroma):
                pitch_class = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                              'F#', 'G', 'G#', 'A', 'A#', 'B'][i]
                features[f'chroma_{pitch_class}_mean'] = float(np.mean(chroma[i]))
                features[f'chroma_{pitch_class}_std'] = float(np.std(chroma[i]))
            
            # Key estimation (simplified)
            chroma_mean = np.mean(chroma, axis=1)
            estimated_key = np.argmax(chroma_mean)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                        'F#', 'G', 'G#', 'A', 'A#', 'B']
            features['estimated_key'] = key_names[estimated_key]
            features['key_confidence'] = float(np.max(chroma_mean) / np.sum(chroma_mean))
            
            # Tonal centroid features
            tonnetz = librosa.feature.tonnetz(
                y=librosa.effects.harmonic(audio_data),
                sr=sample_rate
            )
            
            tonal_features = ['tonnetz_1', 'tonnetz_2', 'tonnetz_3', 
                             'tonnetz_4', 'tonnetz_5', 'tonnetz_6']
            for i, feature_name in enumerate(tonal_features):
                features[f'{feature_name}_mean'] = float(np.mean(tonnetz[i]))
                features[f'{feature_name}_std'] = float(np.std(tonnetz[i]))
                
        except Exception as e:
            print(f"Warning: Error in harmonic feature extraction: {e}")
            # Set default values
            pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                           'F#', 'G', 'G#', 'A', 'A#', 'B']
            for pitch_class in pitch_classes:
                features[f'chroma_{pitch_class}_mean'] = 0.0
                features[f'chroma_{pitch_class}_std'] = 0.0
            
            features.update({
                'estimated_key': 'C',
                'key_confidence': 0.0
            })
            
            for i in range(6):
                features[f'tonnetz_{i+1}_mean'] = 0.0
                features[f'tonnetz_{i+1}_std'] = 0.0
        
        return features
    
    def extract_rhythmic_features(self, 
                                audio_data: np.ndarray, 
                                sample_rate: int) -> Dict[str, Any]:
        """
        Extract rhythmic features including tempogram analysis.
        
        Args:
            audio_data: Audio time series
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary of rhythmic features
        """
        features = {}
        
        try:
            # Tempogram
            hop_length = 512
            tempogram = librosa.feature.tempogram(
                y=audio_data,
                sr=sample_rate,
                hop_length=hop_length
            )
            
            features['tempogram_mean'] = float(np.mean(tempogram))
            features['tempogram_std'] = float(np.std(tempogram))
            features['tempogram_max'] = float(np.max(tempogram))
            
            # Rhythmic regularity (simplified)
            onset_envelope = librosa.onset.onset_strength(
                y=audio_data,
                sr=sample_rate,
                hop_length=hop_length
            )
            
            # Autocorrelation for rhythm analysis
            autocorr = np.correlate(onset_envelope, onset_envelope, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks in autocorrelation (rhythm patterns)
            if len(autocorr) > 10:
                features['rhythm_regularity'] = float(np.std(autocorr[:min(50, len(autocorr))]))
            else:
                features['rhythm_regularity'] = 0.0
                
        except Exception as e:
            print(f"Warning: Error in rhythmic feature extraction: {e}")
            features.update({
                'tempogram_mean': 0.0,
                'tempogram_std': 0.0,
                'tempogram_max': 0.0,
                'rhythm_regularity': 0.0
            })
        
        return features
    
    def extract_statistical_features(self, 
                                   audio_data: np.ndarray, 
                                   sample_rate: int) -> Dict[str, Any]:
        """
        Extract statistical features from the raw audio signal.
        
        Args:
            audio_data: Audio time series
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary of statistical features
        """
        features = {}
        
        try:
            # Basic statistical measures
            features['rms_energy'] = float(np.sqrt(np.mean(audio_data**2)))
            features['max_amplitude'] = float(np.max(np.abs(audio_data)))
            features['mean_amplitude'] = float(np.mean(np.abs(audio_data)))
            features['std_amplitude'] = float(np.std(audio_data))
            
            # Signal statistics
            features['skewness'] = float(stats.skew(audio_data))
            features['kurtosis'] = float(stats.kurtosis(audio_data))
            
            # Dynamic range
            features['dynamic_range'] = float(np.max(audio_data) - np.min(audio_data))
            
        except Exception as e:
            print(f"Warning: Error in statistical feature extraction: {e}")
            features.update({
                'rms_energy': 0.0,
                'max_amplitude': 0.0,
                'mean_amplitude': 0.0,
                'std_amplitude': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'dynamic_range': 0.0
            })
        
        return features
    
    def extract_feature_summary(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of extracted features for quick analysis.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Dictionary of feature summary statistics
        """
        summary = {
            'total_features': len(features),
            'feature_categories': {
                'temporal': len([k for k in features.keys() if any(x in k.lower() for x in ['tempo', 'beat', 'onset', 'zcr'])]),
                'spectral': len([k for k in features.keys() if any(x in k.lower() for x in ['mfcc', 'spectral', 'contrast'])]),
                'harmonic': len([k for k in features.keys() if any(x in k.lower() for x in ['chroma', 'key', 'tonnetz'])]),
                'rhythmic': len([k for k in features.keys() if any(x in k.lower() for x in ['tempogram', 'rhythm'])]),
                'statistical': len([k for k in features.keys() if any(x in k.lower() for x in ['rms', 'amplitude', 'skew', 'kurt', 'dynamic'])])
            }
        }
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor()
    
    print("FeatureExtractor initialized successfully!")
    print(f"MFCC coefficients: {extractor.n_mfcc}")
    print(f"Chroma features: {extractor.n_chroma}")
    print(f"Spectral contrast bands: {extractor.n_contrast}")
