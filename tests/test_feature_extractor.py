"""
Unit tests for FeatureExtractor class.
"""

import pytest
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_extractor import FeatureExtractor


class TestFeatureExtractor:
    """Test cases for FeatureExtractor class."""
    
    @pytest.fixture
    def feature_extractor(self):
        """Create FeatureExtractor instance for testing."""
        return FeatureExtractor(n_mfcc=13, n_chroma=12, n_contrast=6)
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data for testing."""
        duration = 2.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(duration * sample_rate))
        # Create a more complex signal with multiple frequencies
        audio = (
            0.5 * np.sin(2 * np.pi * 440 * t) +  # A4
            0.3 * np.sin(2 * np.pi * 880 * t) +  # A5
            0.1 * np.random.randn(len(t))        # Noise
        )
        return audio, sample_rate
    
    def test_initialization(self):
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor()
        assert extractor.n_mfcc == 13
        assert extractor.n_chroma == 12
        assert extractor.n_contrast == 6
    
    def test_custom_initialization(self):
        """Test FeatureExtractor with custom parameters."""
        extractor = FeatureExtractor(n_mfcc=20, n_chroma=24, n_contrast=7)
        assert extractor.n_mfcc == 20
        assert extractor.n_chroma == 24
        assert extractor.n_contrast == 7
    
    def test_extract_all_features(self, feature_extractor, sample_audio):
        """Test extraction of all features."""
        audio_data, sample_rate = sample_audio
        
        features = feature_extractor.extract_all_features(audio_data, sample_rate)
        
        # Check that features is a dictionary
        assert isinstance(features, dict)
        
        # Check for basic properties
        assert 'duration' in features
        assert 'sample_rate' in features
        assert features['duration'] > 0
        assert features['sample_rate'] == sample_rate
        
        # Check for different feature categories
        temporal_features = [k for k in features.keys() if any(x in k.lower() for x in ['tempo', 'beat', 'onset', 'zcr'])]
        spectral_features = [k for k in features.keys() if any(x in k.lower() for x in ['mfcc', 'spectral', 'contrast'])]
        harmonic_features = [k for k in features.keys() if any(x in k.lower() for x in ['chroma', 'key', 'tonnetz'])]
        
        assert len(temporal_features) > 0
        assert len(spectral_features) > 0
        assert len(harmonic_features) > 0
    
    def test_extract_temporal_features(self, feature_extractor, sample_audio):
        """Test temporal feature extraction."""
        audio_data, sample_rate = sample_audio
        
        features = feature_extractor.extract_temporal_features(audio_data, sample_rate)
        
        # Check required temporal features
        required_features = ['tempo', 'beat_count', 'onset_count', 'onset_density', 'zcr_mean', 'zcr_std']
        
        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
        
        # Check reasonable ranges
        assert features['tempo'] >= 0
        assert features['beat_count'] >= 0
        assert features['onset_count'] >= 0
        assert features['onset_density'] >= 0
        assert 0 <= features['zcr_mean'] <= 1
        assert features['zcr_std'] >= 0
    
    def test_extract_spectral_features(self, feature_extractor, sample_audio):
        """Test spectral feature extraction."""
        audio_data, sample_rate = sample_audio
        
        features = feature_extractor.extract_spectral_features(audio_data, sample_rate)
        
        # Check MFCC features
        for i in range(feature_extractor.n_mfcc):
            assert f'mfcc_{i+1}_mean' in features
            assert f'mfcc_{i+1}_std' in features
            assert isinstance(features[f'mfcc_{i+1}_mean'], float)
            assert isinstance(features[f'mfcc_{i+1}_std'], float)
        
        # Check spectral features
        spectral_features = ['spectral_centroid_mean', 'spectral_centroid_std',
                           'spectral_rolloff_mean', 'spectral_rolloff_std',
                           'spectral_bandwidth_mean', 'spectral_bandwidth_std']
        
        for feature in spectral_features:
            assert feature in features
            assert isinstance(features[feature], float)
            assert features[feature] >= 0
        
        # Check spectral contrast features
        for i in range(feature_extractor.n_contrast + 1):
            assert f'spectral_contrast_{i+1}_mean' in features
            assert f'spectral_contrast_{i+1}_std' in features
    
    def test_extract_harmonic_features(self, feature_extractor, sample_audio):
        """Test harmonic feature extraction."""
        audio_data, sample_rate = sample_audio
        
        features = feature_extractor.extract_harmonic_features(audio_data, sample_rate)
        
        # Check chroma features
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for pitch_class in pitch_classes:
            assert f'chroma_{pitch_class}_mean' in features
            assert f'chroma_{pitch_class}_std' in features
            assert isinstance(features[f'chroma_{pitch_class}_mean'], float)
            assert isinstance(features[f'chroma_{pitch_class}_std'], float)
        
        # Check key estimation
        assert 'estimated_key' in features
        assert 'key_confidence' in features
        assert features['estimated_key'] in pitch_classes
        assert 0 <= features['key_confidence'] <= 1
        
        # Check tonnetz features
        for i in range(6):
            assert f'tonnetz_{i+1}_mean' in features
            assert f'tonnetz_{i+1}_std' in features
    
    def test_extract_rhythmic_features(self, feature_extractor, sample_audio):
        """Test rhythmic feature extraction."""
        audio_data, sample_rate = sample_audio
        
        features = feature_extractor.extract_rhythmic_features(audio_data, sample_rate)
        
        # Check rhythmic features
        rhythmic_features = ['tempogram_mean', 'tempogram_std', 'tempogram_max', 'rhythm_regularity']
        
        for feature in rhythmic_features:
            assert feature in features
            assert isinstance(features[feature], float)
            assert features[feature] >= 0
    
    def test_extract_statistical_features(self, feature_extractor, sample_audio):
        """Test statistical feature extraction."""
        audio_data, sample_rate = sample_audio
        
        features = feature_extractor.extract_statistical_features(audio_data, sample_rate)
        
        # Check statistical features
        statistical_features = ['rms_energy', 'max_amplitude', 'mean_amplitude', 'std_amplitude',
                              'skewness', 'kurtosis', 'dynamic_range']
        
        for feature in statistical_features:
            assert feature in features
            assert isinstance(features[feature], float)
        
        # Check reasonable ranges
        assert features['rms_energy'] >= 0
        assert features['max_amplitude'] >= 0
        assert features['mean_amplitude'] >= 0
        assert features['std_amplitude'] >= 0
        assert features['dynamic_range'] >= 0
    
    def test_extract_feature_summary(self, feature_extractor, sample_audio):
        """Test feature summary extraction."""
        audio_data, sample_rate = sample_audio
        
        features = feature_extractor.extract_all_features(audio_data, sample_rate)
        summary = feature_extractor.extract_feature_summary(features)
        
        assert isinstance(summary, dict)
        assert 'total_features' in summary
        assert 'feature_categories' in summary
        assert summary['total_features'] == len(features)
        
        categories = summary['feature_categories']
        assert 'temporal' in categories
        assert 'spectral' in categories
        assert 'harmonic' in categories
        assert 'rhythmic' in categories
        assert 'statistical' in categories
        
        # Check that category counts sum to total or less (some features might not be categorized)
        total_categorized = sum(categories.values())
        assert total_categorized <= summary['total_features']
    
    def test_error_handling(self, feature_extractor):
        """Test error handling with invalid input."""
        # Test with empty audio
        empty_audio = np.array([])
        sample_rate = 22050
        
        # Should not crash, should return default values
        features = feature_extractor.extract_temporal_features(empty_audio, sample_rate)
        assert isinstance(features, dict)
        
        # Test with very short audio
        short_audio = np.random.randn(100)  # Very short
        features = feature_extractor.extract_all_features(short_audio, sample_rate)
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_feature_consistency(self, feature_extractor):
        """Test that feature extraction is consistent across runs."""
        # Create deterministic audio
        duration = 1.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # Pure tone
        
        # Extract features twice
        features1 = feature_extractor.extract_all_features(audio, sample_rate)
        features2 = feature_extractor.extract_all_features(audio, sample_rate)
        
        # Compare numerical features (allowing for small floating point differences)
        for key in features1.keys():
            if isinstance(features1[key], (int, float)):
                assert abs(features1[key] - features2[key]) < 1e-10
            else:
                assert features1[key] == features2[key]


if __name__ == "__main__":
    pytest.main([__file__])
