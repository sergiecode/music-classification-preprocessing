"""
Unit tests for SpectrogramGenerator class.
"""

import pytest
import numpy as np
import sys
import os
import tempfile
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectrogram_generator import SpectrogramGenerator


class TestSpectrogramGenerator:
    """Test cases for SpectrogramGenerator class."""
    
    @pytest.fixture
    def spectrogram_generator(self):
        """Create SpectrogramGenerator instance for testing."""
        return SpectrogramGenerator(n_mels=128, hop_length=512, n_fft=2048)
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data for testing."""
        duration = 3.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(duration * sample_rate))
        # Create a complex signal
        audio = (
            0.5 * np.sin(2 * np.pi * 440 * t) +  # A4
            0.3 * np.sin(2 * np.pi * 880 * t) +  # A5
            0.2 * np.sin(2 * np.pi * 220 * t) +  # A3
            0.1 * np.random.randn(len(t))        # Noise
        )
        return audio, sample_rate
    
    def test_initialization(self):
        """Test SpectrogramGenerator initialization."""
        generator = SpectrogramGenerator()
        assert generator.n_mels == 128
        assert generator.hop_length == 512
        assert generator.win_length == 2048
        assert generator.n_fft == 2048
        assert generator.fmin == 0.0
        assert generator.fmax is None
    
    def test_custom_initialization(self):
        """Test SpectrogramGenerator with custom parameters."""
        generator = SpectrogramGenerator(
            n_mels=64, hop_length=256, win_length=1024, 
            n_fft=1024, fmin=50.0, fmax=8000.0
        )
        assert generator.n_mels == 64
        assert generator.hop_length == 256
        assert generator.win_length == 1024
        assert generator.n_fft == 1024
        assert generator.fmin == 50.0
        assert generator.fmax == 8000.0
    
    def test_generate_mel_spectrogram(self, spectrogram_generator, sample_audio):
        """Test mel-spectrogram generation."""
        audio_data, sample_rate = sample_audio
        
        # Test with default parameters
        mel_spec = spectrogram_generator.generate_mel_spectrogram(
            audio_data, sample_rate, to_db=True, normalize=True
        )
        
        # Check output shape and properties
        assert isinstance(mel_spec, np.ndarray)
        assert mel_spec.ndim == 2
        assert mel_spec.shape[0] == spectrogram_generator.n_mels
        assert mel_spec.shape[1] > 0  # Should have time frames
        
        # Check normalization (values should be between 0 and 1)
        assert np.min(mel_spec) >= 0
        assert np.max(mel_spec) <= 1
    
    def test_generate_mel_spectrogram_no_db(self, spectrogram_generator, sample_audio):
        """Test mel-spectrogram generation without dB conversion."""
        audio_data, sample_rate = sample_audio
        
        mel_spec = spectrogram_generator.generate_mel_spectrogram(
            audio_data, sample_rate, to_db=False, normalize=False
        )
        
        # Should have positive values (power spectrogram)
        assert np.all(mel_spec >= 0)
        assert isinstance(mel_spec, np.ndarray)
        assert mel_spec.shape[0] == spectrogram_generator.n_mels
    
    def test_generate_fixed_length_spectrogram(self, spectrogram_generator, sample_audio):
        """Test fixed-length spectrogram generation."""
        audio_data, sample_rate = sample_audio
        target_length = 2.0  # 2 seconds
        
        mel_spec = spectrogram_generator.generate_fixed_length_spectrogram(
            audio_data, sample_rate, target_length
        )
        
        # Calculate expected number of frames
        target_samples = int(target_length * sample_rate)
        expected_frames = (target_samples // spectrogram_generator.hop_length) + 1
        
        assert isinstance(mel_spec, np.ndarray)
        assert mel_spec.shape[0] == spectrogram_generator.n_mels
        # Allow some tolerance for frame calculation
        assert abs(mel_spec.shape[1] - expected_frames) <= 2
    
    def test_generate_log_mel_spectrogram(self, spectrogram_generator, sample_audio):
        """Test log mel-spectrogram generation."""
        audio_data, sample_rate = sample_audio
        
        log_mel_spec = spectrogram_generator.generate_log_mel_spectrogram(
            audio_data, sample_rate
        )
        
        assert isinstance(log_mel_spec, np.ndarray)
        assert log_mel_spec.shape[0] == spectrogram_generator.n_mels
        assert log_mel_spec.shape[1] > 0
        
        # Log values should be normalized
        assert np.min(log_mel_spec) >= 0
        assert np.max(log_mel_spec) <= 1
    
    def test_generate_delta_features(self, spectrogram_generator, sample_audio):
        """Test delta feature generation."""
        audio_data, sample_rate = sample_audio
        
        # Generate base spectrogram
        mel_spec = spectrogram_generator.generate_mel_spectrogram(
            audio_data, sample_rate
        )
        
        # Test first-order delta
        delta1 = spectrogram_generator.generate_delta_features(mel_spec, order=1)
        assert delta1.shape == mel_spec.shape
        
        # Test second-order delta (delta-delta)
        delta2 = spectrogram_generator.generate_delta_features(mel_spec, order=2)
        assert delta2.shape == mel_spec.shape
        
        # Test invalid order
        with pytest.raises(ValueError):
            spectrogram_generator.generate_delta_features(mel_spec, order=3)
    
    def test_generate_augmented_spectrograms(self, spectrogram_generator, sample_audio):
        """Test augmented spectrogram generation."""
        audio_data, sample_rate = sample_audio
        n_augmentations = 3
        
        augmented_specs = spectrogram_generator.generate_augmented_spectrograms(
            audio_data, sample_rate, n_augmentations=n_augmentations
        )
        
        # Should return original + augmented versions
        assert len(augmented_specs) == n_augmentations + 1
        
        # All spectrograms should have same mel dimension
        for spec in augmented_specs:
            assert spec.shape[0] == spectrogram_generator.n_mels
            assert isinstance(spec, np.ndarray)
    
    def test_save_spectrogram_npy(self, spectrogram_generator, sample_audio):
        """Test saving spectrogram in NumPy format."""
        audio_data, sample_rate = sample_audio
        
        mel_spec = spectrogram_generator.generate_mel_spectrogram(audio_data, sample_rate)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_spec.npy"
            
            spectrogram_generator.save_spectrogram(mel_spec, output_path, format='npy')
            
            assert output_path.exists()
            
            # Load and verify
            loaded_spec = np.load(output_path)
            np.testing.assert_array_equal(mel_spec, loaded_spec)
    
    def test_save_spectrogram_npz(self, spectrogram_generator, sample_audio):
        """Test saving spectrogram in compressed NumPy format."""
        audio_data, sample_rate = sample_audio
        
        mel_spec = spectrogram_generator.generate_mel_spectrogram(audio_data, sample_rate)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_spec.npz"
            
            spectrogram_generator.save_spectrogram(mel_spec, output_path, format='npz')
            
            assert output_path.exists()
            
            # Load and verify
            loaded_data = np.load(output_path)
            np.testing.assert_array_equal(mel_spec, loaded_data['spectrogram'])
    
    def test_save_spectrogram_csv(self, spectrogram_generator, sample_audio):
        """Test saving spectrogram in CSV format."""
        audio_data, sample_rate = sample_audio
        
        mel_spec = spectrogram_generator.generate_mel_spectrogram(audio_data, sample_rate)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_spec.csv"
            
            spectrogram_generator.save_spectrogram(mel_spec, output_path, format='csv')
            
            assert output_path.exists()
            
            # Load and verify (with some tolerance for CSV precision)
            loaded_spec = np.loadtxt(output_path, delimiter=',')
            np.testing.assert_allclose(mel_spec, loaded_spec, rtol=1e-6)
    
    def test_save_spectrogram_invalid_format(self, spectrogram_generator, sample_audio):
        """Test saving spectrogram with invalid format."""
        audio_data, sample_rate = sample_audio
        
        mel_spec = spectrogram_generator.generate_mel_spectrogram(audio_data, sample_rate)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_spec.xyz"
            
            with pytest.raises(ValueError):
                spectrogram_generator.save_spectrogram(mel_spec, output_path, format='xyz')
    
    def test_save_spectrogram_image(self, spectrogram_generator, sample_audio):
        """Test saving spectrogram as image."""
        audio_data, sample_rate = sample_audio
        
        mel_spec = spectrogram_generator.generate_mel_spectrogram(audio_data, sample_rate)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_spec.png"
            
            spectrogram_generator.save_spectrogram_image(
                mel_spec, output_path, title="Test Spectrogram"
            )
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0  # Should have content
    
    def test_visualize_spectrogram_comparison(self, spectrogram_generator, sample_audio):
        """Test spectrogram comparison visualization."""
        audio_data, sample_rate = sample_audio
        
        # Generate different spectrograms
        spec1 = spectrogram_generator.generate_mel_spectrogram(audio_data, sample_rate)
        spec2 = spectrogram_generator.generate_log_mel_spectrogram(audio_data, sample_rate)
        
        spectrograms = [spec1, spec2]
        labels = ["Linear Scale", "Log Scale"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "comparison.png"
            
            spectrogram_generator.visualize_spectrogram_comparison(
                spectrograms, labels, output_path
            )
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_get_spectrogram_stats(self, spectrogram_generator, sample_audio):
        """Test spectrogram statistics calculation."""
        audio_data, sample_rate = sample_audio
        
        mel_spec = spectrogram_generator.generate_mel_spectrogram(audio_data, sample_rate)
        stats = spectrogram_generator.get_spectrogram_stats(mel_spec)
        
        assert isinstance(stats, dict)
        
        required_stats = ['shape', 'min_value', 'max_value', 'mean_value', 
                         'std_value', 'total_energy', 'spectral_density']
        
        for stat in required_stats:
            assert stat in stats
        
        # Check reasonable values
        assert stats['shape'] == mel_spec.shape
        assert stats['min_value'] <= stats['max_value']
        assert stats['total_energy'] >= 0
        assert stats['spectral_density'] >= 0
    
    def test_normalize_spectrogram(self, spectrogram_generator):
        """Test spectrogram normalization."""
        # Create test spectrogram with known values
        test_spec = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        normalized = spectrogram_generator._normalize_spectrogram(test_spec)
        
        # Should be normalized to [0, 1] range
        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0
        assert normalized.shape == test_spec.shape
    
    def test_normalize_spectrogram_constant(self, spectrogram_generator):
        """Test normalization of constant spectrogram."""
        # Create constant spectrogram
        test_spec = np.ones((10, 20)) * 5.0
        
        normalized = spectrogram_generator._normalize_spectrogram(test_spec)
        
        # Should return zeros for constant input
        assert np.all(normalized == 0.0)
        assert normalized.shape == test_spec.shape
    
    def test_error_handling(self, spectrogram_generator):
        """Test error handling with invalid input."""
        # Test with empty audio
        empty_audio = np.array([])
        sample_rate = 22050
        
        with pytest.raises(RuntimeError):
            spectrogram_generator.generate_mel_spectrogram(empty_audio, sample_rate)
    
    def test_spectrogram_consistency(self, spectrogram_generator):
        """Test that spectrogram generation is consistent."""
        # Create deterministic audio
        duration = 1.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # Pure tone
        
        # Generate spectrograms twice
        spec1 = spectrogram_generator.generate_mel_spectrogram(audio, sample_rate)
        spec2 = spectrogram_generator.generate_mel_spectrogram(audio, sample_rate)
        
        # Should be identical
        np.testing.assert_array_equal(spec1, spec2)


if __name__ == "__main__":
    pytest.main([__file__])
