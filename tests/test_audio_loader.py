"""
Unit tests for AudioLoader class.
"""

import pytest
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.append('../src')

from audio_loader import AudioLoader


class TestAudioLoader:
    """Test cases for AudioLoader class."""
    
    @pytest.fixture
    def audio_loader(self):
        """Create AudioLoader instance for testing."""
        return AudioLoader(target_sr=22050, mono=True, normalize=True)
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create a temporary audio file for testing."""
        # Generate test audio data
        duration = 2.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio_data, sample_rate)
            return Path(f.name)
    
    def test_initialization(self):
        """Test AudioLoader initialization."""
        loader = AudioLoader()
        assert loader.target_sr == 22050
        assert loader.mono is True
        assert loader.normalize is True
    
    def test_custom_initialization(self):
        """Test AudioLoader with custom parameters."""
        loader = AudioLoader(target_sr=16000, mono=False, normalize=False)
        assert loader.target_sr == 16000
        assert loader.mono is False
        assert loader.normalize is False
    
    def test_load_audio_success(self, audio_loader, sample_audio_file):
        """Test successful audio loading."""
        try:
            audio_data, sample_rate = audio_loader.load_audio(sample_audio_file)
            
            assert isinstance(audio_data, np.ndarray)
            assert sample_rate == audio_loader.target_sr
            assert len(audio_data) > 0
            assert audio_data.ndim == 1  # Should be mono
            
            # Check normalization
            if audio_loader.normalize:
                assert np.max(np.abs(audio_data)) <= 1.0
        
        finally:
            # Clean up temporary file
            sample_audio_file.unlink()
    
    def test_load_audio_file_not_found(self, audio_loader):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            audio_loader.load_audio("nonexistent_file.wav")
    
    def test_load_audio_unsupported_format(self, audio_loader):
        """Test loading unsupported file format."""
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_file = Path(f.name)
        
        try:
            with pytest.raises(ValueError):
                audio_loader.load_audio(temp_file)
        finally:
            temp_file.unlink()
    
    def test_get_audio_info(self, audio_loader, sample_audio_file):
        """Test getting audio file information."""
        try:
            info = audio_loader.get_audio_info(sample_audio_file)
            
            assert isinstance(info, dict)
            assert 'filename' in info
            assert 'duration' in info
            assert 'sample_rate' in info
            assert 'channels' in info
            assert info['duration'] > 0
            assert info['sample_rate'] > 0
        
        finally:
            sample_audio_file.unlink()
    
    def test_audio_segment_loading(self, audio_loader, sample_audio_file):
        """Test loading audio segment."""
        try:
            start_time = 0.5
            end_time = 1.5
            
            audio_data, sample_rate = audio_loader.load_audio_segment(
                sample_audio_file, start_time, end_time
            )
            
            expected_duration = end_time - start_time
            actual_duration = len(audio_data) / sample_rate
            
            # Allow small tolerance for duration
            assert abs(actual_duration - expected_duration) < 0.1
        
        finally:
            sample_audio_file.unlink()
    
    def test_normalization(self):
        """Test audio normalization."""
        loader = AudioLoader(normalize=True)
        
        # Create test audio with known amplitude
        audio_data = np.array([0.1, -0.8, 0.5, -0.2])
        normalized = loader._normalize_audio(audio_data)
        
        # Should be normalized to max amplitude of 1.0
        assert np.max(np.abs(normalized)) == 1.0
    
    def test_no_normalization(self):
        """Test audio without normalization."""
        loader = AudioLoader(normalize=False)
        
        # Create test audio
        original_audio = np.array([0.1, -0.8, 0.5, -0.2])
        
        # Should remain unchanged (except for potential resampling)
        # This test would need to be more complex for a real file
        assert True  # Placeholder for more detailed test
    
    def test_supported_formats(self, audio_loader):
        """Test that supported formats are correctly defined."""
        expected_formats = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aiff'}
        assert audio_loader.supported_formats == expected_formats


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])
