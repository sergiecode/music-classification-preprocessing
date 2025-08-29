"""
Unit tests for CLI module.
"""

import pytest
import sys
import os
import tempfile
import json
import soundfile as sf
import numpy as np
from pathlib import Path
from unittest.mock import patch
from io import StringIO

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cli


class TestCLI:
    """Test cases for CLI module."""
    
    @pytest.fixture
    def temp_directories(self):
        """Create temporary directories for testing."""
        temp_dir = tempfile.mkdtemp()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        
        input_dir.mkdir()
        output_dir.mkdir()
        
        return input_dir, output_dir
    
    @pytest.fixture
    def sample_audio_file(self, temp_directories):
        """Create a sample audio file for testing."""
        input_dir, output_dir = temp_directories
        
        # Create test audio
        duration = 2.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        file_path = input_dir / "test_audio.wav"
        sf.write(str(file_path), audio_data, sample_rate)
        
        return file_path, input_dir, output_dir
    
    def test_extract_features_command(self, sample_audio_file):
        """Test extract-features command."""
        file_path, input_dir, output_dir = sample_audio_file
        
        # Mock command line arguments
        class MockArgs:
            input = str(file_path)
            output = str(output_dir)
            summary = False
        
        args = MockArgs()
        
        # Capture stdout
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            cli.extract_features_command(args)
        
        # Check output
        output = mock_stdout.getvalue()
        assert "Features saved to:" in output
        assert "Extracted" in output
        
        # Check that features file was created
        features_file = output_dir / f"{file_path.stem}_features.json"
        assert features_file.exists()
        
        # Verify features content
        with open(features_file, 'r') as f:
            features = json.load(f)
        
        assert isinstance(features, dict)
        assert 'duration' in features
        assert 'sample_rate' in features
        assert 'processing_info' in features
    
    def test_extract_features_command_with_summary(self, sample_audio_file):
        """Test extract-features command with summary."""
        file_path, input_dir, output_dir = sample_audio_file
        
        class MockArgs:
            input = str(file_path)
            output = str(output_dir)
            summary = True
        
        args = MockArgs()
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            cli.extract_features_command(args)
        
        output = mock_stdout.getvalue()
        assert "Feature Summary:" in output
        assert "Total features:" in output
    
    def test_extract_features_command_file_not_found(self, temp_directories):
        """Test extract-features command with non-existent file."""
        input_dir, output_dir = temp_directories
        
        class MockArgs:
            input = str(input_dir / "nonexistent.wav")
            output = str(output_dir)
            summary = False
        
        args = MockArgs()
        
        with pytest.raises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO):
                cli.extract_features_command(args)
    
    def test_generate_spectrograms_command_single_file(self, sample_audio_file):
        """Test generate-spectrograms command with single file."""
        file_path, input_dir, output_dir = sample_audio_file
        
        class MockArgs:
            input = str(file_path)
            output = str(output_dir)
            n_mels = 128
            hop_length = 512
            fixed_length = None
            save_image = False
        
        args = MockArgs()
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            cli.generate_spectrograms_command(args)
        
        output = mock_stdout.getvalue()
        assert "Spectrogram saved to:" in output
        assert "Spectrogram shape:" in output
        
        # Check that spectrogram file was created
        spec_file = output_dir / f"{file_path.stem}_spectrogram.npy"
        assert spec_file.exists()
        
        # Verify spectrogram content
        spectrogram = np.load(spec_file)
        assert isinstance(spectrogram, np.ndarray)
        assert spectrogram.shape[0] == 128  # n_mels
    
    def test_generate_spectrograms_command_with_image(self, sample_audio_file):
        """Test generate-spectrograms command with image saving."""
        file_path, input_dir, output_dir = sample_audio_file
        
        class MockArgs:
            input = str(file_path)
            output = str(output_dir)
            n_mels = 128
            hop_length = 512
            fixed_length = None
            save_image = True
        
        args = MockArgs()
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            cli.generate_spectrograms_command(args)
        
        output = mock_stdout.getvalue()
        assert "Spectrogram image saved to:" in output
        
        # Check that image file was created
        image_file = output_dir / f"{file_path.stem}_spectrogram.png"
        assert image_file.exists()
    
    def test_generate_spectrograms_command_directory(self, sample_audio_file):
        """Test generate-spectrograms command with directory."""
        file_path, input_dir, output_dir = sample_audio_file
        
        class MockArgs:
            input = str(input_dir)
            output = str(output_dir)
            n_mels = 128
            hop_length = 512
            fixed_length = None
            save_image = False
        
        args = MockArgs()
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            cli.generate_spectrograms_command(args)
        
        output = mock_stdout.getvalue()
        assert "Processed" in output
    
    def test_preprocess_dataset_command(self, sample_audio_file):
        """Test preprocess-dataset command."""
        file_path, input_dir, output_dir = sample_audio_file
        
        class MockArgs:
            input = str(input_dir)
            output = str(output_dir)
            features = True
            spectrograms = True
            fixed_length = None
            pattern = None
            manifest = None
            stats = False
        
        args = MockArgs()
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            cli.preprocess_dataset_command(args)
        
        output = mock_stdout.getvalue()
        assert "Processing complete:" in output
        assert "Processed:" in output
    
    def test_preprocess_dataset_command_with_manifest(self, sample_audio_file):
        """Test preprocess-dataset command with manifest creation."""
        file_path, input_dir, output_dir = sample_audio_file
        
        class MockArgs:
            input = str(input_dir)
            output = str(output_dir)
            features = True
            spectrograms = False
            fixed_length = None
            pattern = None
            manifest = "test_dataset"
            stats = False
        
        args = MockArgs()
        
        with patch('sys.stdout', new_callable=StringIO):
            cli.preprocess_dataset_command(args)
        
        # Check that manifest was created
        manifest_file = output_dir / "test_dataset_manifest.json"
        assert manifest_file.exists()
    
    def test_preprocess_dataset_command_with_stats(self, sample_audio_file):
        """Test preprocess-dataset command with statistics."""
        file_path, input_dir, output_dir = sample_audio_file
        
        class MockArgs:
            input = str(input_dir)
            output = str(output_dir)
            features = True
            spectrograms = False
            fixed_length = None
            pattern = None
            manifest = None
            stats = True
        
        args = MockArgs()
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            cli.preprocess_dataset_command(args)
        
        output = mock_stdout.getvalue()
        assert "Dataset Statistics:" in output
        assert "Total duration:" in output
        assert "Success rate:" in output
    
    def test_info_command_single_file(self, sample_audio_file):
        """Test info command with single file."""
        file_path, input_dir, output_dir = sample_audio_file
        
        class MockArgs:
            input = str(file_path)
        
        args = MockArgs()
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            cli.info_command(args)
        
        output = mock_stdout.getvalue()
        assert "Audio File Information:" in output
        assert "Filename:" in output
        assert "Duration:" in output
        assert "Sample Rate:" in output
    
    def test_info_command_directory(self, sample_audio_file):
        """Test info command with directory."""
        file_path, input_dir, output_dir = sample_audio_file
        
        class MockArgs:
            input = str(input_dir)
        
        args = MockArgs()
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            cli.info_command(args)
        
        output = mock_stdout.getvalue()
        assert "Directory Information:" in output
        assert "Total files:" in output
        assert "Total duration:" in output
    
    def test_info_command_empty_directory(self, temp_directories):
        """Test info command with empty directory."""
        input_dir, output_dir = temp_directories
        
        class MockArgs:
            input = str(input_dir)
        
        args = MockArgs()
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            cli.info_command(args)
        
        output = mock_stdout.getvalue()
        assert "No audio files found in directory" in output
    
    def test_info_command_invalid_path(self, temp_directories):
        """Test info command with invalid path."""
        input_dir, output_dir = temp_directories
        
        class MockArgs:
            input = str(input_dir / "nonexistent")
        
        args = MockArgs()
        
        with pytest.raises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO):
                cli.info_command(args)
    
    @patch('sys.argv', ['cli.py'])
    def test_main_no_args(self):
        """Test main function with no arguments."""
        with pytest.raises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO):
                cli.main()
    
    @patch('sys.argv', ['cli.py', '--help'])
    def test_main_help(self):
        """Test main function with help argument."""
        with pytest.raises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO):
                cli.main()
    
    @patch('sys.argv', ['cli.py', 'extract-features', '--help'])
    def test_extract_features_help(self):
        """Test extract-features subcommand help."""
        with pytest.raises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO):
                cli.main()
    
    def test_preprocess_dataset_invalid_directory(self, temp_directories):
        """Test preprocess-dataset with invalid input directory."""
        input_dir, output_dir = temp_directories
        
        class MockArgs:
            input = str(input_dir / "nonexistent")
            output = str(output_dir)
            features = True
            spectrograms = False
            fixed_length = None
            pattern = None
            manifest = None
            stats = False
        
        args = MockArgs()
        
        with pytest.raises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO):
                cli.preprocess_dataset_command(args)


if __name__ == "__main__":
    pytest.main([__file__])
