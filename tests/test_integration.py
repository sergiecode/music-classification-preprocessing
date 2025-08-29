"""
Integration tests for the entire music preprocessing pipeline.
"""

import pytest
import sys
import os
import tempfile
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import patch
from io import StringIO

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from audio_loader import AudioLoader
from feature_extractor import FeatureExtractor
from spectrogram_generator import SpectrogramGenerator
from batch_processor import BatchProcessor
import utils.audio_utils as audio_utils
import utils.visualization as visualization


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def setup_test_environment(self):
        """Set up test environment with sample audio files."""
        temp_dir = tempfile.mkdtemp()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create multiple test audio files
        sample_rate = 22050
        
        # Create different types of audio signals
        audio_files = []
        
        # 1. Simple sine wave
        duration = 3.0
        t = np.linspace(0, duration, int(duration * sample_rate))
        sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)
        sine_file = input_dir / "sine_440hz.wav"
        sf.write(str(sine_file), sine_wave, sample_rate)
        audio_files.append(sine_file)
        
        # 2. Complex signal with harmonics
        complex_signal = (0.3 * np.sin(2 * np.pi * 220 * t) + 
                         0.2 * np.sin(2 * np.pi * 440 * t) + 
                         0.1 * np.sin(2 * np.pi * 880 * t))
        complex_file = input_dir / "complex_harmonics.wav"
        sf.write(str(complex_file), complex_signal, sample_rate)
        audio_files.append(complex_file)
        
        # 3. Noisy signal
        noise = np.random.normal(0, 0.1, len(t))
        noisy_signal = 0.4 * np.sin(2 * np.pi * 330 * t) + noise
        noisy_file = input_dir / "noisy_signal.wav"
        sf.write(str(noisy_file), noisy_signal, sample_rate)
        audio_files.append(noisy_file)
        
        # 4. Very short file
        short_duration = 0.5
        t_short = np.linspace(0, short_duration, int(short_duration * sample_rate))
        short_signal = 0.5 * np.sin(2 * np.pi * 880 * t_short)
        short_file = input_dir / "short_signal.wav"
        sf.write(str(short_file), short_signal, sample_rate)
        audio_files.append(short_file)
        
        return input_dir, output_dir, audio_files
    
    def test_complete_pipeline_single_file(self, setup_test_environment):
        """Test the complete pipeline on a single audio file."""
        input_dir, output_dir, audio_files = setup_test_environment
        test_file = audio_files[0]  # sine wave
        
        # 1. Load audio
        loader = AudioLoader()
        audio_data, sample_rate = loader.load_audio(str(test_file))
        
        assert audio_data is not None
        assert sample_rate > 0
        assert len(audio_data) > 0
        
        # 2. Extract features
        extractor = FeatureExtractor()
        features = extractor.extract_all_features(audio_data, sample_rate)
        
        assert isinstance(features, dict)
        assert len(features) > 20  # Should have many features
        assert 'duration' in features
        assert 'sample_rate' in features
        
        # 3. Generate spectrogram
        spec_gen = SpectrogramGenerator()
        spectrogram = spec_gen.generate_mel_spectrogram(audio_data, sample_rate)
        
        assert isinstance(spectrogram, np.ndarray)
        assert len(spectrogram.shape) == 2
        assert spectrogram.shape[0] == 128  # Default n_mels
        
        # 4. Save results
        features_file = output_dir / f"{test_file.stem}_features.json"
        spec_file = output_dir / f"{test_file.stem}_spectrogram.npy"
        
        with open(features_file, 'w') as f:
            json.dump(features, f, indent=2)
        
        np.save(spec_file, spectrogram)
        
        # Verify files exist and contain expected data
        assert features_file.exists()
        assert spec_file.exists()
        
        # Reload and verify
        with open(features_file, 'r') as f:
            loaded_features = json.load(f)
        
        loaded_spec = np.load(spec_file)
        
        assert loaded_features == features
        assert np.array_equal(loaded_spec, spectrogram)
    
    def test_batch_processing_pipeline(self, setup_test_environment):
        """Test batch processing of multiple files."""
        input_dir, output_dir, audio_files = setup_test_environment
        
        # Use batch processor
        processor = BatchProcessor()
        
        # Process with features and spectrograms
        results = processor.process_directory(
            str(input_dir),
            str(output_dir),
            extract_features=True,
            generate_spectrograms=True,
            spectrogram_params={'n_mels': 64, 'hop_length': 1024}
        )
        
        assert 'processed_files' in results
        assert 'failed_files' in results
        assert 'statistics' in results
        
        processed_files = results['processed_files']
        assert len(processed_files) == len(audio_files)
        
        # Check that all output files were created
        for audio_file in audio_files:
            features_file = output_dir / f"{audio_file.stem}_features.json"
            spec_file = output_dir / f"{audio_file.stem}_spectrogram.npy"
            
            assert features_file.exists()
            assert spec_file.exists()
            
            # Verify content structure
            with open(features_file, 'r') as f:
                features = json.load(f)
            
            assert isinstance(features, dict)
            assert 'processing_info' in features
            
            spectrogram = np.load(spec_file)
            assert isinstance(spectrogram, np.ndarray)
            assert spectrogram.shape[0] == 64  # Custom n_mels
    
    def test_manifest_creation_and_loading(self, setup_test_environment):
        """Test dataset manifest creation and loading."""
        input_dir, output_dir, audio_files = setup_test_environment
        
        processor = BatchProcessor()
        
        # Process files and create manifest
        results = processor.process_directory(
            str(input_dir),
            str(output_dir),
            extract_features=True,
            generate_spectrograms=True
        )
        
        # Create manifest
        manifest_file = processor.create_dataset_manifest(
            str(output_dir),
            "test_dataset"
        )
        
        assert manifest_file.exists()
        
        # Load and verify manifest
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        assert 'dataset_info' in manifest
        assert 'files' in manifest
        
        dataset_info = manifest['dataset_info']
        assert 'total_files' in dataset_info
        assert 'total_duration' in dataset_info
        assert 'created_at' in dataset_info
        
        files_info = manifest['files']
        assert len(files_info) == len(audio_files)
        
        # Verify each file entry
        for file_info in files_info:
            assert 'filename' in file_info
            assert 'features_file' in file_info
            assert 'spectrogram_file' in file_info
            assert 'audio_info' in file_info
            
            # Check that referenced files exist
            features_path = output_dir / file_info['features_file']
            spec_path = output_dir / file_info['spectrogram_file']
            
            assert features_path.exists()
            assert spec_path.exists()
    
    def test_pipeline_error_handling(self, setup_test_environment):
        """Test pipeline error handling with problematic files."""
        input_dir, output_dir, audio_files = setup_test_environment
        
        # Create an invalid audio file
        invalid_file = input_dir / "invalid.wav"
        with open(invalid_file, 'w') as f:
            f.write("This is not an audio file")
        
        # Create an empty file
        empty_file = input_dir / "empty.wav"
        empty_file.touch()
        
        processor = BatchProcessor()
        
        # Process directory with invalid files
        results = processor.process_directory(
            str(input_dir),
            str(output_dir),
            extract_features=True,
            generate_spectrograms=True
        )
        
        # Should have some failed files
        assert len(results['failed_files']) >= 2  # invalid and empty files
        assert len(results['processed_files']) == len(audio_files)  # Original valid files
        
        # Check that valid files were still processed
        for audio_file in audio_files:
            features_file = output_dir / f"{audio_file.stem}_features.json"
            assert features_file.exists()
    
    def test_audio_utilities_integration(self, setup_test_environment):
        """Test audio utilities with the pipeline."""
        input_dir, output_dir, audio_files = setup_test_environment
        test_file = audio_files[1]  # complex harmonics
        
        # Load audio
        loader = AudioLoader()
        audio_data, sample_rate = loader.load_audio(str(test_file))
        
        # Test audio utilities
        normalized = audio_utils.normalize_audio(audio_data)
        assert np.max(np.abs(normalized)) <= 1.0
        
        # Test audio properties
        is_mono = audio_utils.is_mono(audio_data)
        assert is_mono  # Our test files are mono
        
        duration = audio_utils.get_duration(audio_data, sample_rate)
        assert duration == pytest.approx(3.0, rel=0.1)
        
        # Test validation
        is_valid = audio_utils.validate_audio_file(str(test_file))
        assert is_valid
        
        # Test with invalid file
        invalid_file = input_dir / "not_audio.txt"
        with open(invalid_file, 'w') as f:
            f.write("text")
        
        is_valid_invalid = audio_utils.validate_audio_file(str(invalid_file))
        assert not is_valid_invalid
    
    def test_visualization_integration(self, setup_test_environment):
        """Test visualization utilities with real data."""
        input_dir, output_dir, audio_files = setup_test_environment
        test_file = audio_files[0]  # sine wave
        
        # Load and process audio
        loader = AudioLoader()
        audio_data, sample_rate = loader.load_audio(str(test_file))
        
        spec_gen = SpectrogramGenerator()
        spectrogram = spec_gen.generate_mel_spectrogram(audio_data, sample_rate)
        
        extractor = FeatureExtractor()
        features = extractor.extract_all_features(audio_data, sample_rate)
        
        # Test visualization functions (should not raise errors)
        try:
            # Test waveform plotting
            visualization.plot_waveform(audio_data, sample_rate, title="Test Waveform")
            
            # Test spectrogram plotting
            visualization.plot_spectrogram(spectrogram, sample_rate, title="Test Spectrogram")
            
            # Test feature plotting
            feature_subset = {k: v for k, v in features.items() 
                            if isinstance(v, (int, float)) and not np.isnan(v)}
            if feature_subset:
                visualization.plot_feature_distribution(feature_subset, title="Test Features")
            
            # Test comparison plots
            visualization.compare_spectrograms([spectrogram, spectrogram], 
                                             ["Original", "Copy"])
            
        except Exception as e:
            pytest.fail(f"Visualization failed: {e}")
    
    def test_fixed_length_spectrogram_pipeline(self, setup_test_environment):
        """Test pipeline with fixed-length spectrograms."""
        input_dir, output_dir, audio_files = setup_test_environment
        
        processor = BatchProcessor()
        fixed_length = 100  # Fixed number of time frames
        
        # Process with fixed length
        results = processor.process_directory(
            str(input_dir),
            str(output_dir),
            extract_features=True,
            generate_spectrograms=True,
            spectrogram_params={
                'n_mels': 128,
                'hop_length': 512,
                'fixed_length': fixed_length
            }
        )
        
        # Check that all spectrograms have the same shape
        for audio_file in audio_files:
            spec_file = output_dir / f"{audio_file.stem}_spectrogram.npy"
            spectrogram = np.load(spec_file)
            
            assert spectrogram.shape == (128, fixed_length)
    
    def test_comprehensive_feature_consistency(self, setup_test_environment):
        """Test that feature extraction is consistent across different processing methods."""
        input_dir, output_dir, audio_files = setup_test_environment
        test_file = audio_files[0]
        
        # Method 1: Direct processing
        loader = AudioLoader()
        audio_data, sample_rate = loader.load_audio(str(test_file))
        
        extractor = FeatureExtractor()
        features1 = extractor.extract_all_features(audio_data, sample_rate)
        
        # Method 2: Batch processing
        processor = BatchProcessor()
        results = processor.process_directory(
            str(input_dir),
            str(output_dir),
            extract_features=True,
            generate_spectrograms=False
        )
        
        features_file = output_dir / f"{test_file.stem}_features.json"
        with open(features_file, 'r') as f:
            features2 = json.load(f)
        
        # Compare key features (excluding processing_info which may differ)
        key_features = ['duration', 'sample_rate', 'tempo', 'spectral_centroid_mean']
        
        for feature in key_features:
            if feature in features1 and feature in features2:
                if isinstance(features1[feature], (int, float)):
                    assert features1[feature] == pytest.approx(features2[feature], rel=0.01)
                else:
                    assert features1[feature] == features2[feature]


if __name__ == "__main__":
    pytest.main([__file__])
