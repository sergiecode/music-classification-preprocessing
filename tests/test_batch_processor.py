"""
Unit tests for BatchProcessor class.
"""

import pytest
import numpy as np
import sys
import os
import tempfile
import json
import soundfile as sf
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from batch_processor import BatchProcessor


class TestBatchProcessor:
    """Test cases for BatchProcessor class."""
    
    @pytest.fixture
    def temp_directories(self):
        """Create temporary input and output directories."""
        temp_dir = tempfile.mkdtemp()
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        
        input_dir.mkdir()
        output_dir.mkdir()
        
        return input_dir, output_dir
    
    @pytest.fixture
    def sample_audio_files(self, temp_directories):
        """Create sample audio files for testing."""
        input_dir, output_dir = temp_directories
        
        audio_files = []
        
        # Create multiple test audio files
        for i in range(3):
            duration = 2.0
            sample_rate = 22050
            t = np.linspace(0, duration, int(duration * sample_rate))
            
            # Different frequencies for each file
            freq = 220 * (2 ** i)  # A3, A4, A5
            audio_data = 0.5 * np.sin(2 * np.pi * freq * t)
            
            file_path = input_dir / f"test_audio_{i+1}.wav"
            sf.write(str(file_path), audio_data, sample_rate)
            audio_files.append(file_path)
        
        return audio_files, input_dir, output_dir
    
    def test_initialization(self, temp_directories):
        """Test BatchProcessor initialization."""
        input_dir, output_dir = temp_directories
        
        processor = BatchProcessor(input_dir, output_dir)
        
        assert processor.input_dir == input_dir
        assert processor.output_dir == output_dir
        assert processor.max_workers <= 8  # Should be limited
        assert (output_dir / "features").exists()
        assert (output_dir / "spectrograms").exists()
        assert (output_dir / "logs").exists()
    
    def test_custom_max_workers(self, temp_directories):
        """Test BatchProcessor with custom max_workers."""
        input_dir, output_dir = temp_directories
        
        processor = BatchProcessor(input_dir, output_dir, max_workers=4)
        
        assert processor.max_workers == 4
    
    def test_find_audio_files(self, sample_audio_files):
        """Test finding audio files in directory."""
        audio_files, input_dir, output_dir = sample_audio_files
        
        processor = BatchProcessor(input_dir, output_dir)
        found_files = processor._find_audio_files("*")
        
        assert len(found_files) == 3
        for file_path in found_files:
            assert file_path.suffix == ".wav"
            assert file_path.exists()
    
    def test_find_audio_files_pattern(self, sample_audio_files):
        """Test finding audio files with specific pattern."""
        audio_files, input_dir, output_dir = sample_audio_files
        
        processor = BatchProcessor(input_dir, output_dir)
        found_files = processor._find_audio_files("*_1.wav")
        
        assert len(found_files) == 1
        assert "test_audio_1.wav" in str(found_files[0])
    
    def test_process_single_file_features_only(self, sample_audio_files):
        """Test processing single file with features only."""
        audio_files, input_dir, output_dir = sample_audio_files
        
        processor = BatchProcessor(input_dir, output_dir)
        file_path = audio_files[0]
        
        result = processor._process_single_file(
            file_path,
            extract_features=True,
            generate_spectrograms=False,
            save_format="features"
        )
        
        assert isinstance(result, dict)
        assert 'filename' in result
        assert 'features_file' in result
        assert 'spectrogram_file' not in result
        
        # Check that features file was created
        features_file = Path(result['features_file'])
        assert features_file.exists()
        
        # Verify features content
        with open(features_file, 'r') as f:
            features = json.load(f)
        
        assert isinstance(features, dict)
        assert 'duration' in features
        assert 'sample_rate' in features
        assert 'processing_info' in features
    
    def test_process_single_file_spectrograms_only(self, sample_audio_files):
        """Test processing single file with spectrograms only."""
        audio_files, input_dir, output_dir = sample_audio_files
        
        processor = BatchProcessor(input_dir, output_dir)
        file_path = audio_files[0]
        
        result = processor._process_single_file(
            file_path,
            extract_features=False,
            generate_spectrograms=True,
            save_format="spectrograms"
        )
        
        assert isinstance(result, dict)
        assert 'filename' in result
        assert 'spectrogram_file' in result
        assert 'spectrogram_shape' in result
        assert 'features_file' not in result
        
        # Check that spectrogram file was created
        spec_file = Path(result['spectrogram_file'])
        assert spec_file.exists()
        
        # Verify spectrogram content
        spectrogram = np.load(spec_file)
        assert isinstance(spectrogram, np.ndarray)
        assert spectrogram.ndim == 2
        assert spectrogram.shape[0] == 128  # Default n_mels
    
    def test_process_single_file_both(self, sample_audio_files):
        """Test processing single file with both features and spectrograms."""
        audio_files, input_dir, output_dir = sample_audio_files
        
        processor = BatchProcessor(input_dir, output_dir)
        file_path = audio_files[0]
        
        result = processor._process_single_file(
            file_path,
            extract_features=True,
            generate_spectrograms=True,
            save_format="both"
        )
        
        assert isinstance(result, dict)
        assert 'features_file' in result
        assert 'spectrogram_file' in result
        
        # Check both files exist
        assert Path(result['features_file']).exists()
        assert Path(result['spectrogram_file']).exists()
    
    def test_process_single_file_fixed_duration(self, sample_audio_files):
        """Test processing with fixed duration spectrograms."""
        audio_files, input_dir, output_dir = sample_audio_files
        
        processor = BatchProcessor(input_dir, output_dir)
        file_path = audio_files[0]
        
        target_duration = 1.0  # 1 second
        
        result = processor._process_single_file(
            file_path,
            extract_features=False,
            generate_spectrograms=True,
            save_format="spectrograms",
            target_duration=target_duration
        )
        
        # Load spectrogram and check time dimension
        spec_file = Path(result['spectrogram_file'])
        spectrogram = np.load(spec_file)
        
        # Calculate expected frames for 1 second
        hop_length = 512
        sample_rate = 22050
        expected_frames = int(target_duration * sample_rate / hop_length) + 1
        
        # Allow some tolerance
        assert abs(spectrogram.shape[1] - expected_frames) <= 2
    
    def test_process_directory(self, sample_audio_files):
        """Test processing entire directory."""
        audio_files, input_dir, output_dir = sample_audio_files
        
        processor = BatchProcessor(input_dir, output_dir)
        
        results = processor.process_directory(
            extract_features=True,
            generate_spectrograms=True,
            save_format="both"
        )
        
        assert isinstance(results, dict)
        assert results['processed'] == 3
        assert results['errors'] == 0
        assert len(results['files']) == 3
        
        # Check that all files were processed
        for file_result in results['files']:
            assert 'features_file' in file_result
            assert 'spectrogram_file' in file_result
            assert Path(file_result['features_file']).exists()
            assert Path(file_result['spectrogram_file']).exists()
    
    def test_process_file_list(self, sample_audio_files):
        """Test processing specific file list."""
        audio_files, input_dir, output_dir = sample_audio_files
        
        processor = BatchProcessor(input_dir, output_dir)
        
        # Process only first two files
        file_list = audio_files[:2]
        
        results = processor.process_file_list(
            file_list,
            extract_features=True,
            generate_spectrograms=False,
            save_format="features"
        )
        
        assert results['processed'] == 2
        assert results['errors'] == 0
        assert len(results['files']) == 2
    
    def test_create_dataset_manifest(self, sample_audio_files):
        """Test creating dataset manifest."""
        audio_files, input_dir, output_dir = sample_audio_files
        
        processor = BatchProcessor(input_dir, output_dir)
        
        # First process the files
        results = processor.process_directory(
            extract_features=True,
            generate_spectrograms=True,
            save_format="both"
        )
        
        # Create manifest
        dataset_name = "test_dataset"
        processor.create_dataset_manifest(results, dataset_name)
        
        # Check manifest file
        manifest_path = output_dir / f"{dataset_name}_manifest.json"
        assert manifest_path.exists()
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        assert manifest['dataset_name'] == dataset_name
        assert manifest['total_files'] == 3
        assert manifest['total_errors'] == 0
        assert len(manifest['files']) == 3
        
        # Check file entries
        for file_entry in manifest['files']:
            assert 'filename' in file_entry
            assert 'duration' in file_entry
            assert 'features_file' in file_entry
            assert 'spectrogram_file' in file_entry
    
    def test_get_processing_statistics(self, sample_audio_files):
        """Test processing statistics calculation."""
        audio_files, input_dir, output_dir = sample_audio_files
        
        processor = BatchProcessor(input_dir, output_dir)
        
        results = processor.process_directory(
            extract_features=True,
            generate_spectrograms=False,
            save_format="features"
        )
        
        stats = processor.get_processing_statistics(results)
        
        assert isinstance(stats, dict)
        
        required_stats = ['total_files', 'total_duration_hours', 'average_duration_seconds',
                         'total_size_gb', 'average_file_size_mb', 'processing_success_rate',
                         'duration_distribution']
        
        for stat in required_stats:
            assert stat in stats
        
        assert stats['total_files'] == 3
        assert stats['processing_success_rate'] == 100.0
        assert stats['total_duration_hours'] > 0
        assert stats['average_duration_seconds'] > 0
    
    def test_processing_with_errors(self, temp_directories):
        """Test processing with files that cause errors."""
        input_dir, output_dir = temp_directories
        
        # Create a non-audio file that will cause an error
        bad_file = input_dir / "bad_file.wav"
        with open(bad_file, 'w') as f:
            f.write("This is not an audio file")
        
        processor = BatchProcessor(input_dir, output_dir)
        
        results = processor.process_directory(
            extract_features=True,
            generate_spectrograms=False,
            save_format="features"
        )
        
        assert results['processed'] == 0
        assert results['errors'] == 1
        assert len(results['error_files']) == 1
        assert 'bad_file.wav' in results['error_files'][0]['file']
    
    def test_save_processing_summary(self, sample_audio_files):
        """Test saving processing summary."""
        audio_files, input_dir, output_dir = sample_audio_files
        
        processor = BatchProcessor(input_dir, output_dir)
        
        results = processor.process_directory(
            extract_features=True,
            generate_spectrograms=False,
            save_format="features"
        )
        
        # Check that summary files were created
        summary_path = output_dir / "logs" / "processing_summary.json"
        stats_path = output_dir / "logs" / "processing_statistics.json"
        
        assert summary_path.exists()
        assert stats_path.exists()
        
        # Verify summary content
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        assert summary['processed'] == 3
        assert summary['errors'] == 0
        
        # Verify statistics content
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        assert 'total_files' in stats
        assert 'processing_success_rate' in stats
    
    def test_empty_directory(self, temp_directories):
        """Test processing empty directory."""
        input_dir, output_dir = temp_directories
        
        processor = BatchProcessor(input_dir, output_dir)
        
        results = processor.process_directory()
        
        assert results['processed'] == 0
        assert results['errors'] == 0
        assert len(results['files']) == 0


if __name__ == "__main__":
    pytest.main([__file__])
