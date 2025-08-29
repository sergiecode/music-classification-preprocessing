"""
Batch processing utilities for large-scale audio preprocessing.

This module provides tools for processing multiple audio files efficiently,
with support for parallel processing and progress tracking.
"""

import json
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from tqdm import tqdm
import multiprocessing

from audio_loader import AudioLoader
from feature_extractor import FeatureExtractor  
from spectrogram_generator import SpectrogramGenerator
class BatchProcessor:
    """
    Batch processor for large-scale audio preprocessing operations.
    
    Handles directory-level processing with parallel execution,
    progress tracking, and error handling.
    """
    
    def __init__(self,
                 input_dir: Optional[Union[str, Path]] = None,
                 output_dir: Optional[Union[str, Path]] = None,
                 max_workers: Optional[int] = None):
        """
        Initialize BatchProcessor.
        
        Args:
            input_dir: Directory containing input audio files (can be set later)
            output_dir: Directory for processed output files (can be set later)
            max_workers: Maximum number of parallel workers (None for auto)
        """
        self.input_dir = Path(input_dir) if input_dir else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.max_workers = max_workers or min(8, multiprocessing.cpu_count())
        
        # Initialize processors
        self.audio_loader = AudioLoader()
        self.feature_extractor = FeatureExtractor()
        self.spectrogram_generator = SpectrogramGenerator()
        
        # Create output directories if specified
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "features").mkdir(exist_ok=True)
            (self.output_dir / "spectrograms").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
    
    def process_directory(self,
                         extract_features: bool = True,
                         generate_spectrograms: bool = True,
                         save_format: str = "both",
                         target_duration: Optional[float] = None,
                         file_pattern: str = "*") -> Dict[str, Any]:
        """
        Process all audio files in the input directory.
        
        Args:
            extract_features: Whether to extract musical features
            generate_spectrograms: Whether to generate mel-spectrograms
            save_format: Format for saving ('features', 'spectrograms', or 'both')
            target_duration: Fixed duration for spectrograms (None for original length)
            file_pattern: Glob pattern for file selection
            
        Returns:
            Dictionary with processing results and statistics
        """
        # Find all audio files
        audio_files = self._find_audio_files(file_pattern)
        
        if not audio_files:
            print(f"No audio files found in {self.input_dir}")
            return {"processed": 0, "errors": 0, "files": []}
        
        print(f"Found {len(audio_files)} audio files to process")
        
        # Process files
        results = {
            "processed": 0,
            "errors": 0,
            "files": [],
            "error_files": [],
            "processing_stats": {}
        }
        
        # Use single-threaded processing to avoid potential issues with librosa
        for file_path in tqdm(audio_files, desc="Processing audio files"):
            try:
                file_result = self._process_single_file(
                    file_path,
                    extract_features=extract_features,
                    generate_spectrograms=generate_spectrograms,
                    save_format=save_format,
                    target_duration=target_duration
                )
                
                results["files"].append(file_result)
                results["processed"] += 1
                
            except Exception as e:
                error_info = {
                    "file": str(file_path),
                    "error": str(e)
                }
                results["error_files"].append(error_info)
                results["errors"] += 1
                print(f"Error processing {file_path}: {e}")
        
        # Save processing summary
        self._save_processing_summary(results)
        
        print(f"Processing complete: {results['processed']} files processed, {results['errors']} errors")
        
        return results
    
    def process_file_list(self,
                         file_list: List[Union[str, Path]],
                         extract_features: bool = True,
                         generate_spectrograms: bool = True,
                         save_format: str = "both",
                         target_duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Process a specific list of audio files.
        
        Args:
            file_list: List of audio file paths to process
            extract_features: Whether to extract musical features
            generate_spectrograms: Whether to generate mel-spectrograms
            save_format: Format for saving ('features', 'spectrograms', or 'both')
            target_duration: Fixed duration for spectrograms (None for original length)
            
        Returns:
            Dictionary with processing results and statistics
        """
        results = {
            "processed": 0,
            "errors": 0,
            "files": [],
            "error_files": []
        }
        
        for file_path in tqdm(file_list, desc="Processing audio files"):
            try:
                file_result = self._process_single_file(
                    Path(file_path),
                    extract_features=extract_features,
                    generate_spectrograms=generate_spectrograms,
                    save_format=save_format,
                    target_duration=target_duration
                )
                
                results["files"].append(file_result)
                results["processed"] += 1
                
            except Exception as e:
                error_info = {
                    "file": str(file_path),
                    "error": str(e)
                }
                results["error_files"].append(error_info)
                results["errors"] += 1
                print(f"Error processing {file_path}: {e}")
        
        return results
    
    def create_dataset_manifest(self,
                              results: Dict[str, Any],
                              dataset_name: str = "music_dataset") -> None:
        """
        Create a dataset manifest file with metadata for all processed files.
        
        Args:
            results: Results from batch processing
            dataset_name: Name of the dataset
        """
        manifest = {
            "dataset_name": dataset_name,
            "created_by": "music-classification-preprocessing",
            "total_files": results["processed"],
            "total_errors": results["errors"],
            "processing_date": str(Path().cwd()),
            "files": []
        }
        
        for file_info in results["files"]:
            manifest["files"].append({
                "filename": file_info["filename"],
                "duration": file_info.get("duration", 0),
                "sample_rate": file_info.get("sample_rate", 0),
                "features_file": file_info.get("features_file"),
                "spectrogram_file": file_info.get("spectrogram_file"),
                "file_size_mb": file_info.get("file_size_mb", 0)
            })
        
        manifest_path = self.output_dir / f"{dataset_name}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Dataset manifest saved to {manifest_path}")
    
    def get_processing_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate processing statistics from results.
        
        Args:
            results: Results from batch processing
            
        Returns:
            Dictionary with detailed statistics
        """
        if not results["files"]:
            return {"error": "No processed files found"}
        
        durations = [f.get("duration", 0) for f in results["files"]]
        file_sizes = [f.get("file_size_mb", 0) for f in results["files"]]
        
        stats = {
            "total_files": len(results["files"]),
            "total_duration_hours": sum(durations) / 3600,
            "average_duration_seconds": np.mean(durations),
            "total_size_gb": sum(file_sizes) / 1024,
            "average_file_size_mb": np.mean(file_sizes),
            "processing_success_rate": results["processed"] / (results["processed"] + results["errors"]) * 100,
            "duration_distribution": {
                "min": min(durations),
                "max": max(durations),
                "median": np.median(durations),
                "std": np.std(durations)
            }
        }
        
        return stats
    
    def _process_single_file(self,
                           file_path: Path,
                           extract_features: bool,
                           generate_spectrograms: bool,
                           save_format: str,
                           target_duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Process a single audio file.
        
        Args:
            file_path: Path to audio file
            extract_features: Whether to extract features
            generate_spectrograms: Whether to generate spectrograms
            save_format: Save format
            target_duration: Target duration for spectrograms
            
        Returns:
            Dictionary with processing results for the file
        """
        # Load audio
        audio_data, sample_rate = self.audio_loader.load_audio(file_path)
        
        # Get file info
        file_info = self.audio_loader.get_audio_info(file_path)
        
        result = {
            "filename": file_path.name,
            "filepath": str(file_path),
            "duration": file_info["duration"],
            "sample_rate": sample_rate,
            "file_size_mb": file_info["file_size_mb"]
        }
        
        # Extract features if requested
        if extract_features and save_format in ["features", "both"]:
            features = self.feature_extractor.extract_all_features(audio_data, sample_rate)
            
            # Add file metadata to features
            features.update({
                "filename": file_path.name,
                "filepath": str(file_path),
                "processing_info": {
                    "extracted_by": "music-classification-preprocessing",
                    "feature_extractor_config": {
                        "n_mfcc": self.feature_extractor.n_mfcc,
                        "n_chroma": self.feature_extractor.n_chroma,
                        "n_contrast": self.feature_extractor.n_contrast
                    }
                }
            })
            
            # Save features
            features_filename = f"{file_path.stem}_features.json"
            features_path = self.output_dir / "features" / features_filename
            
            with open(features_path, 'w') as f:
                json.dump(features, f, indent=2)
            
            result["features_file"] = str(features_path)
        
        # Generate spectrograms if requested
        if generate_spectrograms and save_format in ["spectrograms", "both"]:
            if target_duration:
                spectrogram = self.spectrogram_generator.generate_fixed_length_spectrogram(
                    audio_data, sample_rate, target_duration
                )
            else:
                spectrogram = self.spectrogram_generator.generate_mel_spectrogram(
                    audio_data, sample_rate
                )
            
            # Save spectrogram
            spectrogram_filename = f"{file_path.stem}_spectrogram.npy"
            spectrogram_path = self.output_dir / "spectrograms" / spectrogram_filename
            
            self.spectrogram_generator.save_spectrogram(spectrogram, spectrogram_path)
            
            result["spectrogram_file"] = str(spectrogram_path)
            result["spectrogram_shape"] = spectrogram.shape
        
        return result
    
    def _find_audio_files(self, pattern: str) -> List[Path]:
        """
        Find all audio files matching the pattern.
        
        Args:
            pattern: Glob pattern for file matching
            
        Returns:
            List of audio file paths
        """
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aiff'}
        
        all_files = list(self.input_dir.rglob(pattern))
        audio_files = [f for f in all_files 
                      if f.is_file() and f.suffix.lower() in audio_extensions]
        
        return sorted(audio_files)
    
    def _save_processing_summary(self, results: Dict[str, Any]) -> None:
        """
        Save processing summary to log file.
        
        Args:
            results: Processing results
        """
        summary_path = self.output_dir / "logs" / "processing_summary.json"
        
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Also save statistics
        if results["files"]:
            stats = self.get_processing_statistics(results)
            stats_path = self.output_dir / "logs" / "processing_statistics.json"
            
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("BatchProcessor module loaded successfully!")
    print("Available functions:")
    print("- process_directory(): Process all files in a directory")
    print("- process_file_list(): Process a specific list of files") 
    print("- create_dataset_manifest(): Create dataset metadata")
    print("- get_processing_statistics(): Calculate processing stats")
