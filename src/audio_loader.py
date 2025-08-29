"""
Audio loading utilities for music preprocessing pipeline.

This module provides robust audio file loading with standardization
for machine learning preprocessing tasks.
"""

import librosa
import soundfile as sf
import numpy as np
from typing import Tuple, Optional, Union
import warnings
from pathlib import Path


class AudioLoader:
    """
    Audio file loader with standardization capabilities.
    
    Handles various audio formats and provides consistent output
    for downstream processing tasks.
    """
    
    def __init__(self, 
                 target_sr: int = 22050,
                 mono: bool = True,
                 normalize: bool = True):
        """
        Initialize AudioLoader with default parameters.
        
        Args:
            target_sr (int): Target sample rate for loaded audio
            mono (bool): Whether to convert to mono
            normalize (bool): Whether to normalize audio amplitude
        """
        self.target_sr = target_sr
        self.mono = mono
        self.normalize = normalize
        
        # Supported audio formats
        self.supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aiff'}
    
    def load_audio(self, 
                   file_path: Union[str, Path],
                   duration: Optional[float] = None,
                   offset: float = 0.0) -> Tuple[np.ndarray, int]:
        """
        Load audio file with standardization.
        
        Args:
            file_path: Path to audio file
            duration: Duration to load in seconds (None for full file)
            offset: Start time offset in seconds
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format is not supported
            RuntimeError: If audio loading fails
        """
        file_path = Path(file_path)
        
        # Check file existence
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Check file format
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported audio format: {file_path.suffix}")
        
        try:
            # Load audio using librosa
            audio_data, sample_rate = librosa.load(
                str(file_path),
                sr=self.target_sr,
                mono=self.mono,
                duration=duration,
                offset=offset
            )
            
            # Normalize if requested
            if self.normalize:
                audio_data = self._normalize_audio(audio_data)
            
            # Validate audio data
            if len(audio_data) == 0:
                raise RuntimeError(f"Loaded audio is empty: {file_path}")
            
            return audio_data, self.target_sr
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {file_path}: {str(e)}")
    
    def load_audio_segment(self,
                          file_path: Union[str, Path],
                          start_time: float,
                          end_time: float) -> Tuple[np.ndarray, int]:
        """
        Load a specific segment of an audio file.
        
        Args:
            file_path: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        duration = end_time - start_time
        return self.load_audio(file_path, duration=duration, offset=start_time)
    
    def get_audio_info(self, file_path: Union[str, Path]) -> dict:
        """
        Get audio file information without loading the full file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio file metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            # Get file info using soundfile
            with sf.SoundFile(str(file_path)) as f:
                info = {
                    'filename': file_path.name,
                    'filepath': str(file_path),
                    'duration': len(f) / f.samplerate,
                    'sample_rate': f.samplerate,
                    'channels': f.channels,
                    'format': f.format,
                    'subtype': f.subtype,
                    'frames': len(f),
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024)
                }
            
            return info
            
        except Exception:
            # Fallback to librosa for more formats
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y, sr = librosa.load(str(file_path), sr=None, duration=0.1)
                    duration = librosa.get_duration(filename=str(file_path))
                
                info = {
                    'filename': file_path.name,
                    'filepath': str(file_path),
                    'duration': duration,
                    'sample_rate': sr,
                    'channels': 1 if len(y.shape) == 1 else y.shape[0],
                    'format': file_path.suffix,
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024)
                }
                
                return info
                
            except Exception as e2:
                raise RuntimeError(f"Failed to get audio info for {file_path}: {str(e2)}")
    
    def batch_load_info(self, directory: Union[str, Path]) -> list:
        """
        Get information for all audio files in a directory.
        
        Args:
            directory: Path to directory containing audio files
            
        Returns:
            List of audio file information dictionaries
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        audio_files = []
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    info = self.get_audio_info(file_path)
                    audio_files.append(info)
                except Exception as e:
                    print(f"Warning: Could not process {file_path}: {e}")
        
        return audio_files
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude to [-1, 1] range.
        
        Args:
            audio_data: Input audio array
            
        Returns:
            Normalized audio array
        """
        if np.max(np.abs(audio_data)) == 0:
            return audio_data
        
        return audio_data / np.max(np.abs(audio_data))
    
    def save_audio(self,
                   audio_data: np.ndarray,
                   output_path: Union[str, Path],
                   sample_rate: int = None) -> None:
        """
        Save audio data to file.
        
        Args:
            audio_data: Audio array to save
            output_path: Output file path
            sample_rate: Sample rate (uses instance default if None)
        """
        if sample_rate is None:
            sample_rate = self.target_sr
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sf.write(str(output_path), audio_data, sample_rate)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    loader = AudioLoader()
    
    # Create example usage (would need actual audio file)
    print("AudioLoader initialized successfully!")
    print(f"Target sample rate: {loader.target_sr}")
    print(f"Supported formats: {loader.supported_formats}")
