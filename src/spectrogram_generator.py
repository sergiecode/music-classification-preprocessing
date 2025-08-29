"""
Mel-spectrogram generation utilities for music classification preprocessing.

This module provides tools for generating mel-spectrograms suitable for
deep learning models, particularly CNNs for audio classification.
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional, Tuple, Union
from pathlib import Path

# Set matplotlib backend for non-interactive use
matplotlib.use('Agg')


class SpectrogramGenerator:
    """
    Mel-spectrogram generator optimized for music classification tasks.
    
    Provides methods for creating mel-spectrograms with various configurations
    suitable for different machine learning approaches.
    """
    
    def __init__(self,
                 n_mels: int = 128,
                 hop_length: int = 512,
                 win_length: int = 2048,
                 n_fft: int = 2048,
                 fmin: float = 0.0,
                 fmax: Optional[float] = None):
        """
        Initialize SpectrogramGenerator with default parameters.
        
        Args:
            n_mels: Number of mel frequency bands
            hop_length: Number of samples between successive frames
            win_length: Length of the windowing function
            n_fft: Length of the FFT window
            fmin: Minimum frequency for mel filter bank
            fmax: Maximum frequency for mel filter bank (None for Nyquist)
        """
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
    
    def generate_mel_spectrogram(self,
                               audio_data: np.ndarray,
                               sample_rate: int,
                               to_db: bool = True,
                               normalize: bool = True) -> np.ndarray:
        """
        Generate mel-spectrogram from audio data.
        
        Args:
            audio_data: Input audio time series
            sample_rate: Sample rate of audio
            to_db: Whether to convert to decibel scale
            normalize: Whether to normalize the spectrogram
            
        Returns:
            Mel-spectrogram array of shape (n_mels, time_frames)
        """
        try:
            # Generate mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=sample_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.n_fft,
                fmin=self.fmin,
                fmax=self.fmax
            )
            
            # Convert to decibel scale if requested
            if to_db:
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize if requested
            if normalize:
                mel_spec = self._normalize_spectrogram(mel_spec)
            
            return mel_spec
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate mel-spectrogram: {str(e)}")
    
    def generate_fixed_length_spectrogram(self,
                                        audio_data: np.ndarray,
                                        sample_rate: int,
                                        target_length: float,
                                        to_db: bool = True,
                                        normalize: bool = True) -> np.ndarray:
        """
        Generate mel-spectrogram with fixed time dimension for batch processing.
        
        Args:
            audio_data: Input audio time series
            sample_rate: Sample rate of audio
            target_length: Target length in seconds
            to_db: Whether to convert to decibel scale
            normalize: Whether to normalize the spectrogram
            
        Returns:
            Fixed-length mel-spectrogram array
        """
        # Calculate target number of samples
        target_samples = int(target_length * sample_rate)
        
        # Pad or truncate audio to target length
        if len(audio_data) < target_samples:
            # Pad with zeros
            padding = target_samples - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), mode='constant')
        elif len(audio_data) > target_samples:
            # Truncate from center
            start_idx = (len(audio_data) - target_samples) // 2
            audio_data = audio_data[start_idx:start_idx + target_samples]
        
        # Generate mel-spectrogram
        return self.generate_mel_spectrogram(
            audio_data, sample_rate, to_db=to_db, normalize=normalize
        )
    
    def generate_log_mel_spectrogram(self,
                                   audio_data: np.ndarray,
                                   sample_rate: int) -> np.ndarray:
        """
        Generate log mel-spectrogram optimized for deep learning.
        
        Args:
            audio_data: Input audio time series
            sample_rate: Sample rate of audio
            
        Returns:
            Log mel-spectrogram array
        """
        mel_spec = self.generate_mel_spectrogram(
            audio_data, sample_rate, to_db=False, normalize=False
        )
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        log_mel_spec = np.log(mel_spec + epsilon)
        
        # Normalize
        log_mel_spec = self._normalize_spectrogram(log_mel_spec)
        
        return log_mel_spec
    
    def generate_delta_features(self,
                              spectrogram: np.ndarray,
                              order: int = 1) -> np.ndarray:
        """
        Generate delta (differential) features from spectrogram.
        
        Args:
            spectrogram: Input spectrogram
            order: Order of delta features (1 for delta, 2 for delta-delta)
            
        Returns:
            Delta features array
        """
        try:
            if order == 1:
                return librosa.feature.delta(spectrogram, order=1)
            elif order == 2:
                return librosa.feature.delta(spectrogram, order=2)
            else:
                raise ValueError("Delta order must be 1 or 2")
                
        except Exception as e:
            raise RuntimeError(f"Failed to generate delta features: {str(e)}")
    
    def generate_augmented_spectrograms(self,
                                      audio_data: np.ndarray,
                                      sample_rate: int,
                                      n_augmentations: int = 3) -> list:
        """
        Generate multiple augmented versions of mel-spectrograms for data augmentation.
        
        Args:
            audio_data: Input audio time series
            sample_rate: Sample rate of audio
            n_augmentations: Number of augmented versions to generate
            
        Returns:
            List of augmented mel-spectrograms
        """
        spectrograms = []
        
        # Original spectrogram
        original_spec = self.generate_mel_spectrogram(audio_data, sample_rate)
        spectrograms.append(original_spec)
        
        for _ in range(n_augmentations):
            # Apply random augmentations
            augmented_audio = self._apply_audio_augmentation(audio_data, sample_rate)
            augmented_spec = self.generate_mel_spectrogram(augmented_audio, sample_rate)
            spectrograms.append(augmented_spec)
        
        return spectrograms
    
    def save_spectrogram(self,
                        spectrogram: np.ndarray,
                        output_path: Union[str, Path],
                        format: str = 'npy') -> None:
        """
        Save spectrogram to file.
        
        Args:
            spectrogram: Spectrogram array to save
            output_path: Output file path
            format: Save format ('npy', 'npz', or 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'npy':
            np.save(str(output_path), spectrogram)
        elif format == 'npz':
            np.savez_compressed(str(output_path), spectrogram=spectrogram)
        elif format == 'csv':
            np.savetxt(str(output_path), spectrogram, delimiter=',')
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def save_spectrogram_image(self,
                             spectrogram: np.ndarray,
                             output_path: Union[str, Path],
                             title: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 8),
                             cmap: str = 'viridis') -> None:
        """
        Save spectrogram as an image file.
        
        Args:
            spectrogram: Spectrogram array to visualize
            output_path: Output image path
            title: Title for the plot
            figsize: Figure size (width, height)
            cmap: Colormap for visualization
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=figsize)
        
        # Create the spectrogram plot
        librosa.display.specshow(
            spectrogram,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel',
            cmap=cmap
        )
        
        plt.colorbar(format='%+2.0f dB')
        plt.title(title or 'Mel-Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Mel Frequency')
        plt.tight_layout()
        
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_spectrogram_comparison(self,
                                       spectrograms: list,
                                       labels: list,
                                       output_path: Union[str, Path]) -> None:
        """
        Create a comparison visualization of multiple spectrograms.
        
        Args:
            spectrograms: List of spectrogram arrays
            labels: List of labels for each spectrogram
            output_path: Output image path
        """
        n_specs = len(spectrograms)
        fig, axes = plt.subplots(1, n_specs, figsize=(6*n_specs, 6))
        
        if n_specs == 1:
            axes = [axes]
        
        for i, (spec, label) in enumerate(zip(spectrograms, labels)):
            librosa.display.specshow(
                spec,
                hop_length=self.hop_length,
                x_axis='time',
                y_axis='mel',
                ax=axes[i],
                cmap='viridis'
            )
            axes[i].set_title(label)
            axes[i].set_xlabel('Time (s)')
            if i == 0:
                axes[i].set_ylabel('Mel Frequency')
        
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_spectrogram_stats(self, spectrogram: np.ndarray) -> dict:
        """
        Calculate statistics of a spectrogram.
        
        Args:
            spectrogram: Input spectrogram array
            
        Returns:
            Dictionary of spectrogram statistics
        """
        stats = {
            'shape': spectrogram.shape,
            'min_value': float(np.min(spectrogram)),
            'max_value': float(np.max(spectrogram)),
            'mean_value': float(np.mean(spectrogram)),
            'std_value': float(np.std(spectrogram)),
            'total_energy': float(np.sum(spectrogram**2)),
            'spectral_density': float(np.sum(spectrogram**2) / spectrogram.size)
        }
        
        return stats
    
    def _normalize_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Normalize spectrogram to [0, 1] range.
        
        Args:
            spectrogram: Input spectrogram
            
        Returns:
            Normalized spectrogram
        """
        min_val = np.min(spectrogram)
        max_val = np.max(spectrogram)
        
        if max_val - min_val == 0:
            return np.zeros_like(spectrogram)
        
        return (spectrogram - min_val) / (max_val - min_val)
    
    def _apply_audio_augmentation(self,
                                audio_data: np.ndarray,
                                sample_rate: int) -> np.ndarray:
        """
        Apply random audio augmentations for data augmentation.
        
        Args:
            audio_data: Input audio
            sample_rate: Sample rate
            
        Returns:
            Augmented audio
        """
        augmented = audio_data.copy()
        
        # Random time stretching
        if np.random.random() < 0.5:
            stretch_factor = np.random.uniform(0.8, 1.2)
            augmented = librosa.effects.time_stretch(augmented, rate=stretch_factor)
        
        # Random pitch shifting
        if np.random.random() < 0.5:
            pitch_shift = np.random.uniform(-2, 2)
            augmented = librosa.effects.pitch_shift(
                augmented, sr=sample_rate, n_steps=pitch_shift
            )
        
        # Random noise addition
        if np.random.random() < 0.3:
            noise_factor = np.random.uniform(0.001, 0.01)
            noise = np.random.normal(0, noise_factor, len(augmented))
            augmented = augmented + noise
        
        return augmented


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    generator = SpectrogramGenerator()
    
    print("SpectrogramGenerator initialized successfully!")
    print(f"Mel bands: {generator.n_mels}")
    print(f"Hop length: {generator.hop_length}")
    print(f"FFT size: {generator.n_fft}")
