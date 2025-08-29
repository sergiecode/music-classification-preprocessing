"""
Visualization utilities for audio analysis and feature exploration.

This module provides functions for creating plots and visualizations
of audio data, features, and spectrograms.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Optional, List, Tuple, Union
from pathlib import Path
import seaborn as sns

# Set style for consistent plots
plt.style.use('default')
sns.set_palette("husl")


def plot_waveform(audio: np.ndarray,
                  sample_rate: int,
                  title: Optional[str] = None,
                  figsize: Tuple[int, int] = (12, 4),
                  save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot audio waveform.
    
    Args:
        audio: Audio time series
        sample_rate: Sample rate of audio
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=figsize)
    
    time = np.linspace(0, len(audio) / sample_rate, len(audio))
    plt.plot(time, audio, alpha=0.8)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title or 'Audio Waveform')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_spectrogram(spectrogram: np.ndarray,
                    sample_rate: int,
                    hop_length: int = 512,
                    title: Optional[str] = None,
                    figsize: Tuple[int, int] = (12, 6),
                    save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot mel-spectrogram.
    
    Args:
        spectrogram: Spectrogram data
        sample_rate: Sample rate of audio
        hop_length: Hop length used for spectrogram
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=figsize)
    
    librosa.display.specshow(
        spectrogram,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        cmap='viridis'
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(title or 'Mel-Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_comparison(features_list: List[dict],
                          labels: List[str],
                          feature_names: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (15, 10),
                          save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot comparison of features across multiple audio files.
    
    Args:
        features_list: List of feature dictionaries
        labels: Labels for each feature set
        feature_names: Specific features to plot (optional)
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
    """
    if not features_list:
        print("No features provided for comparison")
        return
    
    # Get common features across all files
    common_features = set(features_list[0].keys())
    for features in features_list[1:]:
        common_features &= set(features.keys())
    
    # Filter to numeric features only
    numeric_features = []
    for feature in common_features:
        if isinstance(features_list[0][feature], (int, float)):
            numeric_features.append(feature)
    
    # Use specified features or all numeric features
    if feature_names:
        plot_features = [f for f in feature_names if f in numeric_features]
    else:
        plot_features = sorted(numeric_features)
    
    if not plot_features:
        print("No suitable features found for plotting")
        return
    
    # Create subplot grid
    n_features = len(plot_features)
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [[axes]]
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, feature in enumerate(plot_features):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows == 1 and n_cols == 1:
            ax = axes[0][0]
        else:
            ax = axes[row, col]
        
        # Extract feature values
        values = [features[feature] for features in features_list]
        
        # Create bar plot
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, values, alpha=0.8)
        
        # Customize plot
        ax.set_title(feature.replace('_', ' ').title())
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Color bars differently
        for j, bar in enumerate(bars):
            bar.set_color(plt.cm.Set3(j / len(bars)))
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_distribution(features: dict,
                            title: Optional[str] = None,
                            figsize: Tuple[int, int] = (15, 12),
                            save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot distribution of feature values.
    
    Args:
        features: Dictionary of features
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
    """
    # Filter to numeric features
    numeric_features = {k: v for k, v in features.items() 
                       if isinstance(v, (int, float))}
    
    if not numeric_features:
        print("No numeric features found")
        return
    
    # Group features by category
    categories = {
        'Temporal': [k for k in numeric_features.keys() 
                    if any(x in k.lower() for x in ['tempo', 'beat', 'onset', 'zcr'])],
        'Spectral': [k for k in numeric_features.keys() 
                    if any(x in k.lower() for x in ['mfcc', 'spectral', 'contrast'])],
        'Harmonic': [k for k in numeric_features.keys() 
                    if any(x in k.lower() for x in ['chroma', 'key', 'tonnetz'])],
        'Statistical': [k for k in numeric_features.keys() 
                       if any(x in k.lower() for x in ['rms', 'amplitude', 'skew', 'kurt'])]
    }
    
    # Remove empty categories
    categories = {k: v for k, v in categories.items() if v}
    
    if not categories:
        # Fallback: plot all features
        feature_names = list(numeric_features.keys())
        n_features = len(feature_names)
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for i, feature in enumerate(feature_names):
            if i < len(axes):
                axes[i].bar([0], [numeric_features[feature]], alpha=0.8)
                axes[i].set_title(feature.replace('_', ' ').title())
                axes[i].set_xticks([])
        
        # Hide unused subplots
        for i in range(len(feature_names), len(axes)):
            axes[i].set_visible(False)
    
    else:
        # Plot by category
        n_categories = len(categories)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, (category, feature_list) in enumerate(categories.items()):
            if i < 4:  # Maximum 4 categories
                ax = axes[i]
                
                if feature_list:
                    values = [numeric_features[f] for f in feature_list]
                    labels = [f.replace('_', ' ').title() for f in feature_list]
                    
                    bars = ax.bar(range(len(values)), values, alpha=0.8)
                    ax.set_title(f'{category} Features')
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    ax.grid(True, alpha=0.3)
                    
                    # Color bars
                    for j, bar in enumerate(bars):
                        bar.set_color(plt.cm.Set3(j / len(bars)))
        
        # Hide unused subplots
        for i in range(n_categories, 4):
            axes[i].set_visible(False)
    
    plt.suptitle(title or 'Feature Distribution', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_chroma_features(chroma: np.ndarray,
                        sample_rate: int,
                        hop_length: int = 512,
                        title: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 6),
                        save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot chroma features over time.
    
    Args:
        chroma: Chroma feature matrix
        sample_rate: Sample rate of audio
        hop_length: Hop length used for feature extraction
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=figsize)
    
    librosa.display.specshow(
        chroma,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='chroma',
        cmap='Blues'
    )
    
    plt.colorbar()
    plt.title(title or 'Chroma Features')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Class')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_tempo_analysis(audio: np.ndarray,
                       sample_rate: int,
                       title: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 8),
                       save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot tempo analysis including onset strength and tempogram.
    
    Args:
        audio: Audio time series
        sample_rate: Sample rate of audio
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
    """
    hop_length = 512
    
    # Compute onset strength
    onset_envelope = librosa.onset.onset_strength(
        y=audio, sr=sample_rate, hop_length=hop_length
    )
    
    # Compute tempogram
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length
    )
    
    # Beat tracking
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length
    )
    
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Plot waveform
    time = np.linspace(0, len(audio) / sample_rate, len(audio))
    axes[0].plot(time, audio, alpha=0.6)
    axes[0].set_title('Waveform')
    axes[0].set_ylabel('Amplitude')
    
    # Plot onset strength
    times = librosa.frames_to_time(np.arange(len(onset_envelope)), 
                                  sr=sample_rate, hop_length=hop_length)
    axes[1].plot(times, onset_envelope, label=f'Onset Strength (Tempo: {tempo:.1f} BPM)')
    
    # Mark beats
    beat_times = librosa.frames_to_time(beats, sr=sample_rate, hop_length=hop_length)
    axes[1].vlines(beat_times, 0, onset_envelope.max(), color='red', alpha=0.8, 
                  linestyle='--', label='Beats')
    
    axes[1].set_title('Onset Strength and Beat Tracking')
    axes[1].set_ylabel('Strength')
    axes[1].legend()
    
    # Plot tempogram
    librosa.display.specshow(
        tempogram,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='tempo',
        ax=axes[2],
        cmap='coolwarm'
    )
    axes[2].set_title('Tempogram')
    
    plt.suptitle(title or 'Tempo Analysis', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_feature_correlation_matrix(features_list: List[dict],
                                    feature_names: Optional[List[str]] = None,
                                    figsize: Tuple[int, int] = (12, 10),
                                    save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Create correlation matrix of features across multiple files.
    
    Args:
        features_list: List of feature dictionaries
        feature_names: Specific features to include (optional)
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
    """
    if len(features_list) < 2:
        print("Need at least 2 feature sets for correlation analysis")
        return
    
    # Get common numeric features
    all_features = set(features_list[0].keys())
    for features in features_list[1:]:
        all_features &= set(features.keys())
    
    numeric_features = []
    for feature in all_features:
        if all(isinstance(f.get(feature), (int, float)) for f in features_list):
            numeric_features.append(feature)
    
    if feature_names:
        numeric_features = [f for f in feature_names if f in numeric_features]
    
    if len(numeric_features) < 2:
        print("Need at least 2 common numeric features")
        return
    
    # Create feature matrix
    feature_matrix = []
    for features in features_list:
        row = [features[f] for f in numeric_features]
        feature_matrix.append(row)
    
    feature_matrix = np.array(feature_matrix).T  # Features as rows
    
    # Compute correlation
    correlation = np.corrcoef(feature_matrix)
    
    # Create heatmap
    plt.figure(figsize=figsize)
    
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    
    sns.heatmap(
        correlation,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        center=0,
        square=True,
        xticklabels=[f.replace('_', ' ').title() for f in numeric_features],
        yticklabels=[f.replace('_', ' ').title() for f in numeric_features]
    )
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Convenience function for comprehensive visualization
def create_audio_analysis_report(audio: np.ndarray,
                               sample_rate: int,
                               features: dict,
                               output_dir: Union[str, Path],
                               filename_prefix: str = "audio_analysis") -> None:
    """
    Create a comprehensive audio analysis report with multiple visualizations.
    
    Args:
        audio: Audio time series
        sample_rate: Sample rate of audio
        features: Extracted features dictionary
        output_dir: Directory to save visualizations
        filename_prefix: Prefix for output filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Waveform plot
    plot_waveform(
        audio, sample_rate,
        title="Audio Waveform",
        save_path=output_dir / f"{filename_prefix}_waveform.png"
    )
    
    # Feature distribution
    plot_feature_distribution(
        features,
        title="Feature Distribution",
        save_path=output_dir / f"{filename_prefix}_features.png"
    )
    
    # Tempo analysis
    plot_tempo_analysis(
        audio, sample_rate,
        title="Tempo Analysis",
        save_path=output_dir / f"{filename_prefix}_tempo.png"
    )
    
    print(f"Audio analysis report saved to {output_dir}")


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
    print("Available functions:")
    print("- plot_waveform(): Plot audio waveform")
    print("- plot_spectrogram(): Plot mel-spectrogram")
    print("- plot_feature_comparison(): Compare features across files")
    print("- create_audio_analysis_report(): Create comprehensive report")
