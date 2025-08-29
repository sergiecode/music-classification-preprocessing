"""
Command-line interface for music preprocessing pipeline.

This module provides a user-friendly CLI for common preprocessing tasks
including feature extraction, spectrogram generation, and batch processing.
"""

import argparse
import sys
from pathlib import Path
import json

from .audio_loader import AudioLoader
from .feature_extractor import FeatureExtractor
from .spectrogram_generator import SpectrogramGenerator
from .batch_processor import BatchProcessor


def extract_features_command(args):
    """Handle the extract-features command."""
    print(f"Extracting features from: {args.input}")
    
    # Initialize components
    loader = AudioLoader()
    extractor = FeatureExtractor()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load audio
        audio_data, sample_rate = loader.load_audio(input_path)
        
        # Extract features
        features = extractor.extract_all_features(audio_data, sample_rate)
        
        # Add metadata
        features.update({
            "filename": input_path.name,
            "filepath": str(input_path),
            "processing_info": {
                "extracted_by": "music-classification-preprocessing CLI",
                "feature_extractor_config": {
                    "n_mfcc": extractor.n_mfcc,
                    "n_chroma": extractor.n_chroma,
                    "n_contrast": extractor.n_contrast
                }
            }
        })
        
        # Save features
        output_file = output_dir / f"{input_path.stem}_features.json"
        with open(output_file, 'w') as f:
            json.dump(features, f, indent=2)
        
        print(f"Features saved to: {output_file}")
        print(f"Extracted {len(features)} features")
        
        # Print feature summary
        if args.summary:
            summary = extractor.extract_feature_summary(features)
            print("\nFeature Summary:")
            print(f"Total features: {summary['total_features']}")
            for category, count in summary['feature_categories'].items():
                print(f"  {category.capitalize()}: {count}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def generate_spectrograms_command(args):
    """Handle the generate-spectrograms command."""
    print(f"Generating spectrograms from: {args.input}")
    
    # Initialize components
    loader = AudioLoader()
    generator = SpectrogramGenerator(
        n_mels=args.n_mels,
        hop_length=args.hop_length
    )
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if input_path.is_file():
            # Process single file
            audio_data, sample_rate = loader.load_audio(input_path)
            
            if args.fixed_length:
                spectrogram = generator.generate_fixed_length_spectrogram(
                    audio_data, sample_rate, args.fixed_length
                )
            else:
                spectrogram = generator.generate_mel_spectrogram(audio_data, sample_rate)
            
            # Save spectrogram
            output_file = output_dir / f"{input_path.stem}_spectrogram.npy"
            generator.save_spectrogram(spectrogram, output_file)
            
            print(f"Spectrogram saved to: {output_file}")
            print(f"Spectrogram shape: {spectrogram.shape}")
            
            # Save image if requested
            if args.save_image:
                image_file = output_dir / f"{input_path.stem}_spectrogram.png"
                generator.save_spectrogram_image(
                    spectrogram, image_file, title=f"Mel-Spectrogram: {input_path.name}"
                )
                print(f"Spectrogram image saved to: {image_file}")
        
        elif input_path.is_dir():
            # Process directory using batch processor
            processor = BatchProcessor(input_path, output_dir)
            results = processor.process_directory(
                extract_features=False,
                generate_spectrograms=True,
                save_format="spectrograms",
                target_duration=args.fixed_length
            )
            
            print(f"Processed {results['processed']} files")
            if results['errors'] > 0:
                print(f"Errors: {results['errors']}")
        
        else:
            print(f"Error: {input_path} is not a valid file or directory")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def preprocess_dataset_command(args):
    """Handle the preprocess-dataset command."""
    print(f"Preprocessing dataset from: {args.input}")
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a valid directory")
        sys.exit(1)
    
    try:
        # Initialize batch processor
        processor = BatchProcessor(input_dir, output_dir)
        
        # Determine what to process
        save_format = "both"
        if args.features and not args.spectrograms:
            save_format = "features"
        elif args.spectrograms and not args.features:
            save_format = "spectrograms"
        
        # Process dataset
        results = processor.process_directory(
            extract_features=args.features,
            generate_spectrograms=args.spectrograms,
            save_format=save_format,
            target_duration=args.fixed_length,
            file_pattern=args.pattern or "*"
        )
        
        print("\nProcessing complete:")
        print(f"  Processed: {results['processed']} files")
        print(f"  Errors: {results['errors']} files")
        
        # Create dataset manifest
        if args.manifest:
            processor.create_dataset_manifest(results, args.manifest)
        
        # Show statistics
        if args.stats:
            stats = processor.get_processing_statistics(results)
            print("\nDataset Statistics:")
            print(f"  Total duration: {stats['total_duration_hours']:.2f} hours")
            print(f"  Average file duration: {stats['average_duration_seconds']:.2f} seconds")
            print(f"  Total size: {stats['total_size_gb']:.2f} GB")
            print(f"  Success rate: {stats['processing_success_rate']:.2f}%")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def info_command(args):
    """Handle the info command."""
    input_path = Path(args.input)
    
    try:
        loader = AudioLoader()
        
        if input_path.is_file():
            # Single file info
            info = loader.get_audio_info(input_path)
            print("Audio File Information:")
            print(f"  Filename: {info['filename']}")
            print(f"  Duration: {info['duration']:.2f} seconds")
            print(f"  Sample Rate: {info['sample_rate']} Hz")
            print(f"  Channels: {info['channels']}")
            print(f"  Format: {info['format']}")
            print(f"  File Size: {info['file_size_mb']:.2f} MB")
        
        elif input_path.is_dir():
            # Directory info
            files_info = loader.batch_load_info(input_path)
            
            if not files_info:
                print("No audio files found in directory")
                return
            
            print("Directory Information:")
            print(f"  Total files: {len(files_info)}")
            
            durations = [f['duration'] for f in files_info]
            sizes = [f['file_size_mb'] for f in files_info]
            
            print(f"  Total duration: {sum(durations) / 3600:.2f} hours")
            print(f"  Average duration: {sum(durations) / len(durations):.2f} seconds")
            print(f"  Total size: {sum(sizes) / 1024:.2f} GB")
            
            # Format distribution
            formats = {}
            for f in files_info:
                fmt = f.get('format', 'unknown')
                formats[fmt] = formats.get(fmt, 0) + 1
            
            print("  Format distribution:")
            for fmt, count in formats.items():
                print(f"    {fmt}: {count} files")
        
        else:
            print(f"Error: {input_path} is not a valid file or directory")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Music Classification Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract features from a single file
  python -m src.cli extract-features song.mp3 --output features/
  
  # Generate spectrograms for all files in a directory
  python -m src.cli generate-spectrograms audio_dir/ --output spectrograms/ --n-mels 128
  
  # Preprocess entire dataset
  python -m src.cli preprocess-dataset dataset/ --output processed/ --features --spectrograms
  
  # Get information about audio files
  python -m src.cli info audio_file.mp3
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract features command
    extract_parser = subparsers.add_parser(
        'extract-features',
        help='Extract musical features from audio file'
    )
    extract_parser.add_argument('input', help='Input audio file path')
    extract_parser.add_argument('--output', '-o', required=True, help='Output directory')
    extract_parser.add_argument('--summary', action='store_true', help='Show feature summary')
    
    # Generate spectrograms command
    spectrogram_parser = subparsers.add_parser(
        'generate-spectrograms',
        help='Generate mel-spectrograms from audio files'
    )
    spectrogram_parser.add_argument('input', help='Input audio file or directory')
    spectrogram_parser.add_argument('--output', '-o', required=True, help='Output directory')
    spectrogram_parser.add_argument('--n-mels', type=int, default=128, help='Number of mel bands')
    spectrogram_parser.add_argument('--hop-length', type=int, default=512, help='Hop length')
    spectrogram_parser.add_argument('--fixed-length', type=float, help='Fixed length in seconds')
    spectrogram_parser.add_argument('--save-image', action='store_true', help='Save spectrogram images')
    
    # Preprocess dataset command
    preprocess_parser = subparsers.add_parser(
        'preprocess-dataset',
        help='Preprocess entire dataset'
    )
    preprocess_parser.add_argument('input', help='Input directory')
    preprocess_parser.add_argument('--output', '-o', required=True, help='Output directory')
    preprocess_parser.add_argument('--features', action='store_true', help='Extract features')
    preprocess_parser.add_argument('--spectrograms', action='store_true', help='Generate spectrograms')
    preprocess_parser.add_argument('--fixed-length', type=float, help='Fixed length for spectrograms')
    preprocess_parser.add_argument('--pattern', help='File pattern (glob)')
    preprocess_parser.add_argument('--manifest', help='Create dataset manifest with given name')
    preprocess_parser.add_argument('--stats', action='store_true', help='Show processing statistics')
    
    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Get information about audio files'
    )
    info_parser.add_argument('input', help='Input audio file or directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate command handler
    if args.command == 'extract-features':
        extract_features_command(args)
    elif args.command == 'generate-spectrograms':
        generate_spectrograms_command(args)
    elif args.command == 'preprocess-dataset':
        preprocess_dataset_command(args)
    elif args.command == 'info':
        info_command(args)


if __name__ == '__main__':
    main()
