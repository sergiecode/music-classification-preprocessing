"""
Setup script for music-classification-preprocessing package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="music-classification-preprocessing",
    version="1.0.0",
    author="Sergie Code",
    author_email="sergieCode@example.com",
    description="A comprehensive audio preprocessing pipeline for music classification tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/music-classification-preprocessing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "librosa>=0.10.1",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "pandas>=2.0.0",
        "soundfile>=0.12.0",
        "pydub>=0.25.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.15.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "music-preprocess=src.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/music-classification-preprocessing/issues",
        "Source": "https://github.com/yourusername/music-classification-preprocessing",
        "Documentation": "https://github.com/yourusername/music-classification-preprocessing#readme",
    },
)
