# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Audio2DTX is a Python-based tool that converts audio files into DTXMania drumming game chart files. It uses machine learning, audio processing, and signal analysis to automatically generate drum charts from audio input.

## Development Commands

### Docker-based Development (Recommended)
```bash
# Build the Docker image
make build

# Run conversion with audio file (batch mode - no interaction)
make run

# Run conversion with interactive metadata prompts
make run-interactive

# Test the application (batch mode)
make test

# Clean up Docker images and output
make clean
```

### Direct Python Development
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the application (interactive mode)
python main.py <input_audio_file>

# Run the application (batch mode with defaults)
python main.py <input_audio_file> --batch
```

## Architecture Overview

### Core Components

1. **Main Entry Point (`main.py`)**: 
   - Handles command-line arguments and file validation
   - Orchestrates the conversion pipeline
   - Validates input audio formats: `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`

2. **Audio Processing Pipeline (`audio_to_chart.py`)**:
   - **AudioToChart class**: Main processing class with hybrid onset detection
   - **Audio Separation**: Uses Spleeter to separate drums from other instruments
   - **Tempo Detection**: Librosa-based tempo and beat detection with dynamic tempo curves
   - **Onset Detection**: Multi-method approach combining:
     - Enhanced librosa onset detection (5 different methods)
     - Energy-based detection with spectral analysis
     - ML model predictions (TensorFlow/Keras)
   - **Instrument Classification**: Frequency-based classification into 10 drum types
   - **Chart Generation**: Converts detected onsets to DTX format

3. **Machine Learning Model (`PredictOnset.h5`)**:
   - Pre-trained TensorFlow model for drum onset detection
   - Processes mel-spectrograms with 0.01s time resolution
   - Outputs onset probabilities for each drum class

### Key Processing Steps

1. **Audio Preprocessing**:
   - Separate audio into 4 stems (drums, vocals, bass, other)
   - Detect tempo and beat positions
   - Generate BGM track (with/without original audio)

2. **Onset Detection Pipeline**:
   - Librosa methods: energy, HFC, complex, phase, specdiff
   - Energy analysis: RMS, spectral flux, frequency bands
   - ML model: spectrogram-based onset prediction
   - Fusion: Combine all methods with duplicate removal

3. **Instrument Classification**:
   - 10 instrument classes: Hi-hat (open/close), Snare, Bass Drum, Toms (high/low/floor), Ride (cymbal/bell), Crash
   - Frequency domain analysis with spectral features
   - Advanced features: centroid, spread, rolloff, zero-crossing rate

4. **DTX Generation**:
   - Beat quantization for musical timing
   - Adaptive bar calculation (4/4 time signature)
   - Channel mapping to DTX format
   - Simfile template integration

### File Structure

```
audio2dtx/
├── main.py                    # Entry point and CLI handling
├── audio_to_chart.py          # Core audio processing pipeline
├── requirements.txt           # Python dependencies
├── PredictOnset.h5           # Pre-trained ML model
├── SimfilesTemplate.zip      # DTX template files
├── Dockerfile               # Container configuration
├── Makefile                 # Build and run commands
├── AutoChart.ipynb          # Original Jupyter notebook
├── plan.md                  # Development roadmap
└── ideas.md                 # Future feature ideas
```

## Docker Environment

The application runs in a containerized environment with:
- Python 3.10 runtime
- FFmpeg for audio processing
- Pre-installed ML dependencies (TensorFlow, librosa, spleeter)
- Input/output volume mapping: `./input:/app/input` and `./output:/app/output`

## Dependencies

### Core Audio Processing
- `librosa==0.8.1`: Audio analysis and feature extraction
- `spleeter==2.4.2`: Audio source separation
- `soundfile==0.13.1`: Audio file I/O

### Machine Learning
- `tensorflow==2.12.1`: ML model inference
- `scikit-learn==1.2.2`: Feature processing
- `numpy==1.23.5`: Numerical computations

### Audio Utilities
- `pydub==0.25.1`: Audio format handling
- `matplotlib==3.6.3`: Visualization support

## Usage Modes

### Interactive Mode (Default)
When run in an interactive environment with TTY support, the application prompts for metadata:
- Song title
- Artist name
- Author/Charter name
- Difficulty level (1-100)
- Use original audio as BGM (yes/no)
- Time signature (4/4, 3/4, 6/8, 2/4, 5/4)
- Genre
- Custom comment

### Batch Mode
For automated processing or non-interactive environments:
- Uses default metadata values
- No user prompts
- Suitable for CI/CD pipelines and scripts
- Activated with `--batch` flag or automatically detected in non-TTY environments

### Environment Detection
The application automatically detects the runtime environment:
- Interactive: TTY available, prompts for metadata
- Non-interactive: No TTY (Docker, scripts), uses defaults
- Explicit: Use `--batch` or `--interactive` flags to override

## Current Development Status

The project is actively being enhanced with advanced features including:
- Multi-method onset detection fusion
- Improved instrument classification accuracy
- Beat-based timing quantization
- Enhanced spectral analysis features

See `plan.md` for detailed development roadmap and `ideas.md` for future enhancements.

## Testing

Test the application with various audio formats:
- Place audio files in `./input/` directory
- Run `make run` or `python main.py <filename>`
- Check output DTX files in `./output/` directory

### Error Checking
When testing or verifying that the application works correctly, ALWAYS analyze the standard output for error messages. Look for:
- Failed onset detection methods (e.g., "energy: failed", "hfc: failed")
- Python exceptions and stack traces
- Library compatibility issues (librosa, tensorflow, spleeter)
- File I/O errors
- Model loading failures

The application should complete successfully without "failed" messages in the onset detection output. If any onset detection methods show "failed" status, investigate and fix the underlying issues before considering the test successful.

## Coding Standards

- **Language**: All code, comments, variable names, function names, and documentation must be written in English only
- **No French**: Avoid French text in code, even in comments or string literals
- **Consistent Naming**: Use clear, descriptive English names for all identifiers

## Notes

- The application processes audio at 44.1kHz sample rate
- Minimum supported audio duration: ~10 seconds
- Output includes both DTX chart file and accompanying audio files
- Original audio can be used as BGM or separated BGM (without drums)