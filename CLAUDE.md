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

## Advanced Classification Tracks

Audio2DTX includes 6 advanced drum classification tracks, each implementing different approaches to improve instrument recognition accuracy and robustness. These tracks can be selected using CLI flags to test and compare different classification strategies.

### Track 3: Magenta-Only Classification (`--use-magenta-only`)

**Purpose**: Simplified approach using only Google's Magenta OaF (Onsets and Frames) Drums model for classification.

**Usage**:
```bash
python main.py song.mp3 --use-magenta-only --title "MagentaOnly_Track3"
docker-compose run --rm audio2dtx-main song.mp3 --use-magenta-only --batch
```

**Technical Implementation**:
- Eliminates hybrid voting complexity by using only Magenta service
- Direct communication with Magenta microservice via HTTP API
- Simplified classification pipeline with consistent results
- Fallback to frequency-based classification if Magenta is unavailable

**Key Features**:
- Single-method approach for consistency
- Reduced classification complexity
- Magenta service integration via Docker Compose
- Clean separation from hybrid approaches

**Expected Benefits**:
- 20-30% improvement in consistency and reliability
- Eliminates voting conflicts between different methods
- More predictable and stable results
- Simpler debugging and analysis

**Use Cases**: Best for scenarios requiring consistent, predictable results where simplicity is preferred over maximum accuracy.

---

### Track 4: Advanced Spectral Features + Context (`--use-advanced-features`)

**Purpose**: Enhanced frequency-based classification using 139 advanced spectral features with contextual analysis.

**Usage**:
```bash
python main.py song.mp3 --use-advanced-features --title "AdvancedFeatures_Track4"
docker-compose run --rm audio2dtx-main song.mp3 --use-advanced-features --batch
```

**Technical Implementation**:
- **AdvancedSpectralFeatureExtractor**: 139 features (expanded from 47 baseline)
- **MFCC Features**: 13 coefficients with delta and delta-delta variations (39 features)
- **Spectral Statistics**: Kurtosis, skewness, entropy for distribution analysis
- **Beat-Synchronous Extraction**: Features aligned with detected beat positions
- **Multi-Band Analysis**: Low, mid, high frequency band energy analysis
- **Random Forest Classifier**: Trained on extracted features with grid search optimization

**Key Features**:
- Mel-frequency cepstral coefficients (MFCC) with temporal derivatives
- Spectral contrast, chroma, and tonnetz features
- Enhanced contextual pattern analysis (kick-snare alternation detection)
- Beat-position aware feature extraction
- Advanced statistical analysis of spectral characteristics

**Expected Benefits**:
- 25-35% improvement in frequency-based classification
- Better discrimination between similar instruments (e.g., snare vs. rim shot)
- More robust to different recording conditions
- Enhanced temporal context understanding

**Use Cases**: Ideal for complex recordings with challenging instrument separation or when maximum feature-based accuracy is needed.

---

### Track 5: Multi-Scale Temporal Analysis (`--use-multi-scale`)

**Purpose**: Analyzes audio at multiple temporal scales to capture both fast transients and longer decay characteristics.

**Usage**:
```bash
python main.py song.mp3 --use-multi-scale --title "MultiScale_Track5"
docker-compose run --rm audio2dtx-main song.mp3 --use-multi-scale --batch
```

**Technical Implementation**:
- **MultiScaleTemporalAnalyzer**: 4 temporal windows (25ms, 50ms, 100ms, 200ms)
- **Scale-Specific Classifiers**: Separate Random Forest models for each temporal scale
- **Learned Weight Combination**: Optimized weights for each scale per instrument class
- **Scale-Specific Onset Detection**: Different onset detection strategies per scale
- **Temporal Feature Fusion**: Combined features across all scales

**Key Features**:
- 25ms windows: Fast transients (hi-hat, snare attacks)
- 50ms windows: Standard onset detection (balanced approach)
- 100ms windows: Sustained instruments (toms, ride cymbal)
- 200ms windows: Long decays (crash cymbals, floor toms)
- Intelligent scale weighting based on instrument characteristics
- Diversity enhancement to prevent mono-classification

**Expected Benefits**:
- 20-30% improvement in onset detection and classification
- Better capture of instrument-specific temporal characteristics
- Improved detection of both short and long-duration events
- More accurate classification of sustained vs. percussive elements

**Use Cases**: Excellent for recordings with diverse drum sounds, mixed tempos, or when temporal precision is critical.

---

### Track 6: Real-Time Few-Shot Learning (`--use-few-shot`)

**Purpose**: Adapts the classification model to specific song characteristics during processing using few-shot learning techniques.

**Usage**:
```bash
python main.py song.mp3 --use-few-shot --title "FewShot_Track6"
docker-compose run --rm audio2dtx-main song.mp3 --use-few-shot --batch
```

**Technical Implementation**:
- **FewShotLearningSystem**: Dynamic adaptation during processing
- **Song Profile Initialization**: Analysis of global audio characteristics
- **Confidence-Based Adaptation**: Learning from high-confidence initial predictions
- **Instrument-Specific Strategies**: Separate adaptation for each drum type
- **Real-Time Model Updates**: Progressive improvement during processing

**Key Features**:
- Two-phase learning: initial profiling + adaptive refinement
- Confidence threshold-based adaptation (default: 0.6)
- Song-specific feature learning and threshold adjustment
- Adaptive spectral, temporal, and MFCC feature parameters
- Progressive model improvement throughout processing

**Expected Benefits**:
- 15-25% improvement in song-specific accuracy
- Better adaptation to unique drum kit characteristics
- Improved performance on recordings with distinctive sound signatures
- Dynamic learning from song-specific patterns

**Use Cases**: Perfect for recordings with unique drum sounds, unusual recording techniques, or when processing multiple songs from the same album/session.

---

### Track 7: Ensemble of Specialized Models (`--use-ensemble`)

**Purpose**: Hierarchical classification using specialized models for different instrument groups with confidence-based voting.

**Usage**:
```bash
python main.py song.mp3 --use-ensemble --title "Ensemble_Track7"
docker-compose run --rm audio2dtx-main song.mp3 --use-ensemble --batch
```

**Technical Implementation**:
- **EnsembleClassificationSystem**: Hierarchical three-tier classification
- **Specialized Classifiers**: Separate models for kick/snare, cymbals, and toms
- **Tier 1**: Broad category detection (kick/snare vs cymbals vs toms)
- **Tier 2**: Specialized classification within categories
- **Tier 3**: Confidence-based voting and conflict resolution

**Key Features**:
- **Kick/Snare Specialist**: Random Forest trained on energy and spectral features
- **Cymbal Specialist**: High-frequency analysis for crash/ride/hi-hat distinction
- **Tom Specialist**: Mid-frequency analysis for tom sub-classification
- Beat-synchronized training data extraction
- Hierarchical decision tree with confidence weighting
- Ensemble voting across specialized predictions

**Expected Benefits**:
- 25-35% improvement in overall classification accuracy
- Better distinction between similar instruments within categories
- Reduced classification errors through specialized expertise
- More reliable confidence estimates for each instrument type

**Use Cases**: Optimal for complex recordings requiring maximum classification accuracy, especially when precise instrument distinction is critical.

---

### Track 8: Data Augmentation and Preprocessing (`--use-augmentation`)

**Purpose**: Advanced preprocessing and data augmentation for improved robustness and consistency across different recording conditions.

**Usage**:
```bash
python main.py song.mp3 --use-augmentation --title "Augmentation_Track8"
docker-compose run --rm audio2dtx-main song.mp3 --use-augmentation --batch
```

**Technical Implementation**:
- **AdvancedAudioPreprocessor**: Three-stage pipeline for audio enhancement
- **Stage 1**: Advanced preprocessing (noise reduction, compression, normalization)
- **Stage 2**: Data augmentation (pitch, time, noise variants)
- **Stage 3**: Ensemble classification across augmented variants

**Key Features**:
- **Spectral Subtraction**: Noise reduction using frequency domain analysis
- **Dynamic Range Compression**: Adaptive 4:1 compression with threshold detection
- **Adaptive Normalization**: Crest factor-based normalization for optimal signal levels
- **Pitch Shifting**: ±1, ±2 semitones for pitch variation robustness
- **Time Stretching**: 0.8x-1.2x rates for temporal variation tolerance
- **Noise Addition**: Multiple noise levels for robustness testing
- **Ensemble Voting**: Main processed audio weighted 3:1 vs augmented variants

**Expected Benefits**:
- 10-20% improvement in robustness and consistency
- Better performance across different recording conditions
- Enhanced noise tolerance and signal clarity
- More stable results with varying audio quality

**Use Cases**: Essential for processing audio from diverse sources, noisy recordings, or when maximum robustness across different recording conditions is required.

---

### Track Selection Guidelines

**Choose Track 3** when you need simple, consistent results without complexity.

**Choose Track 4** for maximum feature-based accuracy with complex recordings.

**Choose Track 5** when temporal precision and diverse instrument characteristics are important.

**Choose Track 6** for recordings with unique characteristics that benefit from adaptive learning.

**Choose Track 7** when maximum overall accuracy and specialized instrument distinction is required.

**Choose Track 8** for robust processing across diverse recording conditions and audio quality levels.

**Default Behavior**: Without any track flags, the system uses the original hybrid approach combining multiple methods with weighted voting.

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

### Optional Dependencies

**Magenta Integration**: The system supports Google's Magenta library for enhanced drum classification but does not require it. If you see "Magenta not available, using fallback classification", this is normal and acceptable behavior.

- **With Magenta**: Uses Google's OaF (Onsets and Frames) Drums model for more accurate drum instrument classification (40% weight in hybrid system)
- **Without Magenta**: Uses frequency-based fallback classification (simpler but functional)
- **Installation**: Adding Magenta requires resolving complex dependency conflicts with numpy/librosa/matplotlib versions
- **Recommendation**: The fallback classification provides adequate results for most use cases

## Coding Standards

- **Language**: All code, comments, variable names, function names, and documentation must be written in English only
- **No French**: Avoid French text in code, even in comments or string literals
- **Consistent Naming**: Use clear, descriptive English names for all identifiers

## Notes

- The application processes audio at 44.1kHz sample rate
- Minimum supported audio duration: ~10 seconds
- Output includes both DTX chart file and accompanying audio files
- Original audio can be used as BGM or separated BGM (without drums)