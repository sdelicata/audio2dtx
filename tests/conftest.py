"""
Pytest configuration and fixtures for audio2dtx tests.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

# Add src to path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio2dtx.config.settings import Settings
from audio2dtx.classification.base_classifier import ClassificationResult


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        audio=Settings.AudioSettings(
            sample_rate=22050,
            chunk_size=1024,
            hop_length=512,
            n_fft=2048
        ),
        classification=Settings.ClassificationSettings(
            confidence_threshold=0.5,
            voting_method='weighted_confidence',
            feature_cache_size=1000
        ),
        output=Settings.OutputSettings(
            dtx_format='v120',
            include_bgm=True,
            include_preview=False
        ),
        logging=Settings.LoggingSettings(
            level='INFO',
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )


@pytest.fixture
def test_audio_mono():
    """Create test mono audio signal."""
    duration = 1.0  # 1 second
    sample_rate = 22050
    samples = int(duration * sample_rate)
    
    # Create a simple test signal with some frequency content
    t = np.linspace(0, duration, samples)
    signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    signal += 0.5 * np.sin(2 * np.pi * 880 * t)  # Add harmonic
    signal += 0.1 * np.random.randn(samples)  # Add noise
    
    return signal.astype(np.float32)


@pytest.fixture
def test_audio_stereo():
    """Create test stereo audio signal."""
    duration = 1.0
    sample_rate = 22050
    samples = int(duration * sample_rate)
    
    # Create stereo signal
    t = np.linspace(0, duration, samples)
    left = np.sin(2 * np.pi * 440 * t)
    right = np.sin(2 * np.pi * 880 * t)
    
    stereo = np.column_stack([left, right])
    return stereo.astype(np.float32)


@pytest.fixture
def test_audio_file(test_audio_mono):
    """Create temporary test audio file."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        # Simple WAV file creation (mock)
        np.save(f.name + '.npy', test_audio_mono)
        yield f.name + '.npy'
        
        # Cleanup
        try:
            os.unlink(f.name + '.npy')
        except:
            pass


@pytest.fixture
def mock_onset_times():
    """Create mock onset times."""
    return np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])


@pytest.fixture
def mock_beat_times():
    """Create mock beat times."""
    return np.array([0.0, 0.5, 1.0, 1.5, 2.0])


@pytest.fixture
def mock_classification_results():
    """Create mock classification results."""
    results = []
    instruments = ['kick', 'snare', 'hi-hat-close', 'crash', 'tom-high']
    
    for i in range(5):
        result = ClassificationResult(
            instrument=instruments[i],
            confidence=0.8 + i * 0.02,
            velocity=0.5 + i * 0.1,
            features={
                'source': 'test',
                'test_index': i,
                'spectral_centroid': 1000 + i * 100
            }
        )
        results.append(result)
    
    return results


@pytest.fixture
def mock_audio_metadata():
    """Create mock audio metadata."""
    return {
        'title': 'Test Song',
        'artist': 'Test Artist',
        'duration': 180.0,
        'sample_rate': 22050,
        'channels': 1,
        'format': 'wav'
    }


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_magenta_client():
    """Create mock Magenta client."""
    mock_client = Mock()
    mock_client.is_available.return_value = True
    mock_client.classify_audio.return_value = {
        'kick': 0.8,
        'snare': 0.2,
        'hi-hat-close': 0.1,
        'crash': 0.05
    }
    mock_client.get_connection_info.return_value = {
        'status': 'connected',
        'version': '1.0.0'
    }
    return mock_client


@pytest.fixture
def mock_spleeter_separator():
    """Create mock Spleeter separator."""
    mock_separator = Mock()
    mock_separator.is_available.return_value = True
    mock_separator.separate_audio.return_value = {
        'drums': np.random.randn(22050).astype(np.float32),
        'vocals': np.random.randn(22050).astype(np.float32),
        'bass': np.random.randn(22050).astype(np.float32),
        'other': np.random.randn(22050).astype(np.float32)
    }
    return mock_separator


@pytest.fixture
def track_test_data():
    """Create test data for track testing."""
    return {
        'audio_windows': [
            (np.random.randn(1024).astype(np.float32), 0.1),
            (np.random.randn(1024).astype(np.float32), 0.3),
            (np.random.randn(1024).astype(np.float32), 0.5)
        ],
        'context': {
            'full_audio': np.random.randn(22050).astype(np.float32),
            'metadata': {'genre': 'rock', 'tempo': 120.0},
            'beat_times': [0.0, 0.5, 1.0, 1.5],
            'tempo_bpm': 120.0
        }
    }


@pytest.fixture
def performance_test_data():
    """Create larger test data for performance testing."""
    return {
        'large_audio': np.random.randn(220500).astype(np.float32),  # 10 seconds
        'many_onsets': [(np.random.randn(1024).astype(np.float32), i * 0.1) for i in range(50)],
        'context': {
            'full_audio': np.random.randn(220500).astype(np.float32),
            'metadata': {'genre': 'metal', 'tempo': 160.0},
            'beat_times': [i * 0.375 for i in range(27)],  # 160 BPM
            'tempo_bpm': 160.0
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Ensure test environment is clean
    os.environ['AUDIO2DTX_ENV'] = 'test'
    yield
    # Cleanup after test
    if 'AUDIO2DTX_ENV' in os.environ:
        del os.environ['AUDIO2DTX_ENV']


# Test utilities
def assert_classification_result(result: ClassificationResult, expected_instrument: str = None):
    """Assert that a classification result is valid."""
    assert isinstance(result, ClassificationResult)
    assert isinstance(result.instrument, str)
    assert result.instrument in [
        'kick', 'snare', 'hi-hat-close', 'hi-hat-open', 
        'tom-high', 'tom-low', 'tom-floor', 'ride-cymbal', 
        'ride-bell', 'crash'
    ]
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.velocity <= 1.0
    assert isinstance(result.features, dict)
    
    if expected_instrument:
        assert result.instrument == expected_instrument


def assert_onset_detection_result(result: Dict[str, Any]):
    """Assert that an onset detection result is valid."""
    assert isinstance(result, dict)
    assert 'onset_times' in result
    assert 'confidence' in result
    assert isinstance(result['onset_times'], np.ndarray)
    assert len(result['onset_times']) >= 0
    assert 0.0 <= result['confidence'] <= 1.0


def create_test_audio_with_drums(duration: float = 2.0, sample_rate: int = 22050) -> np.ndarray:
    """Create test audio with drum-like characteristics."""
    samples = int(duration * sample_rate)
    audio = np.zeros(samples)
    
    # Add kick drum hits (low frequency)
    kick_times = [0.0, 0.5, 1.0, 1.5]
    for kick_time in kick_times:
        if kick_time < duration:
            start_sample = int(kick_time * sample_rate)
            end_sample = min(start_sample + 1000, samples)
            
            # Create kick drum sound (low frequency burst)
            kick_samples = end_sample - start_sample
            t = np.linspace(0, kick_samples / sample_rate, kick_samples)
            kick_sound = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 20)
            audio[start_sample:end_sample] += kick_sound
    
    # Add snare hits (mid frequency)
    snare_times = [0.25, 0.75, 1.25, 1.75]
    for snare_time in snare_times:
        if snare_time < duration:
            start_sample = int(snare_time * sample_rate)
            end_sample = min(start_sample + 500, samples)
            
            # Create snare sound (mid frequency with noise)
            snare_samples = end_sample - start_sample
            t = np.linspace(0, snare_samples / sample_rate, snare_samples)
            snare_sound = (np.sin(2 * np.pi * 200 * t) + 
                          0.5 * np.random.randn(snare_samples)) * np.exp(-t * 10)
            audio[start_sample:end_sample] += snare_sound
    
    # Add hi-hat (high frequency)
    hihat_times = [0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875]
    for hihat_time in hihat_times:
        if hihat_time < duration:
            start_sample = int(hihat_time * sample_rate)
            end_sample = min(start_sample + 200, samples)
            
            # Create hi-hat sound (high frequency)
            hihat_samples = end_sample - start_sample
            t = np.linspace(0, hihat_samples / sample_rate, hihat_samples)
            hihat_sound = (np.sin(2 * np.pi * 8000 * t) + 
                          np.sin(2 * np.pi * 12000 * t)) * np.exp(-t * 50)
            audio[start_sample:end_sample] += hihat_sound * 0.3
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32)