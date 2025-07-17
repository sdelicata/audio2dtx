import os
import zipfile
import numpy as np
import librosa
import tensorflow as tf
from pydub import AudioSegment
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import soundfile as sf
import math
from pathlib import Path
from shutil import copytree, rmtree, copy2
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import requests
import json
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureExtractor:
    """Advanced feature extractor for drum sound classification"""
    
    def __init__(self, sr=44100, n_mfcc=13, n_chroma=12, n_contrast=7):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.n_contrast = n_contrast
        self.scaler = StandardScaler()
        self.feature_cache = {}
        
    def extract_mfcc(self, audio_window):
        """Extract MFCC features (13 coefficients)"""
        try:
            mfcc = librosa.feature.mfcc(
                y=audio_window, 
                sr=self.sr, 
                n_mfcc=self.n_mfcc,
                hop_length=512,
                n_fft=2048
            )
            return np.mean(mfcc, axis=1)
        except Exception:
            return np.zeros(self.n_mfcc)
    
    def extract_spectral_contrast(self, audio_window):
        """Extract spectral contrast features (7 features)"""
        try:
            contrast = librosa.feature.spectral_contrast(
                y=audio_window,
                sr=self.sr,
                n_bands=self.n_contrast-1,
                hop_length=512,
                n_fft=2048
            )
            return np.mean(contrast, axis=1)
        except Exception:
            return np.zeros(self.n_contrast)
    
    def extract_chroma(self, audio_window):
        """Extract chroma features (12 features)"""
        try:
            chroma = librosa.feature.chroma_stft(
                y=audio_window,
                sr=self.sr,
                hop_length=512,
                n_fft=2048
            )
            return np.mean(chroma, axis=1)
        except Exception:
            return np.zeros(self.n_chroma)
    
    def extract_tonnetz(self, audio_window):
        """Extract tonnetz features (6 features)"""
        try:
            tonnetz = librosa.feature.tonnetz(
                y=audio_window,
                sr=self.sr,
                hop_length=512
            )
            return np.mean(tonnetz, axis=1)
        except Exception:
            return np.zeros(6)
    
    def extract_spectral_stats(self, audio_window):
        """Extract spectral statistics (4 features)"""
        try:
            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(
                y=audio_window, sr=self.sr, hop_length=512
            )
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(
                y=audio_window, sr=self.sr, hop_length=512
            )
            
            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(
                y=audio_window, hop_length=512
            )
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_window, sr=self.sr, hop_length=512
            )
            
            return np.array([
                np.mean(centroid),
                np.mean(rolloff),
                np.mean(flatness),
                np.mean(bandwidth)
            ])
        except Exception:
            return np.zeros(4)
    
    def extract_temporal_features(self, audio_window):
        """Extract temporal features (5 features)"""
        try:
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                audio_window, hop_length=512
            )
            
            # RMS energy
            rms = librosa.feature.rms(
                y=audio_window, hop_length=512
            )
            
            # Attack time (time to peak)
            attack_time = self._calculate_attack_time(audio_window)
            
            # Decay time (time from peak to sustain)
            decay_time = self._calculate_decay_time(audio_window)
            
            # Spectral flux
            flux = self._calculate_spectral_flux(audio_window)
            
            return np.array([
                np.mean(zcr),
                np.mean(rms),
                attack_time,
                decay_time,
                flux
            ])
        except Exception:
            return np.zeros(5)
    
    def _calculate_attack_time(self, audio_window):
        """Calculate attack time to peak"""
        try:
            envelope = np.abs(audio_window)
            peak_idx = np.argmax(envelope)
            return peak_idx / self.sr
        except Exception:
            return 0.0
    
    def _calculate_decay_time(self, audio_window):
        """Calculate decay time from peak"""
        try:
            envelope = np.abs(audio_window)
            peak_idx = np.argmax(envelope)
            if peak_idx < len(envelope) - 1:
                post_peak = envelope[peak_idx:]
                # Find where amplitude drops to 10% of peak
                threshold = envelope[peak_idx] * 0.1
                decay_idx = np.where(post_peak < threshold)[0]
                if len(decay_idx) > 0:
                    return decay_idx[0] / self.sr
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_spectral_flux(self, audio_window):
        """Calculate spectral flux"""
        try:
            stft = librosa.stft(audio_window, hop_length=512, n_fft=2048)
            magnitude = np.abs(stft)
            flux = np.sum(np.diff(magnitude, axis=1) ** 2)
            return flux
        except Exception:
            return 0.0
    
    def extract_comprehensive_features(self, audio_window):
        """Extract all features and combine them"""
        try:
            # Extract all feature types
            mfcc = self.extract_mfcc(audio_window)
            spectral_contrast = self.extract_spectral_contrast(audio_window)
            chroma = self.extract_chroma(audio_window)
            tonnetz = self.extract_tonnetz(audio_window)
            spectral_stats = self.extract_spectral_stats(audio_window)
            temporal = self.extract_temporal_features(audio_window)
            
            # Combine all features (47 features total)
            combined_features = np.concatenate([
                mfcc,              # 13 features
                spectral_contrast, # 7 features
                chroma,            # 12 features
                tonnetz,           # 6 features
                spectral_stats,    # 4 features
                temporal           # 5 features
            ])
            
            # Store individual feature types for analysis
            features_dict = {
                'mfcc': mfcc,
                'spectral_contrast': spectral_contrast,
                'chroma': chroma,
                'tonnetz': tonnetz,
                'spectral_stats': spectral_stats,
                'temporal': temporal
            }
            
            return combined_features, features_dict
        except Exception:
            return np.zeros(47), {}


class MagentaDrumClassifier:
    """Magenta OaF Drums model integration for drum onset classification"""
    
    def __init__(self):
        self.model = None
        self.config = None
        self.confidence_threshold = 0.7
        self.sample_rate = 44100
        # Magenta service configuration with environment detection
        self.magenta_service_url = self._detect_magenta_service_url()
        self.service_available = False
        self.health_timeout = 2.0  # seconds - quick health checks
        self.classify_timeout = 300.0  # seconds - 5 minutes for classification
        self.drum_classes = {
            0: 'kick',
            1: 'snare',
            2: 'hi-hat-close',
            3: 'hi-hat-open',
            4: 'tom-high',
            5: 'tom-low',
            6: 'tom-floor',
            7: 'crash',
            8: 'ride',
            9: 'ride-bell'
        }
        self.class_to_instrument = {
            'kick': 2,
            'snare': 1,
            'hi-hat-close': 0,
            'hi-hat-open': 7,
            'tom-high': 3,
            'tom-low': 4,
            'tom-floor': 6,
            'crash': 9,
            'ride': 5,
            'ride-bell': 8
        }
        
    def _detect_magenta_service_url(self):
        """Detect Magenta service URL based on environment"""
        # Priority 1: Explicit environment variable
        if os.getenv('MAGENTA_SERVICE_URL'):
            return os.getenv('MAGENTA_SERVICE_URL')
        
        # Priority 2: Docker environment detection
        if os.path.exists('/.dockerenv'):
            # Inside Docker container - use service name
            return 'http://magenta-service:5000'
        
        # Priority 3: Local development
        return 'http://localhost:5000'
        
    def load_model(self):
        """Check Magenta service availability with improved detection"""
        try:
            # Check if Magenta service is available
            health_url = f"{self.magenta_service_url}/health"
            print(f"Checking Magenta service at {health_url}...")
            
            response = requests.get(health_url, timeout=self.health_timeout)
            
            if response.status_code == 200:
                health_data = response.json()
                self.service_available = health_data.get('status') == 'healthy'
                magenta_loaded = health_data.get('magenta_loaded', False)
                
                if self.service_available:
                    print(f"✓ Magenta service available (Model loaded: {magenta_loaded})")
                    self.model = True  # Mark as available
                    return True
                else:
                    print("⚠ Magenta service unhealthy, using enhanced simulation")
                    self.model = None
                    return False
            else:
                print(f"⚠ Magenta service returned status {response.status_code}, using enhanced simulation")
                self.service_available = False
                self.model = None
                return False
                
        except requests.exceptions.ConnectionError:
            print("ℹ Magenta service not available - using enhanced simulation mode")
            self.service_available = False
            self.model = None
            return False
        except requests.exceptions.Timeout:
            print("⚠ Magenta service timeout - using enhanced simulation mode")
            self.service_available = False
            self.model = None
            return False
        except requests.exceptions.RequestException as e:
            print(f"⚠ Magenta service error: {e} - using enhanced simulation mode")
            self.service_available = False
            self.model = None
            return False
        except Exception as e:
            print(f"⚠ Unexpected error checking Magenta service: {e}")
            print("Falling back to enhanced simulation mode")
            self.service_available = False
            self.model = None
            return False
    
    def _prepare_for_magenta(self, onset_audio):
        """Preprocessing audio for Magenta OaF"""
        import librosa
        
        # Normalize audio for Magenta
        if np.max(np.abs(onset_audio)) > 0:
            normalized = onset_audio / np.max(np.abs(onset_audio))
        else:
            normalized = onset_audio
        
        # Create mel-spectrogram (format expected by E-GMD)
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=normalized, 
                sr=self.sample_rate,
                n_mels=128,
                hop_length=256,
                n_fft=1024
            )
            return librosa.power_to_db(mel_spec)
        except Exception as e:
            print(f"Error in Magenta preprocessing: {e}")
            # Return a default spectrogram if preprocessing fails
            return np.zeros((128, 64))  # Default shape for mel-spectrogram
    
    def _enhanced_magenta_prediction(self, audio_window, spectrogram):
        """Enhanced prediction using mel-spectrogram analysis"""
        try:
            # Analyze mel-spectrogram features for better classification
            # This is an enhanced version that uses spectral data
            
            # Extract key features from mel-spectrogram
            if spectrogram.size == 0:
                return self._simulate_magenta_prediction(audio_window)
            
            # Calculate spectral statistics from mel-spectrogram
            spectral_mean = np.mean(spectrogram, axis=1)  # Mean across time
            spectral_energy = np.sum(spectrogram ** 2)
            low_freq_energy = np.sum(spectral_mean[:32])   # Lower mel bands
            mid_freq_energy = np.sum(spectral_mean[32:96]) # Mid mel bands  
            high_freq_energy = np.sum(spectral_mean[96:])  # Upper mel bands
            
            total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
            if total_energy == 0:
                return self._simulate_magenta_prediction(audio_window)
            
            # Calculate energy ratios
            low_ratio = low_freq_energy / total_energy
            mid_ratio = mid_freq_energy / total_energy
            high_ratio = high_freq_energy / total_energy
            
            # Enhanced classification using mel-spectrogram features
            rms = np.sqrt(np.mean(audio_window**2)) if len(audio_window) > 0 else 0
            
            # Kick drum: Strong low frequency energy in mel-spectrogram
            if low_ratio > 0.6 and spectral_energy > np.percentile(spectral_mean, 75):
                return {
                    'instrument': 'kick', 
                    'confidence': min(0.9, 0.7 + low_ratio * 0.3), 
                    'velocity': min(rms * 2, 1.0)
                }
            
            # Snare: Strong mid-frequency with balanced spread
            elif mid_ratio > 0.4 and high_ratio > 0.2:
                return {
                    'instrument': 'snare', 
                    'confidence': min(0.85, 0.65 + mid_ratio * 0.3), 
                    'velocity': min(rms * 1.5, 1.0)
                }
            
            # Hi-hat close: High frequency dominance, short duration
            elif high_ratio > 0.5 and np.max(spectral_mean[96:]) > np.mean(spectral_mean):
                return {
                    'instrument': 'hi-hat-close', 
                    'confidence': min(0.8, 0.6 + high_ratio * 0.25), 
                    'velocity': min(rms * 3, 1.0)
                }
            
            # Hi-hat open: High frequency with more spread
            elif high_ratio > 0.35 and mid_ratio > 0.2:
                return {
                    'instrument': 'hi-hat-open', 
                    'confidence': min(0.75, 0.55 + high_ratio * 0.3), 
                    'velocity': min(rms * 2.5, 1.0)
                }
            
            # Crash: Very high frequency spread across spectrum
            elif high_ratio > 0.3 and np.std(spectral_mean) > np.mean(spectral_mean) * 0.5:
                return {
                    'instrument': 'crash', 
                    'confidence': min(0.75, 0.5 + high_ratio * 0.4), 
                    'velocity': min(rms * 1.5, 1.0)
                }
            
            # Toms: Mid-low frequency content
            elif low_ratio > 0.3 and mid_ratio > 0.25:
                if np.argmax(spectral_mean[:64]) < 20:
                    return {
                        'instrument': 'tom-low', 
                        'confidence': 0.65, 
                        'velocity': min(rms * 2, 1.0)
                    }
                else:
                    return {
                        'instrument': 'tom-high', 
                        'confidence': 0.65, 
                        'velocity': min(rms * 2, 1.0)
                    }
            
            # Ride: Mid-high frequency with sustained energy
            elif mid_ratio > 0.3 and high_ratio > 0.25:
                return {
                    'instrument': 'ride', 
                    'confidence': 0.6, 
                    'velocity': min(rms * 1.5, 1.0)
                }
            
            # Fallback to original simulation
            else:
                return self._simulate_magenta_prediction(audio_window)
                
        except Exception as e:
            print(f"Error in enhanced Magenta prediction: {e}")
            return self._simulate_magenta_prediction(audio_window)
            
    def classify_onset(self, audio_window, onset_time):
        """Classify a single onset using Magenta service"""
        if not self.service_available or self.model is None:
            return self._fallback_classification(audio_window, onset_time)
            
        try:
            # Use Magenta service for real classification
            prediction = self._call_magenta_service(audio_window)
            
            if prediction and prediction['confidence'] > self.confidence_threshold:
                return {
                    'instrument': prediction['instrument'],
                    'confidence': prediction['confidence'],
                    'velocity': prediction['velocity'],
                    'class_id': self.class_to_instrument.get(prediction['instrument'], 2)
                }
            else:
                return self._fallback_classification(audio_window, onset_time)
                
        except Exception as e:
            print(f"Error in Magenta service classification: {e}")
            return self._fallback_classification(audio_window, onset_time)
    
    def _call_magenta_service(self, audio_window):
        """Make HTTP call to Magenta service for drum classification"""
        try:
            classify_url = f"{self.magenta_service_url}/classify-drums"
            
            # Prepare payload
            payload = {
                'audio_window': audio_window.tolist()  # Convert numpy array to list
            }
            
            # Make HTTP request
            response = requests.post(
                classify_url, 
                json=payload, 
                timeout=self.classify_timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success', False):
                    prediction = result.get('prediction', {})
                    return {
                        'instrument': prediction.get('instrument', 'snare'),
                        'confidence': float(prediction.get('confidence', 0.5)),
                        'velocity': float(prediction.get('velocity', 0.5))
                    }
                else:
                    print(f"Magenta service returned error: {result.get('error', 'Unknown error')}")
                    return None
            else:
                print(f"Magenta service HTTP error: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print("Magenta service request timeout")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Magenta service request failed: {e}")
            return None
        except Exception as e:
            print(f"Error calling Magenta service: {e}")
            return None
    
    def _simulate_magenta_prediction(self, audio_window):
        """Simulate Magenta model prediction (placeholder)"""
        # This is a placeholder that simulates what Magenta would do
        # In a real implementation, this would use the actual model
        
        # Analyze audio characteristics
        if len(audio_window) < 100:
            return {'instrument': 'kick', 'confidence': 0.3, 'velocity': 0.5}
        
        # Simple heuristic based on audio characteristics
        rms = np.sqrt(np.mean(audio_window**2))
        zcr = np.mean(np.abs(np.diff(np.sign(audio_window))))
        
        # Frequency analysis
        fft = np.fft.fft(audio_window)
        freqs = np.fft.fftfreq(len(audio_window), 1/44100)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = freqs[:len(freqs)//2]
        
        # Calculate frequency bands
        low_energy = np.sum(magnitude[(freqs >= 20) & (freqs <= 200)])
        mid_energy = np.sum(magnitude[(freqs >= 200) & (freqs <= 2000)])
        high_energy = np.sum(magnitude[(freqs >= 2000) & (freqs <= 8000)])
        
        total_energy = low_energy + mid_energy + high_energy
        if total_energy == 0:
            return {'instrument': 'kick', 'confidence': 0.3, 'velocity': 0.5}
        
        low_ratio = low_energy / total_energy
        mid_ratio = mid_energy / total_energy
        high_ratio = high_energy / total_energy
        
        # Classification logic
        if low_ratio > 0.6 and rms > 0.1:
            return {'instrument': 'kick', 'confidence': 0.85, 'velocity': min(rms * 2, 1.0)}
        elif mid_ratio > 0.4 and high_ratio > 0.3:
            return {'instrument': 'snare', 'confidence': 0.80, 'velocity': min(rms * 1.5, 1.0)}
        elif high_ratio > 0.5 and zcr > 0.05:
            if rms < 0.05:
                return {'instrument': 'hi-hat-close', 'confidence': 0.75, 'velocity': min(rms * 3, 1.0)}
            else:
                return {'instrument': 'hi-hat-open', 'confidence': 0.70, 'velocity': min(rms * 2, 1.0)}
        elif mid_ratio > 0.3 and low_ratio > 0.2:
            if np.mean(magnitude) < 0.1:
                return {'instrument': 'tom-high', 'confidence': 0.65, 'velocity': min(rms * 2, 1.0)}
            else:
                return {'instrument': 'tom-low', 'confidence': 0.60, 'velocity': min(rms * 2, 1.0)}
        elif high_ratio > 0.4:
            return {'instrument': 'crash', 'confidence': 0.70, 'velocity': min(rms * 1.5, 1.0)}
        else:
            return {'instrument': 'ride', 'confidence': 0.55, 'velocity': min(rms * 1.5, 1.0)}
    
    def _fallback_classification(self, audio_window, onset_time):
        """Fallback classification when Magenta is not available"""
        # Simple fallback based on energy distribution
        if len(audio_window) < 100:
            return {'instrument': 'kick', 'confidence': 0.3, 'velocity': 0.5, 'class_id': 2}
        
        rms = np.sqrt(np.mean(audio_window**2))
        
        # Simple energy-based classification
        if rms > 0.1:
            return {'instrument': 'kick', 'confidence': 0.6, 'velocity': min(rms * 2, 1.0), 'class_id': 2}
        elif rms > 0.05:
            return {'instrument': 'snare', 'confidence': 0.5, 'velocity': min(rms * 1.5, 1.0), 'class_id': 1}
        else:
            return {'instrument': 'hi-hat-close', 'confidence': 0.4, 'velocity': min(rms * 3, 1.0), 'class_id': 0}


class HybridDrumClassifier:
    """Hybrid classifier combining multiple approaches with weighted voting"""
    
    def __init__(self):
        self.feature_extractor = AdvancedFeatureExtractor()
        self.magenta_classifier = MagentaDrumClassifier()
        self.weights = {
            'magenta': 0.4,
            'features': 0.3,
            'context': 0.3
        }
        self.confidence_threshold = 0.6
        self.sample_rate = 44100
        self.context_history = []
        self.max_history = 10
        
    def initialize(self):
        """Initialize the hybrid classifier"""
        print("Initializing hybrid classifier...")
        self.magenta_classifier.load_model()
        print("Hybrid classifier initialized")
        
    def classify_onset(self, onset_time, drum_audio, context_window=0.1):
        """Classify a single onset using hybrid approach"""
        try:
            # Extract audio window around onset
            window = self._extract_audio_window(drum_audio, onset_time, context_window)
            
            if len(window) < 100:
                return self._get_default_result()
            
            # Approach 1: Magenta OaF Drums
            magenta_result = self.magenta_classifier.classify_onset(window, onset_time)
            
            # Approach 2: Advanced features + ML
            features_result = self._classify_with_features(window, onset_time)
            
            # Approach 3: Contextual analysis
            context_result = self._analyze_context(onset_time, magenta_result, features_result)
            
            # Weighted voting
            final_result = self._weighted_vote(magenta_result, features_result, context_result)
            
            # Update context history
            self._update_context_history(onset_time, final_result)
            
            return final_result
            
        except Exception as e:
            print(f"Error in hybrid classification: {e}")
            return self._get_default_result()
    
    def _extract_audio_window(self, drum_audio, onset_time, context_window):
        """Extract audio window around onset"""
        try:
            # Convert to mono if stereo
            if len(drum_audio.shape) > 1:
                drum_mono = librosa.to_mono(np.transpose(drum_audio))
            else:
                drum_mono = drum_audio
            
            # Calculate sample indices
            start_sample = int((onset_time - context_window/2) * self.sample_rate)
            end_sample = int((onset_time + context_window/2) * self.sample_rate)
            
            # Ensure bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(drum_mono), end_sample)
            
            return drum_mono[start_sample:end_sample]
            
        except Exception:
            return np.zeros(1000)
    
    def _classify_with_features(self, audio_window, onset_time):
        """Classify using advanced features"""
        try:
            # Extract comprehensive features
            features, features_dict = self.feature_extractor.extract_comprehensive_features(audio_window)
            
            # Rule-based classification using advanced features
            mfcc = features_dict.get('mfcc', np.zeros(13))
            spectral_contrast = features_dict.get('spectral_contrast', np.zeros(7))
            chroma = features_dict.get('chroma', np.zeros(12))
            tonnetz = features_dict.get('tonnetz', np.zeros(6))
            spectral_stats = features_dict.get('spectral_stats', np.zeros(4))
            temporal = features_dict.get('temporal', np.zeros(5))
            
            # Extract key features for classification
            spectral_centroid = spectral_stats[0] if len(spectral_stats) > 0 else 0
            spectral_rolloff = spectral_stats[1] if len(spectral_stats) > 1 else 0
            spectral_flatness = spectral_stats[2] if len(spectral_stats) > 2 else 0
            attack_time = temporal[2] if len(temporal) > 2 else 0
            rms = temporal[1] if len(temporal) > 1 else 0
            zcr = temporal[0] if len(temporal) > 0 else 0
            
            # Advanced classification rules
            if attack_time < 0.01 and spectral_centroid < 200 and rms > 0.1:
                return {'instrument': 'kick', 'confidence': 0.85, 'velocity': min(rms * 2, 1.0), 'class_id': 2}
            elif spectral_contrast[2] > 0.5 and spectral_centroid > 500 and spectral_centroid < 4000:
                return {'instrument': 'snare', 'confidence': 0.80, 'velocity': min(rms * 1.5, 1.0), 'class_id': 1}
            elif spectral_flatness > 0.5 and zcr > 0.1 and spectral_centroid > 6000:
                if rms < 0.05:
                    return {'instrument': 'hi-hat-close', 'confidence': 0.75, 'velocity': min(rms * 3, 1.0), 'class_id': 0}
                else:
                    return {'instrument': 'hi-hat-open', 'confidence': 0.70, 'velocity': min(rms * 2, 1.0), 'class_id': 7}
            elif spectral_centroid > 200 and spectral_centroid < 1000 and attack_time > 0.01:
                if spectral_centroid < 400:
                    return {'instrument': 'tom-low', 'confidence': 0.65, 'velocity': min(rms * 2, 1.0), 'class_id': 4}
                elif spectral_centroid < 600:
                    return {'instrument': 'tom-high', 'confidence': 0.60, 'velocity': min(rms * 2, 1.0), 'class_id': 3}
                else:
                    return {'instrument': 'tom-floor', 'confidence': 0.55, 'velocity': min(rms * 2, 1.0), 'class_id': 6}
            elif spectral_centroid > 8000 and spectral_rolloff > 10000:
                return {'instrument': 'crash', 'confidence': 0.70, 'velocity': min(rms * 1.5, 1.0), 'class_id': 9}
            elif spectral_centroid > 4000 and spectral_centroid < 8000:
                return {'instrument': 'ride', 'confidence': 0.60, 'velocity': min(rms * 1.5, 1.0), 'class_id': 5}
            else:
                # Default based on spectral centroid
                if spectral_centroid < 200:
                    return {'instrument': 'kick', 'confidence': 0.50, 'velocity': min(rms * 2, 1.0), 'class_id': 2}
                elif spectral_centroid < 1000:
                    return {'instrument': 'snare', 'confidence': 0.45, 'velocity': min(rms * 1.5, 1.0), 'class_id': 1}
                else:
                    return {'instrument': 'hi-hat-close', 'confidence': 0.40, 'velocity': min(rms * 3, 1.0), 'class_id': 0}
                    
        except Exception as e:
            print(f"Error in features classification: {e}")
            return self._get_default_result()
    
    def _analyze_context(self, onset_time, magenta_result, features_result):
        """Analyze temporal context and patterns"""
        try:
            # Context analysis based on recent history
            if len(self.context_history) == 0:
                return magenta_result
            
            # Get recent onsets (last 2 seconds)
            recent_onsets = [h for h in self.context_history if onset_time - h['time'] < 2.0]
            
            if len(recent_onsets) == 0:
                return magenta_result
            
            # Analyze patterns
            recent_instruments = [h['instrument'] for h in recent_onsets]
            
            # Simple pattern analysis
            if len(recent_instruments) >= 2:
                # Check for alternating patterns (kick-snare)
                if recent_instruments[-1] == 'kick' and recent_instruments[-2] == 'snare':
                    if magenta_result['instrument'] in ['kick', 'snare']:
                        return {
                            'instrument': 'snare',
                            'confidence': 0.75,
                            'velocity': magenta_result['velocity'],
                            'class_id': 1
                        }
                elif recent_instruments[-1] == 'snare' and recent_instruments[-2] == 'kick':
                    if magenta_result['instrument'] in ['kick', 'snare']:
                        return {
                            'instrument': 'kick',
                            'confidence': 0.75,
                            'velocity': magenta_result['velocity'],
                            'class_id': 2
                        }
            
            # Boost confidence if same instrument detected recently
            if magenta_result['instrument'] in recent_instruments[-3:]:
                boosted_confidence = min(magenta_result['confidence'] * 1.2, 0.95)
                return {
                    'instrument': magenta_result['instrument'],
                    'confidence': boosted_confidence,
                    'velocity': magenta_result['velocity'],
                    'class_id': magenta_result.get('class_id', 2)
                }
            
            return magenta_result
            
        except Exception:
            return magenta_result
    
    def _weighted_vote(self, magenta_result, features_result, context_result):
        """Combine results using weighted voting"""
        try:
            # Create vote dictionary
            votes = {}
            
            # Magenta vote
            magenta_instrument = magenta_result['instrument']
            magenta_confidence = magenta_result['confidence'] * self.weights['magenta']
            votes[magenta_instrument] = votes.get(magenta_instrument, 0) + magenta_confidence
            
            # Features vote
            features_instrument = features_result['instrument']
            features_confidence = features_result['confidence'] * self.weights['features']
            votes[features_instrument] = votes.get(features_instrument, 0) + features_confidence
            
            # Context vote
            context_instrument = context_result['instrument']
            context_confidence = context_result['confidence'] * self.weights['context']
            votes[context_instrument] = votes.get(context_instrument, 0) + context_confidence
            
            # Find winning instrument
            winning_instrument = max(votes, key=votes.get)
            winning_confidence = votes[winning_instrument]
            
            # Get velocity from the most confident individual result
            individual_results = [magenta_result, features_result, context_result]
            best_individual = max(individual_results, key=lambda x: x['confidence'])
            winning_velocity = best_individual['velocity']
            
            # Map to class ID
            class_mapping = {
                'kick': 2, 'snare': 1, 'hi-hat-close': 0, 'hi-hat-open': 7,
                'tom-high': 3, 'tom-low': 4, 'tom-floor': 6,
                'crash': 9, 'ride': 5, 'ride-bell': 8
            }
            
            return {
                'instrument': winning_instrument,
                'confidence': winning_confidence,
                'velocity': winning_velocity,
                'class_id': class_mapping.get(winning_instrument, 2)
            }
            
        except Exception:
            return magenta_result
    
    def _update_context_history(self, onset_time, result):
        """Update context history with new result"""
        try:
            self.context_history.append({
                'time': onset_time,
                'instrument': result['instrument'],
                'confidence': result['confidence']
            })
            
            # Keep only recent history
            if len(self.context_history) > self.max_history:
                self.context_history.pop(0)
                
        except Exception:
            pass
    
    def _get_default_result(self):
        """Get default classification result"""
        return {
            'instrument': 'kick',
            'confidence': 0.5,
            'velocity': 0.7,
            'class_id': 2
        }


class AudioToChart:
    def __init__(self, input_audio_path, metadata=None, use_magenta_only=False):
        self.input_audio_path = input_audio_path
        self.original_filename = os.path.basename(input_audio_path)
        self.use_magenta_only = use_magenta_only  # Track 3 parameter
        
        # Initialize metadata with defaults or provided values
        if metadata is None:
            metadata = {
                'title': os.path.splitext(os.path.basename(input_audio_path))[0],
                'artist': 'Unknown Artist',
                'author': 'Audio2DTX',
                'difficulty': 5,
                'use_original_bgm': True,
                'time_signature': '4/4',
                'genre': 'Electronic',
                'comment': 'Auto-generated by Audio2DTX'
            }
        
        # Store metadata
        self.metadata = metadata
        self.song_name = metadata['title']
        self.use_original_bgm = metadata['use_original_bgm']
        
        self.setup_channel_mappings()
        self.model = None
        self.drum_audio = None
        self.onset_predictions = None
        self.onset_results = None
        self.hybrid_classifier = HybridDrumClassifier()
        
        # Log the processing mode
        if self.use_magenta_only:
            print("🔮 Track 3: Using MAGENTA-ONLY classification (simplified approach)")
        
    def setup_channel_mappings(self):
        """Setup drum channel mappings for DTXMania format"""
        # Map pitch to DTX channels
        self.PITCH_TO_CHANNEL = {
            22: '11', 42: '11', 44: '11',  # Hi-hat close
            38: '12', 40: '12',            # Snare
            36: '13',                      # Bass Drum
            48: '14', 50: '14',            # High Tom
            45: '15', 47: '15',            # Low Tom
            51: '16', 59: '16',            # Ride Cymbal
            43: '17',                      # Floor Tom
            26: '18', 46: '18',            # Hi-hat open
            53: '19',                      # Ride Cymbal (bell)
            49: '1A', 52: '1A', 55: '1A', 57: '1A',  # Left Cymbal
        }
        
        # Define channel names and classes
        self.CHANNEL_NAMES = {
            '11': 'Hi-hat Close',
            '12': 'Snare',
            '13': 'Bass Drum',
            '14': 'High Tom',
            '15': 'Low Tom',
            '16': 'Ride Cymbal',
            '17': 'Floor Tom',
            '18': 'Hi-hat Open',
            '19': 'Ride Cymbal (Bell)',
            '1A': 'Left Cymbal (Crash)',
        }
        
        # Map channels to integer classes for ML model
        self.CHANNEL_TO_INT = dict((channel, no) for no, channel in enumerate(self.CHANNEL_NAMES.keys()))
        self.INT_TO_CHANNEL = dict((no, channel) for no, channel in enumerate(self.CHANNEL_NAMES.keys()))
        self.num_class = len(self.CHANNEL_NAMES)
        
    def separate_audio_tracks(self):
        """Separate audio into stems using Spleeter"""
        print("Separating audio tracks...")
        separator = Separator('spleeter:4stems')
        audio_loader = AudioAdapter.default()
        
        # Load audio file
        waveform, _ = audio_loader.load(self.input_audio_path, sample_rate=44100)
        
        # Store original audio for tempo analysis
        self.original_audio = librosa.to_mono(np.transpose(waveform))
        self.sample_rate = 44100
        
        # Perform separation
        prediction = separator.separate(waveform)
        
        # Extract drums
        self.drum_audio = prediction['drums']
        
        # Create BGM without drums (vocals + bass + other)
        sound1 = librosa.to_mono(np.transpose(prediction['vocals']))
        sound2 = librosa.to_mono(np.transpose(prediction['bass']))
        sound3 = librosa.to_mono(np.transpose(prediction['other']))
        
        self.bgm_audio = (sound1 + sound2 + sound3) / 3
        
        print("Audio separation completed")
        
    def detect_tempo_and_beats(self):
        """Detect tempo and beat positions from audio"""
        print("Analyzing tempo and beats...")
        
        # Detect tempo and beat frames
        tempo, beats = librosa.beat.beat_track(y=self.original_audio, sr=self.sample_rate)
        
        # Convert beat frames to time positions
        beat_times = librosa.frames_to_time(beats, sr=self.sample_rate)
        
        # Calculate dynamic tempo curve
        tempo_curve = librosa.beat.tempo(y=self.original_audio, sr=self.sample_rate, hop_length=512)
        
        # Store tempo information
        self.tempo_bpm = float(tempo)
        self.beat_times = beat_times
        self.tempo_curve = tempo_curve
        
        print(f"Detected tempo: {self.tempo_bpm:.1f} BPM")
        print(f"Found {len(beat_times)} beats")
        
        return tempo, beat_times, tempo_curve
        
    def quantize_onsets_to_beats(self, onset_times, beat_times):
        """Quantize onset times to the nearest beat positions"""
        print("Quantizing onsets to beats...")
        
        quantized_onsets = []
        
        for onset_time in onset_times:
            # Find closest beat
            closest_beat_idx = np.argmin(np.abs(beat_times - onset_time))
            closest_beat_time = beat_times[closest_beat_idx]
            
            # Only quantize if within reasonable distance (quarter note)
            quarter_note_duration = 60.0 / self.tempo_bpm
            if abs(onset_time - closest_beat_time) < quarter_note_duration * 0.5:
                quantized_onsets.append(closest_beat_time)
            else:
                # Keep original timing for off-beat notes
                quantized_onsets.append(onset_time)
        
        print(f"Quantized {len(quantized_onsets)} onsets to beats")
        return np.array(quantized_onsets)
        
    def calculate_adaptive_bar_timing(self, beat_times):
        """Calculate bar positions based on detected beats"""
        # Assume 4/4 time signature
        beats_per_bar = 4
        
        bar_positions = []
        bar_times = []
        
        for i in range(0, len(beat_times), beats_per_bar):
            if i < len(beat_times):
                bar_positions.append(i // beats_per_bar)
                bar_times.append(beat_times[i])
                
        return np.array(bar_positions), np.array(bar_times)
        
    def to_spectrogram(self, mono_audio, time_per_hop, sr=44100):
        """Convert mono audio to mel-spectrogram"""
        s = librosa.feature.melspectrogram(y=mono_audio, sr=sr, hop_length=int(time_per_hop * sr))
        s_db = librosa.power_to_db(s, ref=np.max)
        return s_db
        
    def create_dataset(self, spec_list, time_per_hop, sr=44100, hop_per_window=1, stride=1):
        """Create dataset for ML model from spectrograms"""
        x_data = []
        
        if len(np.shape(spec_list)) == 2:
            spec_list = [spec_list]
            
        for spec in spec_list:
            spec = np.transpose(spec)
            
            # Create windows for training data
            for hop in range(0, len(spec) - hop_per_window, stride):
                window_start = hop
                window_end = hop + hop_per_window
                current_window = spec[window_start:window_end]
                
                # Pad with zeros if needed
                if len(current_window) != hop_per_window:
                    current_window = np.pad(current_window, ((0, hop_per_window - len(current_window)), (0, 0)), 'constant')
                    
                x_data.append(np.transpose(current_window))
                
        return np.array(x_data)
        
    def preprocess_drums(self):
        """Preprocess drum audio for ML model"""
        print("Preprocessing drums...")
        time_per_hop = 0.01
        hop_per_window = 4
        
        # Convert to mono
        drum_mono = librosa.to_mono(np.transpose(self.drum_audio))
        
        # Convert to spectrogram
        spec = self.to_spectrogram(drum_mono, time_per_hop=time_per_hop)
        
        # Create dataset
        model_input = self.create_dataset(spec, time_per_hop=time_per_hop, 
                                        hop_per_window=hop_per_window, stride=1)
        
        print("Drum preprocessing completed")
        return model_input
        
    def load_model(self):
        """Load the pre-trained onset detection model"""
        print("Loading ML model...")
        self.model = tf.keras.models.load_model("/app/PredictOnset.h5", compile=False)
        print("Model loaded successfully")
        
    def detect_onsets_librosa(self):
        """Enhanced librosa onset detection with optimized parameters"""
        print("Detecting onsets with enhanced librosa methods...")
        
        # Convert drum audio to mono for analysis
        drum_mono = librosa.to_mono(np.transpose(self.drum_audio))
        
        # Different onset detection methods using onset_strength
        onset_methods = [
            {'method': 'energy', 'threshold': 0.05, 'pre_max': 8, 'post_max': 8, 'pre_avg': 8, 'post_avg': 8, 'wait': 4},
            {'method': 'hfc', 'threshold': 0.03, 'pre_max': 12, 'post_max': 12, 'pre_avg': 12, 'post_avg': 12, 'wait': 6},
            {'method': 'complex', 'threshold': 0.02, 'pre_max': 10, 'post_max': 10, 'pre_avg': 10, 'post_avg': 10, 'wait': 5},
            {'method': 'phase', 'threshold': 0.08, 'pre_max': 6, 'post_max': 6, 'pre_avg': 6, 'post_avg': 6, 'wait': 3},
            {'method': 'specdiff', 'threshold': 0.04, 'pre_max': 10, 'post_max': 10, 'pre_avg': 10, 'post_avg': 10, 'wait': 5},
        ]
        
        all_onsets = []
        hop_length = int(0.01 * self.sample_rate)  # 10ms hop
        
        for method_config in onset_methods:
            try:
                method_name = method_config['method']
                # Create onset strength function for this method
                if method_name == 'energy':
                    onset_strength = librosa.onset.onset_strength(
                        y=drum_mono, 
                        sr=self.sample_rate,
                        hop_length=hop_length,
                        aggregate=np.median
                    )
                elif method_name == 'hfc':
                    onset_strength = librosa.onset.onset_strength(
                        y=drum_mono, 
                        sr=self.sample_rate,
                        hop_length=hop_length,
                        feature=librosa.feature.spectral_centroid,
                        aggregate=np.median
                    )
                elif method_name == 'complex':
                    onset_strength = librosa.onset.onset_strength(
                        y=drum_mono, 
                        sr=self.sample_rate,
                        hop_length=hop_length,
                        feature=librosa.feature.spectral_rolloff,
                        aggregate=np.median
                    )
                elif method_name == 'phase':
                    onset_strength = librosa.onset.onset_strength(
                        y=drum_mono, 
                        sr=self.sample_rate,
                        hop_length=hop_length,
                        aggregate=np.median
                    )
                elif method_name == 'specdiff':
                    onset_strength = librosa.onset.onset_strength(
                        y=drum_mono, 
                        sr=self.sample_rate,
                        hop_length=hop_length,
                        feature=librosa.feature.mfcc,
                        aggregate=np.median
                    )
                else:
                    onset_strength = librosa.onset.onset_strength(
                        y=drum_mono, 
                        sr=self.sample_rate,
                        hop_length=hop_length,
                        aggregate=np.median
                    )
                
                # Detect onsets from strength function
                onsets = librosa.onset.onset_detect(
                    onset_envelope=onset_strength,
                    sr=self.sample_rate,
                    hop_length=hop_length,
                    delta=method_config['threshold'],
                    pre_max=method_config['pre_max'],
                    post_max=method_config['post_max'],
                    pre_avg=method_config['pre_avg'],
                    post_avg=method_config['post_avg'],
                    wait=method_config['wait']
                )
                
                onset_times = librosa.frames_to_time(onsets, sr=self.sample_rate, hop_length=hop_length)
                all_onsets.extend(onset_times)
                print(f"  {method_name}: {len(onset_times)} onsets")
            except Exception as e:
                print(f"  {method_name}: failed ({str(e)})")
                continue
        
        # Remove duplicates with small tolerance and sort
        unique_onsets = []
        tolerance = 0.01  # 10ms tolerance
        
        for onset in sorted(all_onsets):
            is_duplicate = False
            for existing_onset in unique_onsets:
                if abs(onset - existing_onset) < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_onsets.append(onset)
        
        print(f"Enhanced librosa detected {len(unique_onsets)} unique onsets")
        return np.array(unique_onsets)
    
    def detect_onsets_energy(self):
        """Enhanced energy-based onset detection with multiple energy features"""
        print("Detecting onsets with enhanced energy analysis...")
        
        drum_mono = librosa.to_mono(np.transpose(self.drum_audio))
        hop_length = int(0.005 * self.sample_rate)  # 5ms hop for higher resolution
        
        # Multiple energy features
        # 1. RMS energy
        rms = librosa.feature.rms(y=drum_mono, hop_length=hop_length, frame_length=hop_length*4)[0]
        
        # 2. Spectral flux
        stft = librosa.stft(drum_mono, hop_length=hop_length, n_fft=1024)
        spectral_flux = np.diff(np.abs(stft), axis=1)
        spectral_flux = np.sum(np.maximum(0, spectral_flux), axis=0)  # Only positive flux
        
        # 3. High-frequency energy (for detecting cymbals/hi-hats)
        high_freq_energy = np.sum(np.abs(stft[512:]), axis=0)  # Upper half of spectrum
        
        # 4. Low-frequency energy (for detecting kicks/toms)
        low_freq_energy = np.sum(np.abs(stft[:256]), axis=0)  # Lower quarter of spectrum
        
        # 5. Mid-frequency energy (for detecting snares)
        mid_freq_energy = np.sum(np.abs(stft[128:512]), axis=0)  # Mid range
        
        # Align all features to the same length
        min_len = min(len(rms), len(spectral_flux), len(high_freq_energy), len(low_freq_energy), len(mid_freq_energy))
        rms = rms[:min_len]
        spectral_flux = spectral_flux[:min_len]
        high_freq_energy = high_freq_energy[:min_len]
        low_freq_energy = low_freq_energy[:min_len]
        mid_freq_energy = mid_freq_energy[:min_len]
        
        # Normalize all features
        rms_norm = rms / (np.max(rms) + 1e-10)
        flux_norm = spectral_flux / (np.max(spectral_flux) + 1e-10)
        high_norm = high_freq_energy / (np.max(high_freq_energy) + 1e-10)
        low_norm = low_freq_energy / (np.max(low_freq_energy) + 1e-10)
        mid_norm = mid_freq_energy / (np.max(mid_freq_energy) + 1e-10)
        
        # Combine features with weights
        combined_energy = (0.3 * rms_norm + 0.4 * flux_norm + 0.1 * high_norm + 
                          0.1 * low_norm + 0.1 * mid_norm)
        
        # Apply smoothing to reduce noise
        from scipy.signal import savgol_filter
        if len(combined_energy) > 21:
            combined_energy = savgol_filter(combined_energy, 21, 3)
        
        # Adaptive thresholding with percentile-based approach
        threshold_percentile = 85  # Use 85th percentile as threshold
        threshold = np.percentile(combined_energy, threshold_percentile)
        
        # Additional dynamic threshold adjustment based on local statistics
        window_size = int(0.5 * self.sample_rate / hop_length)  # 0.5 second window
        dynamic_threshold = np.zeros_like(combined_energy)
        
        for i in range(len(combined_energy)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(combined_energy), i + window_size // 2)
            local_window = combined_energy[start_idx:end_idx]
            dynamic_threshold[i] = np.mean(local_window) + 1.5 * np.std(local_window)
        
        # Use the higher of global and dynamic thresholds
        final_threshold = np.maximum(threshold, dynamic_threshold)
        
        # Peak picking with adaptive parameters
        peaks = []
        for i in range(5, len(combined_energy) - 5):
            if combined_energy[i] > final_threshold[i]:
                # Check if it's a local maximum
                if (combined_energy[i] > combined_energy[i-1] and 
                    combined_energy[i] > combined_energy[i+1] and
                    combined_energy[i] >= np.max(combined_energy[i-5:i+6])):
                    peaks.append(i)
        
        # Remove peaks that are too close together
        filtered_peaks = []
        min_distance = int(0.02 * self.sample_rate / hop_length)  # 20ms minimum distance
        
        for peak in peaks:
            if not filtered_peaks or peak - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(peak)
        
        onset_times = librosa.frames_to_time(filtered_peaks, sr=self.sample_rate, hop_length=hop_length)
        
        print(f"Enhanced energy method detected {len(onset_times)} onsets")
        return onset_times
    
    def predict_onsets(self, model_input):
        """Use ML model to predict drum onsets"""
        print("Predicting onsets with ML model...")
        self.onset_predictions = self.model.predict(model_input)
        print("ML onset prediction completed")
        
    def classify_instrument_by_frequency(self, onset_time, window_size=0.05):
        """Enhanced instrument classification based on spectral analysis"""
        drum_mono = librosa.to_mono(np.transpose(self.drum_audio))
        
        # Extract window around onset with adaptive size
        start_sample = int((onset_time - window_size/2) * self.sample_rate)
        end_sample = int((onset_time + window_size/2) * self.sample_rate)
        
        if start_sample < 0:
            start_sample = 0
        if end_sample >= len(drum_mono):
            end_sample = len(drum_mono) - 1
            
        window = drum_mono[start_sample:end_sample]
        
        if len(window) < 512:
            return 2  # Default to kick
        
        # Apply windowing for better frequency resolution
        windowed = window * np.hanning(len(window))
        
        # Compute FFT with zero-padding for better frequency resolution
        fft_size = max(2048, len(windowed))
        fft = np.fft.fft(windowed, fft_size)
        freqs = np.fft.fftfreq(fft_size, 1/self.sample_rate)
        magnitude = np.abs(fft[:fft_size//2])
        freqs = freqs[:fft_size//2]
        
        # Enhanced frequency band analysis
        sub_bass = np.sum(magnitude[(freqs >= 20) & (freqs <= 60)])         # Sub-bass
        bass = np.sum(magnitude[(freqs >= 60) & (freqs <= 120)])            # Bass
        low_mid = np.sum(magnitude[(freqs >= 120) & (freqs <= 400)])        # Low-mid
        mid = np.sum(magnitude[(freqs >= 400) & (freqs <= 1600)])           # Mid
        high_mid = np.sum(magnitude[(freqs >= 1600) & (freqs <= 6400)])     # High-mid
        high = np.sum(magnitude[(freqs >= 6400) & (freqs <= 12800)])        # High
        ultra_high = np.sum(magnitude[(freqs >= 12800) & (freqs <= 20000)]) # Ultra-high
        
        total_energy = sub_bass + bass + low_mid + mid + high_mid + high + ultra_high
        
        if total_energy == 0:
            return 2  # Default to kick
        
        # Calculate energy ratios
        sub_bass_ratio = sub_bass / total_energy
        bass_ratio = bass / total_energy
        low_mid_ratio = low_mid / total_energy
        mid_ratio = mid / total_energy
        high_mid_ratio = high_mid / total_energy
        high_ratio = high / total_energy
        ultra_high_ratio = ultra_high / total_energy
        
        # Calculate advanced spectral features
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude))
        
        # Calculate spectral rolloff points
        cumsum = np.cumsum(magnitude)
        rolloff_85 = freqs[np.where(cumsum >= 0.85 * cumsum[-1])[0][0]]
        rolloff_95 = freqs[np.where(cumsum >= 0.95 * cumsum[-1])[0][0]]
        
        # Calculate spectral flux (change in magnitude)
        if hasattr(self, '_prev_magnitude'):
            spectral_flux = np.sum(np.maximum(0, magnitude - self._prev_magnitude))
        else:
            spectral_flux = np.sum(magnitude)
        self._prev_magnitude = magnitude
        
        # Calculate zero crossing rate for the time domain signal
        zero_crossings = np.sum(np.diff(np.signbit(window)))
        zcr = zero_crossings / len(window)
        
        # Enhanced classification with multiple features
        
        # Kick (Bass Drum): Strong sub-bass and bass, low centroid
        if (sub_bass_ratio > 0.3 or bass_ratio > 0.4) and spectral_centroid < 150:
            return 2  # Bass Drum
        
        # Snare: Mid-frequency dominant with high harmonics, high spectral flux
        elif (mid_ratio > 0.25 and high_mid_ratio > 0.2) and spectral_centroid > 500 and spectral_centroid < 4000:
            return 1  # Snare
        
        # Hi-hat Close: High frequency, low spread, high ZCR
        elif high_ratio > 0.25 and spectral_centroid > 6000 and spectral_spread < 3000 and zcr > 0.1:
            return 0  # Hi-hat Close
            
        # Hi-hat Open: High frequency, higher spread, very high ZCR
        elif (high_ratio > 0.2 or ultra_high_ratio > 0.15) and spectral_centroid > 8000 and spectral_spread > 3000 and zcr > 0.15:
            return 7  # Hi-hat Open
        
        # Crash Cymbal: Very high frequency, very high spread, extremely high ZCR
        elif ultra_high_ratio > 0.1 and spectral_centroid > 10000 and spectral_spread > 4000 and zcr > 0.2:
            return 9  # Crash Cymbal
            
        # Ride Cymbal: High-mid to high frequency, moderate spread
        elif high_mid_ratio > 0.2 and spectral_centroid > 3000 and spectral_centroid < 8000:
            if rolloff_95 > 12000:
                return 5  # Ride Cymbal
            else:
                return 8  # Ride Bell
        
        # Toms: Low-mid frequency dominant, moderate centroid
        elif low_mid_ratio > 0.3 and spectral_centroid > 200 and spectral_centroid < 1000:
            if spectral_centroid < 400:
                return 6  # Floor Tom
            elif spectral_centroid < 600:
                return 4  # Low Tom
            else:
                return 3  # High Tom
        
        # Fallback classification based on spectral centroid
        elif spectral_centroid < 200:
            return 2  # Bass Drum
        elif spectral_centroid < 1000:
            return 4  # Low Tom (default tom)
        elif spectral_centroid < 4000:
            return 1  # Snare
        elif spectral_centroid < 8000:
            return 5  # Ride Cymbal
        else:
            return 0  # Hi-hat Close
        
        # Final fallback
        return 2  # Bass Drum
        
    def fuse_onset_detections(self):
        """Fuse all onset detection methods"""
        print("Fusing onset detection results...")
        
        # Get detections from all methods
        librosa_onsets = self.detect_onsets_librosa()
        energy_onsets = self.detect_onsets_energy()
        
        # Convert ML predictions to onset times
        ml_onsets = []
        time_per_hop = 0.01
        for class_idx in range(self.num_class):
            for time_idx, has_onset in enumerate(self.onset_results[class_idx]):
                if has_onset == 1:
                    onset_time = time_idx * time_per_hop
                    ml_onsets.append(onset_time)
        
        # Combine all onsets
        all_onsets = list(librosa_onsets) + list(energy_onsets) + ml_onsets
        
        # Remove duplicates with tolerance
        tolerance = 0.02  # 20ms tolerance
        fused_onsets = []
        
        for onset in sorted(all_onsets):
            # Check if this onset is close to any existing onset
            is_duplicate = False
            for existing_onset in fused_onsets:
                if abs(onset - existing_onset) < tolerance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                fused_onsets.append(onset)
        
        print(f"Fused {len(all_onsets)} onsets into {len(fused_onsets)} unique onsets")
        return fused_onsets
        
    def improve_onset_classification(self, fused_onsets):
        """Classify each onset into instrument categories"""
        print("Classifying instruments for each onset...")
        
        classified_onsets = [[] for _ in range(self.num_class)]
        
        for onset_time in fused_onsets:
            instrument_class = self.classify_instrument_by_frequency(onset_time)
            if 0 <= instrument_class < self.num_class:
                classified_onsets[instrument_class].append(onset_time)
        
        print(f"Classified onsets: {[len(class_onsets) for class_onsets in classified_onsets]}")
        return classified_onsets
    
    def magenta_only_classification(self, fused_onsets):
        """Track 3: Magenta-only classification (simplified approach)"""
        print("🔮 Track 3: Magenta-only onset classification...")
        
        # Initialize Magenta classifier
        self.hybrid_classifier.initialize()
        
        # Check if Magenta service is available
        if not self.hybrid_classifier.magenta_classifier.service_available:
            print("❌ Magenta service not available, using simplified fallback")
            return self._simple_frequency_classification(fused_onsets)
        
        classified_onsets = [[] for _ in range(self.num_class)]
        
        for onset_time in fused_onsets:
            try:
                # Extract audio window around onset
                drum_mono = librosa.to_mono(np.transpose(self.drum_audio))
                start_sample = int((onset_time - 0.1) * self.sample_rate)
                end_sample = int((onset_time + 0.1) * self.sample_rate)
                
                # Ensure bounds
                start_sample = max(0, start_sample)
                end_sample = min(len(drum_mono), end_sample)
                
                if end_sample <= start_sample:
                    classified_onsets[2].append(onset_time)  # Default to kick
                    continue
                
                window = drum_mono[start_sample:end_sample]
                
                if len(window) < 100:
                    classified_onsets[2].append(onset_time)  # Default to kick
                    continue
                
                # Use Magenta service for classification
                result = self.hybrid_classifier.magenta_classifier.classify_onset(window, onset_time)
                
                if result is not None and result.get('confidence', 0) > 0.6:
                    # Map Magenta result to our class system
                    instrument = result.get('instrument', 'kick')
                    class_id = self._map_instrument_to_class(instrument)
                    classified_onsets[class_id].append(onset_time)
                else:
                    # Low confidence, use simple frequency analysis
                    class_id = self._simple_frequency_classify(window)
                    classified_onsets[class_id].append(onset_time)
                
            except Exception as e:
                print(f"⚠️  Error classifying onset at {onset_time}: {e}")
                # Default to kick on error
                classified_onsets[2].append(onset_time)
        
        print(f"🔮 Magenta-only classified onsets: {[len(class_onsets) for class_onsets in classified_onsets]}")
        return classified_onsets
    
    def _map_instrument_to_class(self, instrument):
        """Map instrument name to class ID"""
        instrument_map = {
            'kick': 2, 'bass': 2, 'bass_drum': 2,
            'snare': 1, 'snare_drum': 1,
            'hi-hat': 0, 'hihat': 0, 'hi_hat': 0,
            'hi-hat-open': 7, 'hihat_open': 7, 'open_hihat': 7,
            'tom': 3, 'tom_high': 3, 'high_tom': 3,
            'tom_low': 4, 'low_tom': 4,
            'tom_floor': 6, 'floor_tom': 6,
            'ride': 5, 'ride_cymbal': 5,
            'ride_bell': 8, 'bell': 8,
            'crash': 9, 'crash_cymbal': 9
        }
        return instrument_map.get(instrument.lower(), 2)  # Default to kick
    
    def _simple_frequency_classify(self, window):
        """Simple frequency-based classification"""
        try:
            # FFT analysis
            fft = np.fft.rfft(window)
            freqs = np.fft.rfftfreq(len(window), 1/self.sample_rate)
            magnitude = np.abs(fft)
            
            # Find peak frequency
            peak_idx = np.argmax(magnitude)
            peak_freq = freqs[peak_idx]
            
            # Simple classification based on peak frequency
            if peak_freq < 100:
                return 2  # kick
            elif peak_freq < 300:
                return 1  # snare
            elif peak_freq < 800:
                return 3  # tom-high
            elif peak_freq < 1500:
                return 4  # tom-low
            elif peak_freq < 3000:
                return 5  # ride
            elif peak_freq < 6000:
                return 0  # hi-hat-close
            elif peak_freq < 12000:
                return 7  # hi-hat-open
            else:
                return 9  # crash
        except:
            return 2  # Default to kick
    
    def _simple_frequency_classification(self, fused_onsets):
        """Fallback classification when Magenta is not available"""
        print("Using simple frequency classification as fallback")
        
        classified_onsets = [[] for _ in range(self.num_class)]
        
        for onset_time in fused_onsets:
            try:
                # Extract audio window
                drum_mono = librosa.to_mono(np.transpose(self.drum_audio))
                start_sample = int((onset_time - 0.05) * self.sample_rate)
                end_sample = int((onset_time + 0.05) * self.sample_rate)
                
                # Ensure bounds
                start_sample = max(0, start_sample)
                end_sample = min(len(drum_mono), end_sample)
                
                if end_sample <= start_sample:
                    classified_onsets[2].append(onset_time)  # Default to kick
                    continue
                
                window = drum_mono[start_sample:end_sample]
                
                if len(window) < 100:
                    classified_onsets[2].append(onset_time)  # Default to kick
                    continue
                
                # Simple frequency classification
                class_id = self._simple_frequency_classify(window)
                classified_onsets[class_id].append(onset_time)
                
            except Exception as e:
                print(f"⚠️  Error in fallback classification: {e}")
                classified_onsets[2].append(onset_time)  # Default to kick
        
        return classified_onsets
    
    def hybrid_onset_classification(self, fused_onsets):
        """Ultra-improved instrument classification using hybrid approach"""
        print("Ultra-improved hybrid onset classification...")
        
        # Initialize hybrid classifier
        self.hybrid_classifier.initialize()
        
        classified_onsets = [[] for _ in range(self.num_class)]
        
        for onset_time in fused_onsets:
            try:
                # Use hybrid classifier
                result = self.hybrid_classifier.classify_onset(onset_time, self.drum_audio)
                
                # Only accept confident predictions
                if result['confidence'] > 0.5:
                    instrument_class = result['class_id']
                    
                    # Ensure class is within valid range
                    if 0 <= instrument_class < self.num_class:
                        classified_onsets[instrument_class].append({
                            'time': onset_time,
                            'confidence': result['confidence'],
                            'velocity': result['velocity'],
                            'instrument': result['instrument']
                        })
                else:
                    # Fallback to original method for low confidence
                    instrument_class = self.classify_instrument_by_frequency(onset_time)
                    if 0 <= instrument_class < self.num_class:
                        classified_onsets[instrument_class].append({
                            'time': onset_time,
                            'confidence': 0.5,
                            'velocity': 0.7,
                            'instrument': 'fallback'
                        })
                        
            except Exception as e:
                print(f"Error in hybrid classification for onset {onset_time}: {e}")
                # Fallback to original method
                instrument_class = self.classify_instrument_by_frequency(onset_time)
                if 0 <= instrument_class < self.num_class:
                    classified_onsets[instrument_class].append({
                        'time': onset_time,
                        'confidence': 0.5,
                        'velocity': 0.7,
                        'instrument': 'fallback'
                    })
        
        # Convert back to simple onset lists for compatibility
        simple_classified_onsets = []
        for class_onsets in classified_onsets:
            simple_classified_onsets.append([onset['time'] for onset in class_onsets])
        
        print(f"Hybrid classified onsets: {[len(class_onsets) for class_onsets in simple_classified_onsets]}")
        return simple_classified_onsets
        
    def peak_picking(self):
        """Apply peak picking algorithm to onset predictions"""
        print("Applying peak picking...")
        
        # Peak picking parameters
        k = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        threshold = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        tolerance = [2, 2, 4, 2, 2, 2, 2, 2, 2, 2]
        
        result_each_class = np.transpose(self.onset_predictions)
        onset_each_class = [[] for i in range(self.num_class)]
        
        for c in range(self.num_class):
            T = max(np.average(result_each_class[c]) * k[c], threshold[c])
            
            for t in range(len(result_each_class[c])):
                is_append = False
                for nearby_result in result_each_class[c][max(t-tolerance[c], 0):min(t+tolerance[c], len(result_each_class[c])-1)+1]:
                    if nearby_result > result_each_class[c][t] or T > result_each_class[c][t]:
                        onset_each_class[c].append(0)
                        is_append = True
                        break
                if not is_append:
                    onset_each_class[c].append(1)
                    
        self.onset_results = onset_each_class
        print("Peak picking completed")
        
    def apply_beat_based_timing(self):
        """Convert onset results to beat-based timing"""
        print("Applying beat-based timing...")
        
        # Convert onset results to time-based format
        time_per_hop = 0.01  # From preprocessing
        onset_times_per_class = []
        
        for class_idx in range(self.num_class):
            class_onset_times = []
            for time_idx, has_onset in enumerate(self.onset_results[class_idx]):
                if has_onset == 1:
                    onset_time = time_idx * time_per_hop
                    class_onset_times.append(onset_time)
            onset_times_per_class.append(class_onset_times)
        
        # Quantize onsets to beat positions for each class
        self.beat_aligned_onsets = []
        for class_idx in range(self.num_class):
            if len(onset_times_per_class[class_idx]) > 0:
                quantized_times = self.quantize_onsets_to_beats(
                    onset_times_per_class[class_idx], 
                    self.beat_times
                )
                self.beat_aligned_onsets.append(quantized_times)
            else:
                self.beat_aligned_onsets.append([])
        
        print("Beat-based timing applied")
        
    def apply_improved_timing(self, classified_onsets):
        """Apply improved beat-based timing to classified onsets"""
        print("Applying improved timing to classified onsets...")
        
        # Initialize beat-aligned onsets for each instrument class
        self.beat_aligned_onsets = [[] for _ in range(self.num_class)]
        
        for instrument_class, onset_times in enumerate(classified_onsets):
            if len(onset_times) > 0:
                # Apply quantization to beat positions
                quantized_times = self.quantize_onsets_to_beats(onset_times, self.beat_times)
                
                # Apply additional filtering based on instrument characteristics
                filtered_times = self.filter_onsets_by_instrument(quantized_times, instrument_class)
                
                self.beat_aligned_onsets[instrument_class] = filtered_times
            else:
                self.beat_aligned_onsets[instrument_class] = []
        
        # Print statistics
        total_onsets = sum(len(onsets) for onsets in self.beat_aligned_onsets)
        print(f"Applied improved timing to {total_onsets} onsets across {self.num_class} instruments")
        for i, onsets in enumerate(self.beat_aligned_onsets):
            if len(onsets) > 0:
                print(f"  {self.CHANNEL_NAMES[self.INT_TO_CHANNEL[i]]}: {len(onsets)} onsets")
        
        print("Improved timing applied")
        
    def filter_onsets_by_instrument(self, onset_times, instrument_class):
        """Filter onsets based on instrument-specific characteristics"""
        if len(onset_times) == 0:
            return onset_times
            
        filtered_onsets = []
        min_interval = self.get_min_interval_for_instrument(instrument_class)
        
        # Remove onsets that are too close together for this instrument
        for i, onset_time in enumerate(onset_times):
            if i == 0:
                filtered_onsets.append(onset_time)
            else:
                time_diff = onset_time - filtered_onsets[-1]
                if time_diff >= min_interval:
                    filtered_onsets.append(onset_time)
        
        return filtered_onsets
        
    def get_min_interval_for_instrument(self, instrument_class):
        """Get minimum interval between onsets for specific instrument"""
        # Define minimum intervals based on instrument characteristics
        # Values in seconds
        instrument_intervals = {
            0: 0.05,  # Hi-hat Close - can be played very fast
            1: 0.1,   # Snare - moderate speed
            2: 0.08,  # Bass Drum - can be played quite fast
            3: 0.12,  # High Tom - moderate speed
            4: 0.12,  # Low Tom - moderate speed
            5: 0.15,  # Ride Cymbal - slower decay
            6: 0.15,  # Floor Tom - slower decay
            7: 0.08,  # Hi-hat Open - can be played fast
            8: 0.2,   # Ride Bell - longer decay
            9: 0.3,   # Crash Cymbal - very long decay
        }
        
        return instrument_intervals.get(instrument_class, 0.1)
        
    def extract_beats(self):
        """Extract beats from audio using the improved hybrid pipeline"""
        print("Starting improved hybrid onset detection pipeline...")
        
        # Step 1: Audio separation and tempo analysis
        self.separate_audio_tracks()
        self.detect_tempo_and_beats()
        
        # Step 2: ML-based onset detection
        self.load_model()
        model_input = self.preprocess_drums()
        self.predict_onsets(model_input)
        self.peak_picking()
        
        # Step 3: Ultra-improved hybrid onset detection and classification
        fused_onsets = self.fuse_onset_detections()
        
        # Track 3: Use Magenta-only classification if enabled
        if self.use_magenta_only:
            print("🔮 Track 3: Using Magenta-only classification")
            classified_onsets = self.magenta_only_classification(fused_onsets)
        else:
            classified_onsets = self.hybrid_onset_classification(fused_onsets)
        
        # Step 4: Apply beat-based timing to classified onsets
        self.apply_improved_timing(classified_onsets)
        
        print("Improved hybrid onset detection pipeline completed")
        
    def create_chart(self):
        """Create DTX chart from onset results"""
        print("Creating DTX chart...")
        # This is handled in export method
        pass
        
    def export(self, output_dir):
        """Export complete DTXMania simfile"""
        print(f"Exporting to {output_dir}...")
        
        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract simfile template
        with zipfile.ZipFile('/app/SimfilesTemplate.zip', 'r') as zip_ref:
            zip_ref.extractall('/tmp')
            
        # Copy template to output directory
        song_output_dir = os.path.join(output_dir, self.song_name)
        if os.path.exists(song_output_dir):
            rmtree(song_output_dir)
        copytree("/tmp/Simfiles", song_output_dir)
        
        # Handle BGM audio based on configuration
        if self.use_original_bgm:
            # Copy original audio file
            original_bgm_path = os.path.join(song_output_dir, self.original_filename)
            copy2(self.input_audio_path, original_bgm_path)
            print(f"Using original audio as BGM: {self.original_filename}")
        else:
            # Save separated BGM audio
            bgm_path = os.path.join(song_output_dir, 'bgm.wav')
            sf.write(bgm_path, self.bgm_audio, 44100)
            print("Using separated BGM (without drums)")
        
        # Create DTX file
        self.create_dtx_file(song_output_dir)
        
        # Create ZIP file
        zip_path = os.path.join(output_dir, f'{self.song_name}.zip')
        self.zip_directory(song_output_dir, zip_path)
        
        print(f"Export completed: {zip_path}")
        
    def create_dtx_file(self, output_dir):
        """Create DTX file with chart data"""
        dtx_header = f"""; {self.metadata['comment']}
#TITLE: {self.metadata['title']}
#ARTIST: {self.metadata['artist']}
#AUTHOR: {self.metadata['author']}
#BPM: {self.tempo_bpm:.0f}
#DLEVEL: {self.metadata['difficulty']}
#GENRE: {self.metadata['genre']}
#HIDDENLEVEL ON

#WAV11: drums/crash.ogg
#VOLUME11: 10
#PAN11: -60
#WAV12: drums/hihat_close.ogg
#VOLUME12: 10
#PAN12: -70
#WAV14: drums/snare.ogg
#VOLUME14: 10
#WAV15: drums/kick.ogg
#VOLUME15: 10
#WAV16: drums/high_tom.ogg
#VOLUME16: 10
#PAN16: -50
#WAV17: drums/low_tom.ogg
#VOLUME17: 10
#PAN17: 20
#WAV18: drums/floor_tom.ogg
#VOLUME18: 10
#PAN18: 50
#WAV1B: drums/ride_bell.ogg
#VOLUME1B: 10
#PAN1B: 60
#WAV1O: drums/hihat_open.ogg
#VOLUME1O: 10
#PAN1O: -70
#WAV1R: drums/ride.ogg
#VOLUME1R: 10
#PAN1R: 60

#WAVZZ: {self.original_filename if self.use_original_bgm else 'bgm.wav'}
#VOLUMEZZ: 50
#BGMWAV: ZZ

"""
        
        # DTX channel mappings
        wav_mappings = {
            0: "12",  # Hi-hat Close
            1: "14",  # Snare
            2: "15",  # Bass Drum
            3: "16",  # High Tom
            4: "17",  # Low Tom
            5: "1R",  # Ride Cymbal
            6: "18",  # Floor Tom
            7: "1O",  # Hi-hat Open
            8: "1B",  # Ride Bell
            9: "11",  # Crash Cymbal
        }
        
        dtx_path = os.path.join(output_dir, f'{self.song_name}.dtx')
        with open(dtx_path, "w+", encoding='utf-8') as dtx_file:
            dtx_file.write(dtx_header)
            
            bar_before_song_begin = 2
            
            # Calculate bars and their timing based on detected beats
            bar_positions, bar_times = self.calculate_adaptive_bar_timing(self.beat_times)
            
            # Line to start BGM
            dtx_file.write(f'#{bar_before_song_begin:03}01: ZZ\n')
            
            # Generate notes based on beat-aligned onsets
            self.generate_beat_based_notes(dtx_file, bar_before_song_begin, wav_mappings)
            
    def generate_beat_based_notes(self, dtx_file, bar_offset, wav_mappings):
        """Generate DTX notes based on beat-aligned timing"""
        print("Generating beat-based DTX notes...")
        
        # Calculate total song duration and resolution
        song_duration = len(self.beat_times) * (60.0 / self.tempo_bpm)
        resolution = 192  # DTX resolution per quarter note
        
        # Create note grid based on beats
        max_bars = int(song_duration / (240.0 / self.tempo_bpm)) + 1  # 4 beats per bar
        note_grid = {}
        
        for bar_num in range(max_bars):
            for instrument in range(self.num_class):
                channel = self.INT_TO_CHANNEL[instrument]
                note_grid[f'{bar_num}_{channel}'] = ['00'] * resolution
        
        # Place notes from beat-aligned onsets
        for instrument in range(self.num_class):
            channel = self.INT_TO_CHANNEL[instrument]
            
            for onset_time in self.beat_aligned_onsets[instrument]:
                # Convert time to bar and position
                bar_num = int(onset_time / (240.0 / self.tempo_bpm))
                bar_time = onset_time - (bar_num * (240.0 / self.tempo_bpm))
                position = int((bar_time / (240.0 / self.tempo_bpm)) * resolution)
                
                if position < resolution and bar_num < max_bars:
                    key = f'{bar_num}_{channel}'
                    if key in note_grid:
                        note_grid[key][position] = wav_mappings[instrument]
        
        # Write notes to DTX file
        for bar_num in range(max_bars):
            for instrument in range(self.num_class):
                channel = self.INT_TO_CHANNEL[instrument]
                key = f'{bar_num}_{channel}'
                
                if key in note_grid:
                    notes_line = ''.join(note_grid[key])
                    # Only write non-empty lines
                    if notes_line.replace('00', ''):
                        dtx_file.write(f'#{bar_num + bar_offset:03}{channel}: {notes_line}\n')
        
        print("Beat-based DTX notes generated")
                    
    def zip_directory(self, folder_path, zip_path):
        """Create ZIP file from directory"""
        with zipfile.ZipFile(zip_path, mode='w') as zipf:
            len_dir_path = len(folder_path)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, file_path[len_dir_path:])