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
import warnings
warnings.filterwarnings('ignore')


class AudioToChart:
    def __init__(self, input_audio_path, use_original_bgm=True):
        self.input_audio_path = input_audio_path
        self.song_name = os.path.splitext(os.path.basename(input_audio_path))[0]
        self.original_filename = os.path.basename(input_audio_path)
        self.use_original_bgm = use_original_bgm
        self.setup_channel_mappings()
        self.model = None
        self.drum_audio = None
        self.onset_predictions = None
        self.onset_results = None
        
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
                onset_strength = librosa.onset.onset_strength(
                    y=drum_mono, 
                    sr=self.sample_rate,
                    hop_length=hop_length,
                    feature=method_name,
                    aggregate=np.median
                )
                
                # Detect onsets from strength function
                onsets = librosa.onset.onset_detect(
                    onset_envelope=onset_strength,
                    sr=self.sample_rate,
                    hop_length=hop_length,
                    threshold=method_config['threshold'],
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
        
        # Step 3: Hybrid onset detection and classification
        fused_onsets = self.fuse_onset_detections()
        classified_onsets = self.improve_onset_classification(fused_onsets)
        
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
        dtx_header = f"""; Auto-generated by AudioToChart

#TITLE: {self.song_name}
#ARTIST: -
#BPM: {self.tempo_bpm:.0f}
#DLEVEL: 1
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