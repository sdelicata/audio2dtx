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
from shutil import copytree, rmtree


class AudioToChart:
    def __init__(self, input_audio_path):
        self.input_audio_path = input_audio_path
        self.song_name = os.path.splitext(os.path.basename(input_audio_path))[0]
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
        
    def predict_onsets(self, model_input):
        """Use ML model to predict drum onsets"""
        print("Predicting onsets...")
        self.onset_predictions = self.model.predict(model_input)
        print("Onset prediction completed")
        
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
        
    def extract_beats(self):
        """Extract beats from audio using the complete pipeline"""
        self.separate_audio_tracks()
        self.detect_tempo_and_beats()
        self.load_model()
        model_input = self.preprocess_drums()
        self.predict_onsets(model_input)
        self.peak_picking()
        self.apply_beat_based_timing()
        
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
        copytree("/tmp/Simfiles", song_output_dir)
        
        # Save BGM audio
        bgm_path = os.path.join(song_output_dir, 'bgm.wav')
        sf.write(bgm_path, self.bgm_audio, 44100)
        
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

#WAVZZ: bgm.wav
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