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
        
    def extract_beats(self):
        """Extract beats from audio using the complete pipeline"""
        self.separate_audio_tracks()
        self.load_model()
        model_input = self.preprocess_drums()
        self.predict_onsets(model_input)
        self.peak_picking()
        
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
#BPM: 120
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
            bar = 0
            old_bar = bar
            notes = [""] * self.num_class
            
            # Line to start BGM
            dtx_file.write(f'#{int(old_bar + bar_before_song_begin):03}01: ZZ\n')
            
            for time, onset in enumerate(np.transpose(self.onset_results)):
                for instrument, result in enumerate(onset):
                    if result == 1:
                        notes[instrument] += wav_mappings[instrument]
                    elif result == 0:
                        notes[instrument] += "00"
                        
                bar = math.floor(time / (2 * 100))
                if old_bar != bar:
                    for instrument in range(len(onset)):
                        dtx_file.write(f'#{int(old_bar + bar_before_song_begin):03}{self.INT_TO_CHANNEL[instrument]}: {notes[instrument]}\n')
                    notes = [""] * self.num_class
                    old_bar = bar
                    
    def zip_directory(self, folder_path, zip_path):
        """Create ZIP file from directory"""
        with zipfile.ZipFile(zip_path, mode='w') as zipf:
            len_dir_path = len(folder_path)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, file_path[len_dir_path:])