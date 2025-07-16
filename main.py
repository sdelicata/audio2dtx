import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

INPUT_FILE = 'input/song.mp3'
OUTPUT_DIR = 'output'

# Initialiser
separator = Separator('spleeter:4stems')
audio_loader = AudioAdapter.default()

# Charger l'audio
print(f"🔊 Chargement de {INPUT_FILE}")
waveform, _ = audio_loader.load(INPUT_FILE, sample_rate=44100)

# Séparer les stems
print("🎛️ Séparation des stems...")
prediction = separator.separate(waveform)

# Créer output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Exporter tous les stems séparément
for stem in prediction:
    print(f"💾 Export du stem : {stem}")
    stem_mono = librosa.to_mono(np.transpose(prediction[stem]))
    sf.write(os.path.join(OUTPUT_DIR, f"{stem}.wav"), stem_mono, 44100)

# Créer le BGM (tout sauf drums)
print("🎼 Construction du BGM (sans drums)...")
bgm = (librosa.to_mono(np.transpose(prediction['vocals'])) +
       librosa.to_mono(np.transpose(prediction['bass'])) +
       librosa.to_mono(np.transpose(prediction['other']))) / 3

sf.write(os.path.join(OUTPUT_DIR, "bgm.wav"), bgm, 44100)
print("✅ bgm.wav exporté")

# (Optionnel) Générer un spectrogramme pour chaque stem
print("📊 Génération des spectrogrammes...")
for stem in prediction:
    y = librosa.to_mono(np.transpose(prediction[stem]))
    S = librosa.feature.melspectrogram(y=y, sr=44100, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=44100, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogramme - {stem}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{stem}_spectrogram.png"))
    plt.close()
