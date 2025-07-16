# Plan POC Magenta - Audio2DTX

## Objectif
Remplacer la simulation Magenta actuelle par le vrai modèle E-GMD pour tester l'amélioration de qualité réelle.

## État Actuel
- **Système hybride** : Magenta (40%) + Features (30%) + Contexte (30%)
- **Problème** : Magenta utilise une simulation basique au lieu du vrai modèle
- **Gain attendu** : Classification plus précise des 10 types d'instruments de batterie

## Phase 1 : Résolution Dépendances (Semaines 1-2)

### Versions Compatibles Magenta
```bash
# Downgrade requis pour compatibilité Magenta
numpy==1.21.6      # (était 1.23.5)
librosa==0.7.2     # (était 0.8.1) 
tensorflow==2.11.0 # (était 2.12.1)
matplotlib==3.5.2  # (était 3.6.3)
magenta==2.1.4     # nouveau
```

### Actions
1. **Mise à jour requirements.txt** avec versions compatibles
2. **Test du pipeline existant** avec librosa 0.7.2 (vérifier API changes)
3. **Mise à jour Dockerfile** pour build avec nouvelles dépendances
4. **Validation** que l'application actuelle fonctionne toujours

## Phase 2 : Implémentation Magenta Réel (Semaines 3-4)

### Objectif
Remplacer `_simulate_magenta_prediction()` par une vraie inference avec le modèle E-GMD de Magenta.

### Composants à Implémenter

#### 1. Chargement Modèle E-GMD
```python
def load_model(self):
    """Charge le modèle Magenta OaF Drums"""
    try:
        import magenta
        from magenta.models.onsets_frames_transcription import model_util
        
        # Chargement checkpoint E-GMD
        self.oaf_model = model_util.get_default_hparams()
        # ... configuration modèle
        
    except ImportError:
        print("Magenta not available, using fallback")
        return None
```

#### 2. Preprocessing Audio Compatible
```python
def _prepare_for_magenta(self, onset_audio):
    """Preprocessing audio pour Magenta OaF"""
    # Normalisation pour Magenta
    normalized = onset_audio / np.max(np.abs(onset_audio))
    
    # Spectrogramme mel (format attendu par E-GMD)
    mel_spec = librosa.feature.melspectrogram(
        y=normalized, 
        sr=self.sample_rate,
        n_mels=128,
        hop_length=256
    )
    return librosa.power_to_db(mel_spec)
```

#### 3. Remplacement de la Simulation
```python
def predict_instrument(self, onset_audio, onset_time):
    """Remplace _simulate_magenta_prediction par vraie inference"""
    if self.oaf_model is not None:
        # Vrai Magenta
        spectrogram = self._prepare_for_magenta(onset_audio)
        prediction = self.oaf_model.predict(spectrogram)
        return self._map_to_dtx_format(prediction)
    else:
        # Fallback existant
        return self._simulate_magenta_prediction(onset_audio, onset_time)
```

### Intégration
- Garder le système de vote hybride intact (40% Magenta)
- Préserver le fallback si Magenta échoue
- Pas de changement dans l'interface existante

## Livrable POC
Version fonctionnelle avec vrai modèle Magenta prête pour test manuel.