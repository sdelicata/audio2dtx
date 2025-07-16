# Plan Détaillé - Amélioration Ultra-Avancée de la Détection d'Instruments de Batterie

## 📊 État Actuel (Progress)

### ✅ Terminé
- [x] Ajout des dépendances TensorFlow-Magenta et nouvelles librairies
- [x] Mise à jour requirements.txt avec magenta==2.1.4, tensorflow-probability==0.19.0, pretty_midi==0.2.10, scikit-learn==1.2.2, mir-eval==0.7
- [x] Ajout des imports nécessaires (sklearn, scipy, warnings)

### 🔄 En Cours
- [ ] Implémentation AdvancedFeatureExtractor avec MFCC, spectral contrast, chroma, tonnetz

### 📋 À Faire
- [ ] Créer MagentaDrumClassifier pour intégrer le modèle OaF Drums
- [ ] Développer HybridDrumClassifier combinant plusieurs approches
- [ ] Intégrer le système hybride dans le pipeline principal
- [ ] Tester et valider les améliorations de performance

## 🎯 Objectifs du Projet

### Problème Identifié
L'approche actuelle basée uniquement sur l'analyse FFT et les bandes de fréquence est insuffisante :
- Classification imprécise des instruments (snare ~40%, hi-hat ~30%, toms ~20%)
- Pas de distinction entre hi-hat open/close
- Mauvaise détection des nuances et vélocité
- Timing approximatif des onsets

### Objectifs de Performance
- **Kick Detection** : 90%+ (vs ~60% actuel)
- **Snare Detection** : 85%+ (vs ~40% actuel)
- **Hi-hat Classification** : 80%+ (vs ~30% actuel)
- **Toms/Cymbales** : 75%+ (vs ~20% actuel)

## 🏗️ Architecture du Système

### 1. AdvancedFeatureExtractor
```python
class AdvancedFeatureExtractor:
    def __init__(self, sr=44100, n_mfcc=13, n_chroma=12):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.scaler = StandardScaler()
        
    def extract_features(self, audio_window):
        # MFCC (13 coefficients)
        # Spectral Contrast (7 features)
        # Chroma Features (12 features)
        # Tonnetz (6 features)
        # Spectral Statistics (centroïde, rolloff, flatness, bandwidth)
        # Temporal Features (ZCR, RMS, attack/decay)
```

### 2. MagentaDrumClassifier
```python
class MagentaDrumClassifier:
    def __init__(self):
        self.model = None  # Modèle Magenta OaF Drums
        self.confidence_threshold = 0.7
        
    def load_model(self):
        # Charger le modèle pré-entraîné OaF Drums
        
    def classify_onsets(self, drum_audio, onset_times):
        # Utiliser OaF Drums pour classification
        # Retourner classe + confidence + vélocité
```

### 3. HybridDrumClassifier
```python
class HybridDrumClassifier:
    def __init__(self):
        self.feature_extractor = AdvancedFeatureExtractor()
        self.magenta_classifier = MagentaDrumClassifier()
        self.weights = {
            'magenta': 0.4,
            'features': 0.3,
            'context': 0.3
        }
        
    def classify_onset(self, onset_time, drum_audio):
        # Combiner les 3 approches avec vote pondéré
        # Validation croisée des résultats
```

## 📝 Plan d'Implémentation Détaillé

### Phase 1 : AdvancedFeatureExtractor (EN COURS)

#### 1.1 Structure de Base
```python
class AdvancedFeatureExtractor:
    def __init__(self, sr=44100, n_mfcc=13, n_chroma=12, n_contrast=7):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.n_contrast = n_contrast
        self.scaler = StandardScaler()
        self.feature_cache = {}
```

#### 1.2 Méthodes d'Extraction
- `extract_mfcc(audio_window)` : 13 coefficients MFCC
- `extract_spectral_contrast(audio_window)` : 7 features de contraste spectral
- `extract_chroma(audio_window)` : 12 features chromatiques
- `extract_tonnetz(audio_window)` : 6 features tonnetz
- `extract_spectral_stats(audio_window)` : centroïde, rolloff, flatness, bandwidth
- `extract_temporal_features(audio_window)` : ZCR, RMS, attack/decay

#### 1.3 Méthode Principale
```python
def extract_comprehensive_features(self, audio_window):
    features = {}
    
    # MFCC (13 features)
    features['mfcc'] = self.extract_mfcc(audio_window)
    
    # Spectral Contrast (7 features)
    features['spectral_contrast'] = self.extract_spectral_contrast(audio_window)
    
    # Chroma (12 features)
    features['chroma'] = self.extract_chroma(audio_window)
    
    # Tonnetz (6 features)
    features['tonnetz'] = self.extract_tonnetz(audio_window)
    
    # Spectral Statistics (4 features)
    features['spectral_stats'] = self.extract_spectral_stats(audio_window)
    
    # Temporal Features (5 features)
    features['temporal'] = self.extract_temporal_features(audio_window)
    
    # Combiner toutes les features (47 features total)
    combined_features = np.concatenate([
        features['mfcc'],
        features['spectral_contrast'],
        features['chroma'],
        features['tonnetz'],
        features['spectral_stats'],
        features['temporal']
    ])
    
    return combined_features, features
```

### Phase 2 : MagentaDrumClassifier

#### 2.1 Installation et Configuration
```python
# Dans requirements.txt (déjà fait)
magenta==2.1.4
tensorflow-probability==0.19.0
pretty_midi==0.2.10
```

#### 2.2 Intégration du Modèle
```python
class MagentaDrumClassifier:
    def __init__(self):
        self.model = None
        self.confidence_threshold = 0.7
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
        
    def load_model(self):
        # Charger le modèle OaF Drums pré-entraîné
        try:
            from magenta.models.onsets_frames_transcription import model
            self.model = model.load_model()
        except ImportError:
            print("Magenta not available, using fallback")
            self.model = None
            
    def classify_onsets(self, drum_audio, onset_times):
        if self.model is None:
            return self.fallback_classification(drum_audio, onset_times)
            
        # Utiliser le modèle Magenta pour classification
        predictions = self.model.predict(drum_audio)
        
        # Mapper les onsets aux prédictions
        classified_onsets = []
        for onset_time in onset_times:
            prediction = self.get_prediction_at_time(predictions, onset_time)
            if prediction['confidence'] > self.confidence_threshold:
                classified_onsets.append({
                    'time': onset_time,
                    'instrument': prediction['instrument'],
                    'confidence': prediction['confidence'],
                    'velocity': prediction['velocity']
                })
                
        return classified_onsets
```

### Phase 3 : HybridDrumClassifier

#### 3.1 Système de Vote Pondéré
```python
class HybridDrumClassifier:
    def __init__(self):
        self.feature_extractor = AdvancedFeatureExtractor()
        self.magenta_classifier = MagentaDrumClassifier()
        self.weights = {
            'magenta': 0.4,
            'features': 0.3,
            'context': 0.3
        }
        self.confidence_threshold = 0.6
        
    def classify_onset(self, onset_time, drum_audio, context_window=0.1):
        # Extraire fenêtre audio autour de l'onset
        window = self.extract_audio_window(drum_audio, onset_time, context_window)
        
        # Approche 1: Magenta OaF Drums
        magenta_result = self.magenta_classifier.classify_single_onset(window)
        
        # Approche 2: Features avancées + ML
        features = self.feature_extractor.extract_comprehensive_features(window)
        features_result = self.classify_with_features(features)
        
        # Approche 3: Analyse contextuelle
        context_result = self.analyze_context(onset_time, drum_audio)
        
        # Vote pondéré
        final_result = self.weighted_vote(magenta_result, features_result, context_result)
        
        return final_result
```

#### 3.2 Classification par Features
```python
def classify_with_features(self, features):
    # Utiliser un modèle ML entraîné sur les features
    # Ou utiliser des règles basées sur les features
    
    mfcc = features['mfcc']
    spectral_contrast = features['spectral_contrast']
    chroma = features['chroma']
    tonnetz = features['tonnetz']
    spectral_stats = features['spectral_stats']
    temporal = features['temporal']
    
    # Règles de classification basées sur les features
    if temporal['attack_time'] < 0.01 and spectral_stats['centroid'] < 200:
        return {'instrument': 'kick', 'confidence': 0.8}
    elif spectral_contrast[2] > 0.5 and temporal['zcr'] > 0.1:
        return {'instrument': 'snare', 'confidence': 0.7}
    # ... autres règles
```

### Phase 4 : Intégration dans le Pipeline

#### 4.1 Modification de la Classe AudioToChart
```python
class AudioToChart:
    def __init__(self, input_audio_path, use_original_bgm=True):
        # ... existing code ...
        self.hybrid_classifier = HybridDrumClassifier()
        
    def improved_instrument_classification(self, fused_onsets):
        """Classification ultra-améliorée des instruments"""
        print("Classification ultra-améliorée des instruments...")
        
        classified_onsets = [[] for _ in range(self.num_class)]
        
        for onset_time in fused_onsets:
            # Utiliser le système hybride
            result = self.hybrid_classifier.classify_onset(
                onset_time, 
                self.drum_audio
            )
            
            if result['confidence'] > 0.6:
                instrument_class = self.map_instrument_to_class(result['instrument'])
                classified_onsets[instrument_class].append({
                    'time': onset_time,
                    'confidence': result['confidence'],
                    'velocity': result.get('velocity', 0.7)
                })
        
        return classified_onsets
```

#### 4.2 Nouveau Pipeline
```python
def extract_beats(self):
    """Pipeline amélioré avec classification hybride"""
    print("Starting ultra-improved hybrid onset detection pipeline...")
    
    # Étapes 1-3: Audio separation, tempo, ML model (inchangé)
    self.separate_audio_tracks()
    self.detect_tempo_and_beats()
    self.load_model()
    model_input = self.preprocess_drums()
    self.predict_onsets(model_input)
    self.peak_picking()
    
    # Étape 4: Fusion des onsets (inchangé)
    fused_onsets = self.fuse_onset_detections()
    
    # Étape 5: Classification ultra-améliorée (NOUVEAU)
    classified_onsets = self.improved_instrument_classification(fused_onsets)
    
    # Étape 6: Timing amélioré
    self.apply_improved_timing(classified_onsets)
    
    print("Ultra-improved hybrid onset detection pipeline completed")
```

## 🧪 Tests et Validation

### Tests Unitaires
```python
class TestAdvancedFeatureExtractor:
    def test_mfcc_extraction(self):
        # Test MFCC extraction
        
    def test_spectral_contrast(self):
        # Test spectral contrast
        
    def test_feature_dimensions(self):
        # Vérifier que les dimensions sont correctes
```

### Tests d'Intégration
```python
class TestHybridClassifier:
    def test_kick_detection(self):
        # Test sur échantillons de kick
        
    def test_snare_detection(self):
        # Test sur échantillons de snare
        
    def test_hihat_classification(self):
        # Test distinction hi-hat open/close
```

### Métriques de Performance
```python
def evaluate_performance(self, ground_truth, predictions):
    # Precision, Recall, F1-Score par instrument
    # Confusion Matrix
    # Accuracy globale
    # Temps de traitement
```

## 📁 Structure des Fichiers

```
audio2dtx/
├── audio_to_chart.py          # Classe principale (modifiée)
├── advanced_features.py       # AdvancedFeatureExtractor (nouveau)
├── magenta_classifier.py      # MagentaDrumClassifier (nouveau)
├── hybrid_classifier.py       # HybridDrumClassifier (nouveau)
├── requirements.txt           # Dépendances (modifié)
├── plan.md                   # Ce fichier
└── tests/
    ├── test_features.py       # Tests unitaires
    ├── test_magenta.py        # Tests Magenta
    └── test_hybrid.py         # Tests système hybride
```

## 🚀 Prochaines Étapes

### Immédiate (Reprendre ici)
1. Finir l'implémentation de `AdvancedFeatureExtractor`
2. Créer les méthodes d'extraction pour chaque type de feature
3. Tester l'extraction de features sur des échantillons

### Moyen Terme
1. Intégrer le modèle Magenta OaF Drums
2. Créer le système de vote pondéré
3. Tester les performances sur le fichier song.mp3

### Long Terme
1. Optimiser les poids du système hybride
2. Ajouter des métriques de performance
3. Créer une interface de debugging pour visualiser les features

## 🔧 Commandes pour Reprendre

```bash
# Construire avec les nouvelles dépendances
make clean && make build

# Tester le système actuel
make run

# Développer les nouvelles features
# Modifier audio_to_chart.py pour ajouter AdvancedFeatureExtractor
```

## 📊 Résultats Attendus

### Performance Cible
- **Kick** : 90%+ précision
- **Snare** : 85%+ précision  
- **Hi-hat** : 80%+ précision (avec distinction open/close)
- **Toms** : 75%+ précision (avec distinction high/low/floor)
- **Cymbales** : 75%+ précision (crash/ride/bell)

### Temps de Traitement
- Maintenir < 2 minutes pour un fichier de 3 minutes
- Optimiser les calculs avec mise en cache des features

### Qualité DTX
- Rythmes plus naturels et musicaux
- Meilleure distribution des instruments
- Vélocité et expressivité améliorées

---

**Note** : Ce plan peut être exécuté étape par étape. Chaque phase peut être testée indépendamment avant de passer à la suivante.