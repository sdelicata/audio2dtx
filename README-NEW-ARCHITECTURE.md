# Audio2DTX - Nouvelle Architecture

## Vue d'ensemble

Cette version refactorisée d'Audio2DTX introduit une architecture modulaire qui sépare clairement les responsabilités et améliore la maintenabilité du code.

## Structure du projet

```
audio2dtx/
├── src/audio2dtx/           # Code source principal
│   ├── audio/               # Traitement audio
│   ├── classification/      # Classification des instruments
│   ├── core/               # Logique métier principale
│   ├── config/             # Configuration
│   ├── dtx/                # Génération DTX
│   ├── services/           # Services externes
│   └── utils/              # Utilitaires
├── main_new.py             # Nouveau point d'entrée
├── compatibility.py        # Couche de compatibilité
└── requirements-new.txt    # Dépendances
```

## Avantages de la nouvelle architecture

### 1. Séparation des responsabilités
- **Audio Processing**: Chargement, préprocessing, séparation
- **Classification**: Détection et classification des instruments
- **Core Logic**: Orchestration du pipeline
- **DTX Generation**: Création des fichiers de sortie

### 2. Modularité
- Chaque composant est indépendant et testable
- Interfaces claires entre les modules
- Facilite l'ajout de nouvelles fonctionnalités

### 3. Configuration centralisée
- Système de configuration unifié avec `Settings`
- Support des variables d'environnement
- Configuration par fichier YAML

### 4. Gestion d'erreurs robuste
- Exceptions personnalisées par domaine
- Logging structuré et configurable
- Validation des entrées

## Utilisation

### Mode compatibilité (API existante)

```python
from audio2dtx.compatibility import AudioToChart

# Utilisation identique à l'ancienne version
chart = AudioToChart(
    input_audio="song.mp3",
    metadata={"title": "Ma Chanson", "artist": "Mon Artiste"},
    use_ensemble=True
)

chart.extract_beats()
chart.create_chart()
chart.export("/output")
```

### Nouvelle API modulaire

```python
from audio2dtx.core.audio_processor import AudioProcessor
from audio2dtx.config.settings import get_settings

# Nouvelle approche modulaire
processor = AudioProcessor(get_settings())

dtx_path = processor.process_audio_file(
    input_file="song.mp3",
    output_dir="/output",
    metadata={"title": "Ma Chanson", "artist": "Mon Artiste"},
    track_type="ensemble"
)
```

### CLI modernisé

```bash
# Utilisation du nouveau CLI
python main_new.py song.mp3 --use-ensemble --title "Ma Chanson"

# Test des composants
python main_new.py --test-components

# Mode verbose
python main_new.py song.mp3 --verbose --batch
```

## Configuration

### Fichier de configuration YAML

```yaml
# config.yaml
audio:
  sample_rate: 44100
  hop_length: 512

classification:
  confidence_threshold: 0.6
  n_mfcc: 13

services:
  magenta_url: "http://magenta-service:5000"
  magenta_timeout: 30.0

dtx:
  resolution: 192
  bars_before_song: 2
```

### Variables d'environnement

```bash
export MAGENTA_SERVICE_URL="http://localhost:5000"
export AUDIO2DTX_SAMPLE_RATE=44100
export AUDIO2DTX_CONFIDENCE_THRESHOLD=0.7
```

## Extensibilité

### Ajout d'un nouveau track de classification

1. Créer une classe héritant de `BaseClassifier`
2. Implémenter les méthodes `initialize()` et `classify_onset()`
3. Ajouter le track au système de routing

```python
from audio2dtx.classification.base_classifier import BaseClassifier

class MonNouveauTrack(BaseClassifier):
    def initialize(self):
        # Initialisation du modèle/algorithme
        pass
    
    def classify_onset(self, audio_window, onset_time, context=None):
        # Classification d'un onset
        return ClassificationResult(instrument, confidence, velocity)
```

### Ajout d'un nouveau service externe

1. Créer un client dans `services/`
2. Implémenter les méthodes de communication
3. Ajouter la configuration nécessaire

## Migration depuis l'ancienne version

### Compatibilité complète
- L'ancienne API reste fonctionnelle via `compatibility.py`
- Aucun changement nécessaire dans le code existant
- Migration progressive possible

### Étapes de migration

1. **Phase 1**: Utiliser la couche de compatibilité
2. **Phase 2**: Migrer vers la nouvelle API module par module
3. **Phase 3**: Adopter les nouveaux patterns et configurations

## Tests

```bash
# Installation des dépendances de test
pip install pytest pytest-cov

# Lancement des tests
pytest tests/

# Tests avec couverture
pytest --cov=audio2dtx tests/
```

## Développement

### Setup de développement

```bash
# Installation en mode développement
pip install -e .

# Installation des outils de développement
pip install black flake8 mypy

# Formatage du code
black src/

# Vérification du style
flake8 src/

# Vérification des types
mypy src/
```

### Structure des tests

```
tests/
├── unit/              # Tests unitaires
│   ├── test_audio/
│   ├── test_classification/
│   └── test_core/
├── integration/       # Tests d'intégration
└── e2e/              # Tests end-to-end
```

## Roadmap

### Phase 2: Extraction des tracks
- Migration des tracks 3-9 vers la nouvelle architecture
- Optimisation des algorithmes de classification
- Tests de performance

### Phase 3: Services avancés
- Amélioration du client Magenta
- Cache intelligent des features
- Monitoring et métriques

### Phase 4: Tests et documentation
- Suite de tests complète
- Documentation API
- Guides d'utilisation

### Phase 5: Optimisations
- Performance et parallélisation
- Optimisation Docker
- Scaling horizontal

## Support

- **Issues**: Utiliser le système d'issues GitHub
- **Documentation**: Voir `/docs` pour la documentation complète
- **Exemples**: Voir `/examples` pour des cas d'usage