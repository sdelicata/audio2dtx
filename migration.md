# Migration vers l'Image Docker TensorFlow/Magenta Officielle

## Contexte Actuel

La solution actuelle utilise une architecture hybride avec :
- **Service Principal** : Container Python 3.10 avec pipeline audio complet
- **Service Magenta** : Container Python 3.9 avec simulation avancée
- **Communication** : API REST entre containers
- **Fallback** : Analyse spectrale sophistiquée au lieu du vrai Magenta

## Problème Identifié

**Nous n'utilisons pas l'image officielle `tensorflow/magenta`** :
- Solution actuelle : `FROM python:3.9-slim` + simulation
- Objectif manqué : Utiliser les vrais modèles Magenta pré-entraînés
- Raisons : Problèmes de compatibilité Python 2.7 vs Python 3.x

## Options de Migration

### Option A : Adapter pour Python 2.7 (Image Officielle)

**Principe** : Utiliser `tensorflow/magenta:latest` tel quel

**Avantages** :
- Utilise l'image officielle Google
- Accès direct aux modèles E-GMD
- Pas de problèmes de compatibilité TensorFlow

**Inconvénients** :
- Python 2.7 déprécié depuis 2020
- Réécriture complète du service Flask
- Versions limitées des dépendances

**Implémentation** :
```dockerfile
FROM tensorflow/magenta:latest

# Installer Flask compatible Python 2.7
RUN pip install flask==1.1.4 requests==2.25.1

# Copier service réécrit en Python 2.7
COPY magenta_service_py27.py /app/magenta_service.py

# Reste identique...
```

### Option B : Multi-stage Build (Recommandée)

**Principe** : Extraire les modèles de l'image officielle, les utiliser dans Python 3.x

**Avantages** :
- Utilise les vrais modèles Magenta officiels
- Code moderne en Python 3.x
- Image finale optimisée
- Meilleure sécurité (Python 3.x)

**Inconvénients** :
- Build plus complexe
- Nécessite compréhension des chemins de modèles Magenta

**Implémentation** :
```dockerfile
# Stage 1: Extraire les modèles Magenta
FROM tensorflow/magenta:latest AS magenta-models
RUN find /magenta -name "*.pb" -o -name "*.ckpt*" -o -name "*.json" | head -20

# Stage 2: Service Python 3.x avec modèles
FROM python:3.9-slim
COPY --from=magenta-models /magenta/models /app/magenta_models/
COPY requirements-magenta.txt /app/
RUN pip install -r requirements-magenta.txt
COPY magenta_service.py /app/

# Charger modèles via TensorFlow directement
ENV MAGENTA_MODELS_PATH=/app/magenta_models
```

### Option C : Image Alternative Maintenue

**Principe** : Utiliser `xychelsea/magenta:latest` (Python 3.x)

**Avantages** :
- Python 3.x moderne
- Magenta pré-installé et fonctionnel
- Pas de réécriture de code
- GPU support disponible

**Inconvénients** :
- Image lourde (>1GB)
- Dépendance externe (non-officielle)
- Temps de build plus long

**Implémentation** :
```dockerfile
FROM xychelsea/magenta:latest

# Installer dépendances additionnelles
RUN pip install flask==2.3.3 requests==2.31.0

# Copier service (code actuel fonctionne)
COPY magenta_service.py /app/
```

## Recommandation : Option B (Multi-stage)

**Justification** :
1. **Authenticité** : Utilise les vrais modèles Google Magenta
2. **Modernité** : Code Python 3.x maintenable
3. **Performance** : Image finale optimisée
4. **Sécurité** : Python 3.x avec mises à jour

## Guide d'Implémentation

### Étape 1 : Recherche des Modèles

```bash
# Inspecter l'image officielle pour trouver les modèles
docker run --rm tensorflow/magenta:latest find /magenta -name "*.pb" -o -name "*.ckpt*" | head -10
```

### Étape 2 : Nouveau Dockerfile

```dockerfile
# Dockerfile.magenta.official
FROM tensorflow/magenta:latest AS magenta-models
RUN mkdir -p /models && \
    find /magenta -name "*drums*" -type f | xargs -I {} cp {} /models/

FROM python:3.9-slim
RUN apt-get update && apt-get install -y libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=magenta-models /models /app/magenta_models/
COPY requirements-magenta.txt ./
RUN pip install -r requirements-magenta.txt

COPY magenta_service_official.py ./magenta_service.py
EXPOSE 5000
CMD ["python", "magenta_service.py"]
```

### Étape 3 : Service Adapté

```python
# magenta_service_official.py
import tensorflow as tf
import os

class MagentaDrumService:
    def __init__(self):
        self.model_path = "/app/magenta_models"
        self.model = self._load_real_magenta_model()
    
    def _load_real_magenta_model(self):
        # Charger le vrai modèle E-GMD
        model_files = os.listdir(self.model_path)
        pb_files = [f for f in model_files if f.endswith('.pb')]
        
        if pb_files:
            graph_def = tf.GraphDef()
            with open(os.path.join(self.model_path, pb_files[0]), 'rb') as f:
                graph_def.ParseFromString(f.read())
            
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name='')
                return graph
        return None
    
    def classify_drums(self, audio_window):
        if self.model:
            # Utiliser le vrai modèle Magenta
            return self._real_magenta_inference(audio_window)
        else:
            # Fallback si pas de modèle
            return self._enhanced_spectral_analysis(audio_window)
```

### Étape 4 : Docker Compose

```yaml
# docker-compose.official.yml
services:
  magenta-service:
    build:
      context: .
      dockerfile: Dockerfile.magenta.official
    container_name: audio2dtx-magenta-official
    # Reste identique...
```

## Commandes de Test

```bash
# Build avec vraie image Magenta
docker build -f Dockerfile.magenta.official -t magenta-official .

# Test du service
docker run --rm -p 5000:5000 magenta-official

# Test santé
curl http://localhost:5000/health

# Test avec Docker Compose
docker-compose -f docker-compose.official.yml up -d
```

## Migration Complète

```bash
# 1. Backup actuel
cp docker-compose.yml docker-compose.backup.yml
cp Dockerfile.magenta Dockerfile.magenta.backup

# 2. Mise à jour
cp docker-compose.official.yml docker-compose.yml
cp Dockerfile.magenta.official Dockerfile.magenta

# 3. Test
make build-all
make test-magenta

# 4. Rollback si problème
cp docker-compose.backup.yml docker-compose.yml
cp Dockerfile.magenta.backup Dockerfile.magenta
```

## Avantages vs Inconvénients

| Critère | Actuel | Option A | Option B | Option C |
|---------|--------|----------|----------|----------|
| **Authenticité Magenta** | ❌ Simulation | ✅ Officiel | ✅ Officiel | ✅ Fonctionnel |
| **Python Version** | ✅ 3.9 | ❌ 2.7 | ✅ 3.9 | ✅ 3.x |
| **Facilité Migration** | - | ❌ Réécriture | ⚠️ Complexe | ✅ Simple |
| **Performance** | ✅ Rapide | ✅ Rapide | ✅ Optimisé | ❌ Lourd |
| **Maintenance** | ✅ Simple | ❌ Python 2.7 | ✅ Moderne | ✅ Moderne |

## Conclusion

**Pour une migration future** :
1. **Court terme** : Option C (xychelsea/magenta) pour test rapide
2. **Long terme** : Option B (multi-stage) pour production
3. **Éviter** : Option A (Python 2.7) sauf cas spécifique

La solution actuelle reste parfaitement fonctionnelle pour les besoins immédiats, avec possibilité de migration vers Magenta réel quand nécessaire.