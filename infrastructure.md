# Infrastructure Audio2DTX

## Vue d'ensemble

Cette documentation décrit l'infrastructure Docker mise à jour pour Audio2DTX après la phase de refactoring. L'infrastructure est basée sur **docker-compose uniquement** et élimine le mode standalone Docker.

## Architecture des Services

### 🎵 Service Principal (audio2dtx-main)
- **Image**: Construite depuis `Dockerfile`
- **Rôle**: Traitement audio principal avec nouvelle architecture modulaire
- **Accès**: Interface CLI pour toutes les opérations
- **Volumes**: 
  - `./input:/app/input` - Fichiers audio d'entrée
  - `./output:/app/output` - Charts DTX générés

### 🎯 Service Magenta (magenta-service)
- **Image**: Construite depuis `Dockerfile.magenta`
- **Rôle**: Microservice de classification de batterie
- **API**: REST API sur port 5000 (interne)
- **Health Check**: Endpoint `/health` avec surveillance automatique

## Commandes Disponibles

### 🔧 Commandes de Base

```bash
# Construire tous les services
make build

# Exécuter avec arguments personnalisés
make run ARGS="song.mp3 --use-rock-ultimate --batch"

# Exécuter en mode interactif
make run-interactive

# Tester avec configuration par défaut
make test
```

### 🚀 Gestion des Services

```bash
# Démarrer tous les services en arrière-plan
make start

# Arrêter tous les services
make stop

# Voir les logs de tous les services
make logs

# Voir les logs du service principal uniquement
make logs-main

# Voir les logs du service magenta uniquement
make logs-magenta
```

### 🔍 Monitoring et Maintenance

```bash
# Vérifier la santé du service magenta
make health

# Nettoyer complètement l'environnement
make clean

# Afficher l'aide
make help
```

## Exemples d'Utilisation

### Configuration Basique
```bash
# Processing simple avec mode batch
make run ARGS="song.mp3 --batch"

# Processing avec titre personnalisé
make run ARGS="song.mp3 --title 'Ma Chanson' --batch"
```

### Tracks Avancés
```bash
# Track 9: Ultimate Rock/Metal
make run ARGS="metal_song.mp3 --use-rock-ultimate --batch"

# Track 7: Ensemble de modèles spécialisés
make run ARGS="complex_song.mp3 --use-ensemble --batch"

# Track 4: Features avancées
make run ARGS="song.mp3 --use-advanced-features --batch"
```

### Mode Développement
```bash
# Démarrer les services en arrière-plan
make start

# Traiter plusieurs fichiers
make run ARGS="song1.mp3 --batch"
make run ARGS="song2.mp3 --use-multi-scale --batch"
make run ARGS="song3.mp3 --use-rock-ultimate --batch"

# Monitoring
make logs-main
make health

# Arrêter les services
make stop
```

## Structure des Fichiers

```
audio2dtx/
├── Dockerfile                 # Image principale (nouvelle architecture)
├── Dockerfile.magenta        # Image service magenta
├── docker-compose.yml        # Configuration services
├── Makefile                  # Commandes simplifiées
├── requirements.txt          # Dépendances Python principales
├── requirements-magenta.txt  # Dépendances service magenta
├── main.py                   # Point d'entrée principal
├── src/                      # Code source modulaire
│   └── audio2dtx/
│       ├── core/             # Pipeline principal
│       ├── classification/   # Système de classification
│       ├── audio/            # Traitement audio
│       ├── dtx/              # Export DTX
│       ├── services/         # Services externes
│       ├── config/           # Configuration
│       └── utils/            # Utilitaires
├── input/                    # Fichiers audio d'entrée
├── output/                   # Charts DTX générés
└── PredictOnset.h5          # Modèle ML pré-entraîné
```

## Changements par Rapport à l'Ancienne Version

### ✅ Améliorations
- **Dockerfile optimisé** : Suppression des références legacy, meilleur cache layer
- **Makefile simplifié** : Suppression du mode standalone, commandes plus intuitives
- **docker-compose optimisé** : Configuration plus simple et robuste
- **Architecture modulaire** : Code source dans `src/` au lieu de fichiers individuels

### ❌ Suppressions
- **Mode standalone Docker** : Plus de `docker run` direct
- **Commandes legacy** : Suppression des anciens noms de commandes
- **Fichiers obsolètes** : `audio_to_chart.py` et autres fichiers monolithiques

### 🔄 Migrations
- **Ancien** : `make run` (standalone) → **Nouveau** : `make run ARGS="..."`
- **Ancien** : `make run-magenta` → **Nouveau** : `make run ARGS="..."`
- **Ancien** : `make build && make run` → **Nouveau** : `make build && make run ARGS="..."`

## Dépendances

### Service Principal
- Python 3.10
- FFmpeg (traitement audio)
- TensorFlow 2.12.1 (ML)
- Librosa 0.8.1 (analyse audio)
- Spleeter 2.4.2 (séparation sources)

### Service Magenta
- Python 3.9
- Flask 2.3.3 (API REST)
- Librosa 0.10.0 (analyse audio)
- NumPy, SciPy (calculs scientifiques)

## Monitoring et Debugging

### Health Checks
```bash
# Vérifier l'état du service magenta
make health

# Vérifier l'état de tous les services
docker-compose ps
```

### Logs
```bash
# Logs détaillés avec suivi temps réel
make logs

# Logs d'un service spécifique
make logs-main
make logs-magenta

# Logs avec filtre
docker-compose logs -f --tail=100 audio2dtx-main
```

### Debugging
```bash
# Connexion interactive au container principal
docker-compose exec audio2dtx-main bash

# Connexion interactive au service magenta
docker-compose exec magenta-service bash

# Vérifier les variables d'environnement
docker-compose exec audio2dtx-main env
```

## Performance et Optimisations

### Build Optimisé
- **Cache Docker** : Requirements installés avant copie du code
- **Layers minimisés** : Commandes combinées pour réduire la taille
- **Nettoyage automatique** : Suppression des paquets de build inutiles

### Runtime Optimisé
- **Health checks** : Surveillance automatique des services
- **Restart policies** : Redémarrage automatique en cas d'erreur
- **Volumes optimisés** : Montage direct des dossiers input/output

## Sécurité

### Réseau
- **Réseau isolé** : Services communiquent via `audio2dtx-network`
- **Ports non exposés** : Service magenta accessible uniquement en interne
- **Variables d'environnement** : Configuration sécurisée des services

### Conteneurs
- **Images minimal** : Python slim pour réduire la surface d'attaque
- **Utilisateur non-root** : Exécution avec privilèges limités
- **Volumes read-only** : Montage en lecture seule quand possible

## Maintenance

### Mise à jour
```bash
# Reconstruction complète
make clean
make build

# Mise à jour des dépendances
docker-compose build --no-cache
```

### Nettoyage
```bash
# Nettoyage standard
make clean

# Nettoyage complet (images, volumes, networks)
docker system prune -a --volumes
```

## Support et Dépannage

### Problèmes Courants

**Le service ne démarre pas** :
```bash
# Vérifier les logs
make logs

# Reconstruire l'image
make build
```

**Erreur de connexion Magenta** :
```bash
# Vérifier l'état du service
make health

# Redémarrer les services
make stop && make start
```

**Problème de volumes** :
```bash
# Vérifier les permissions
ls -la input/ output/

# Nettoyer et recréer
make clean
make build
```

### Contact
Pour les problèmes spécifiques à l'infrastructure, consultez :
- **CLAUDE.md** : Documentation complète du projet
- **Docker logs** : `make logs` pour diagnostic
- **Health checks** : `make health` pour vérifier l'état des services