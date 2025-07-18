# Bugs à corriger

- les 2 seules tracks qui semblent donner un résultat à peu près satisfaisant sont : magenta et advanced feature. think et verifie que les autres tracks fonctionnent              │
  │   correctement :                                                                                                                                                                   │
  │   - pas de message d'erreur                                                                                                                                                        │
  │   - pas de simulation, mais traitement réel                                                                                                                                        │
  │   - pas de calcul de poids qui viendrait perturber le résultat                                                                                                                     │
  │   - cherche tout ce qui pourrait nuire au bon fonctionnement de chaque track                                                                                                       │
  │   assure-toi de bien lancer les tracks individuellement, pour ne pas que l'une ai une influence sur une autre
- /add-dir /Users/simon/Dropbox/Documents/DTXMania/dtx/
je dispose de tout un tas d'audio dont j'ai réalisé la conversion dtx moi même avec dtxcreator. est-ce que celà peut être utilise pour améliorer la qualité de la conversion,      │
  │   entrainer un modèle ou établir une batterie de tests de qualité ? ultrathink et mets à jour le plan dans tracks.md pour prendre en compte ces données.
- simplifier la génération du bgm, ffmpeg était suffisant
- nettoyer le dossier output/temp après la conversion
- implémenter une sorte de magnétisme pour que les notes soient calées sur la grille. si besoin déplacer légèrement le déclenchement du bgm => fonctionne pas

# Idées pour de nouvelles fonctionalités

- Proposer de couper l'audio (BMG) pour faire tomber la première note de batterie avec le début d'une mesure
- Améliorer la détection du tempo / création de la tempo map
- Chercher sur le web fichier midi qui pourraient servir de base pour la conversion
- Améliorer l'architecture du code
- Réfléchir à une méthode pour tester l'efficacité de la conversion
- Tout réécrire en go
- Est-ce réeelement utilse de simuler un output de magenta s'il n'est pas dispo ?
- Est-ce utile de choisir les tracks en fonction du genre ? peut-être faut-il un mode où on les joue tous ?
