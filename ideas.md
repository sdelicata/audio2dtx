# Bugs à corriger

- simplifier la génération du bgm, ffmpeg était suffisant
- nettoyer le dossier output/tmp après la conversion
- corriger la track ML
- implémenter une sorte de magnétisme pour que les notes soient calées sur la grille. si besoin déplacer légèrement le déclenchement du bgm

# Idées pour de nouvelles fonctionalités

- Proposer de couper l'audio (BMG) pour faire tomber la première note de batterie avec le début d'une mesure
- Améliorer la détection du tempo / création de la tempo map
- Chercher sur le web fichier midi qui pourraient servir de base pour la conversion
- Améliorer l'architecture du code
- Réfléchir à une méthode pour tester l'efficacité de la conversion
- Tout réécrire en go
- Est-ce réeelement utilse de simuler un output de magenta s'il n'est pas dispo ?
- Est-ce utile de choisir les tracks en fonction du genre ? peut-être faut-il un mode où on les joue tous ?
