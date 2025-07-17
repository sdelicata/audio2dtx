l'utilisation de magenta améliore un peu la qualité de la conversion, mais on est encore loin du compte.
les instruments ne sont pas reconnu correctement. j'obtiens par exemple des toms là où il n'y en a pas du tout, confusion entre snare et kick, etc.

ultrathink pour améliorer le systeme de détection des instruments et écrire un plan dans le fichier `improve.md` 

voici des idées à creuser, mais cherches-en aussi d'autres par toi même :
- est-ce qu'utiliser l'audio original plutot que le stem de drum généré avec splitter permettrait d'avoir une meilleure conversion ?
- cherche d'autres models plus performants pour faire cette conversion
- est-ce que le systeme de pondération est le meilleur moyen d'obtenir un résultat fiable ? peut-être vaudrait-il mieux se concentrer sur une seule méthode ? ou revoir la pondération ?

plan :
- propose moi plusieurs pistes d'amélioration que je pourrai valider ou pas
- pour chacune de celles que j'aurai validé, je veux que tu :
  - implémente la solution
  - lance une conversion en renseignant l'argument --title avec un titre specifique à la solution testée pour que je puisse juger par moi même et comparer les différents résultat dans dtxmania

pré-requis :
- utilise tous les moyens à ta disposition pour améliorer la qualité de la conversion
- assure-toi que le build des images docker fonctionne bien
- assure-toi que la méthode à tester est bien utilisée (pas d'erreur ou de fallback dans les logs par ex)