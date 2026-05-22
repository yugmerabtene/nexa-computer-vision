# ÉNONCÉ DE PROJET

# Reconnaissance d’objets avec CNN et détection avec Faster R-CNN via image et webcam

## 1. Contexte du projet

Dans ce projet, vous allez développer une application de vision par ordinateur capable de reconnaître des objets à partir :

```text
d’images statiques
```

et éventuellement :

```text
d’un flux vidéo issu d’une webcam
```

L’objectif est de comprendre la différence entre :

```text
CNN
```

et :

```text
R-CNN / Fast R-CNN / Faster R-CNN
```

Un **CNN** permet principalement de faire de la **classification d’image**.

Il répond à la question :

```text
Quel est l’objet principal présent dans l’image ?
```

Exemple :

```text
Image d’une souris
↓
CNN
↓
Résultat : souris
```

Après entraînement, le CNN peut aussi être utilisé avec une webcam.

Exemple :

```text
Webcam
↓
Objet montré devant la caméra
↓
Capture d’une image
↓
CNN
↓
Résultat affiché : souris
```

Un modèle comme **Faster R-CNN** permet de faire de la **détection d’objets**.

Il répond à deux questions :

```text
Quel objet est présent ?
Où se trouve l’objet dans l’image ?
```

Exemple :

```text
Image contenant une souris et un clavier
↓
Faster R-CNN
↓
Objet détecté : souris
Boîte autour de la souris

Objet détecté : clavier
Boîte autour du clavier
```

---

# 2. Objectif général du projet

L’objectif est de créer une application Python complète permettant de :

```text
préparer un dataset d’images
entraîner un CNN sur plusieurs classes d’objets
tester le CNN sur des images
utiliser le CNN avec une webcam
afficher en temps réel le nom de l’objet reconnu
utiliser Faster R-CNN pour détecter et localiser des objets
comparer classification et détection
expliquer les résultats obtenus
```

Le projet doit permettre de comprendre clairement la différence entre :

```text
reconnaître un objet
```

et :

```text
reconnaître un objet et le localiser
```

---

# 3. Résultat attendu

À la fin du projet, l’application devra permettre deux usages principaux.

## 3.1. Reconnaissance avec CNN

Le CNN devra être capable de reconnaître la classe principale d’une image.

Exemple :

```text
Image testée : souris.jpg
↓
CNN
↓
Classe prédite : souris
Confiance : 94 %
```

Le CNN pourra aussi être utilisé avec la webcam.

Exemple :

```text
L’utilisateur montre une souris à la caméra
↓
Le programme capture l’image
↓
Le CNN analyse l’image
↓
Le programme affiche : souris
```

Dans ce cas, le CNN ne dessine pas obligatoirement une boîte autour de l’objet.
Il donne surtout le **nom de l’objet principal**.

---

## 3.2. Détection avec Faster R-CNN

Faster R-CNN devra être capable de détecter des objets dans une image.

Il devra produire :

```text
le nom de l’objet détecté
le score de confiance
les coordonnées de la boîte
une image annotée avec les boîtes visibles
```

Exemple :

```text
Image contenant un clavier
↓
Faster R-CNN
↓
Objet détecté : keyboard
Score : 0.91
Boîte : x1=45, y1=80, x2=320, y2=210
```

Faster R-CNN peut aussi être utilisé avec la webcam, mais il est plus lourd qu’un CNN simple.

---

# 4. Sujet du projet

Vous devez développer une application Python permettant de comparer deux approches :

```text
CNN pour reconnaître un objet
```

et :

```text
Faster R-CNN pour reconnaître et localiser un objet
```

Le projet devra montrer clairement que :

```text
CNN = classification
Faster R-CNN = détection
```

Le projet devra aussi montrer qu’un modèle entraîné peut être utilisé avec une webcam pour faire de la reconnaissance en temps réel.

---

# 5. Fonctionnement global attendu

Le fonctionnement général du projet sera le suivant :

```text
Dataset d’images
↓
Préparation des données
↓
Entraînement du CNN
↓
Évaluation du CNN
↓
Test sur une image
↓
Test avec la webcam
↓
Détection avec Faster R-CNN
↓
Comparaison des résultats
```

---

# 6. Dataset attendu

Vous devez constituer un dataset d’images avec au moins **3 classes d’objets**.

Exemple :

```text
clavier
souris
écran
```

Autre exemple :

```text
stylo
cahier
téléphone
```

Autre exemple :

```text
bouteille
tasse
livre
```

L’important est que chaque classe corresponde à un objet visible et reconnaissable.

---

## 6.1. Organisation du dataset brut

Le dataset devra être organisé ainsi :

```text
dataset_raw/
├── clavier/
├── souris/
└── ecran/
```

Chaque dossier représente une classe.

Exemple :

```text
dataset_raw/
├── clavier/
│   ├── clavier_001.jpg
│   ├── clavier_002.jpg
│   └── clavier_003.jpg
│
├── souris/
│   ├── souris_001.jpg
│   ├── souris_002.jpg
│   └── souris_003.jpg
│
└── ecran/
    ├── ecran_001.jpg
    ├── ecran_002.jpg
    └── ecran_003.jpg
```

---

## 6.2. Nombre d’images attendu

Minimum conseillé :

```text
30 images par classe
```

Recommandé :

```text
80 à 150 images par classe
```

Les images doivent être variées :

```text
angles différents
luminosités différentes
arrière-plans différents
distances différentes
positions différentes de l’objet
```

Cette diversité est importante, car le modèle devra ensuite reconnaître les objets dans des conditions proches du réel, par exemple avec une webcam.

---

## 6.3. Qualité des images

Images acceptées :

```text
image nette
objet bien visible
classe correcte
format jpg, jpeg, png ou webp
image non corrompue
```

Images à éviter :

```text
image floue
objet coupé
objet trop petit
image trop sombre
mauvais dossier de classe
doublon exact
image impossible à ouvrir
image sans rapport avec le sujet
```

---

# 7. Arborescence attendue du projet

Le projet devra respecter une organisation claire.

```text
projet-cnn-faster-rcnn/
├── dataset_raw/
│   ├── classe_1/
│   ├── classe_2/
│   └── classe_3/
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   └── best_cnn.pth
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── cnn_predictions.png
│   ├── faster_rcnn_detection.png
│   ├── webcam_cnn_capture.png
│   └── metrics.json
│
├── src/
│   ├── prepare_dataset.py
│   ├── train_cnn.py
│   ├── evaluate_cnn.py
│   ├── predict_cnn.py
│   ├── webcam_cnn.py
│   ├── detect_faster_rcnn.py
│   ├── webcam_faster_rcnn.py
│   └── metrics.py
│
├── requirements.txt
├── README.md
└── rapport.md
```

---

# 8. Travail demandé

## Étape 1 — Préparer le dataset

Vous devez créer le script :

```text
src/prepare_dataset.py
```

Ce script doit :

```text
lire les images depuis dataset_raw/
vérifier que les images sont valides
ignorer les images non lisibles
créer automatiquement le dossier dataset/
séparer les images en train, validation et test
conserver les classes à partir des noms de dossiers
```

Répartition attendue :

```text
70 % pour l’entraînement
15 % pour la validation
15 % pour le test
```

Résultat attendu :

```text
dataset/
├── train/
│   ├── clavier/
│   ├── souris/
│   └── ecran/
│
├── val/
│   ├── clavier/
│   ├── souris/
│   └── ecran/
│
└── test/
    ├── clavier/
    ├── souris/
    └── ecran/
```

---

## Étape 2 — Entraîner un CNN

Vous devez créer le script :

```text
src/train_cnn.py
```

Ce script doit entraîner un CNN capable de classifier les images.

Le CNN devra contenir au minimum :

```text
une couche de convolution
une activation ReLU
une couche MaxPooling
une deuxième couche de convolution
une deuxième activation ReLU
une deuxième couche MaxPooling
une couche Flatten
une couche Fully Connected
une couche de sortie
```

Pipeline attendu :

```text
Image
↓
Redimensionnement
↓
Transformation en tenseur
↓
Normalisation
↓
Convolution
↓
ReLU
↓
MaxPooling
↓
Convolution
↓
ReLU
↓
MaxPooling
↓
Flatten
↓
Fully Connected
↓
Classe prédite
```

Le modèle entraîné devra être sauvegardé ici :

```text
models/best_cnn.pth
```

---

## Étape 3 — Évaluer le CNN

Vous devez créer le script :

```text
src/evaluate_cnn.py
```

Ce script doit évaluer le CNN sur les données de test.

Il doit produire :

```text
accuracy sur le train
accuracy sur la validation
accuracy sur le test
loss d’entraînement
loss de validation
matrice de confusion
exemples de bonnes prédictions
exemples de mauvaises prédictions
```

Fichiers attendus :

```text
outputs/metrics.json
outputs/confusion_matrix.png
outputs/cnn_predictions.png
```

---

## Étape 4 — Tester une image avec le CNN

Vous devez créer le script :

```text
src/predict_cnn.py
```

Ce script doit permettre de tester une seule image.

Exemple d’utilisation :

```bash
python src/predict_cnn.py --image images/test_souris.jpg
```

Résultat attendu :

```text
Image analysée : images/test_souris.jpg
Classe prédite : souris
Confiance : 94 %
```

---

## Étape 5 — Utiliser le CNN avec une webcam

Vous devez créer le script :

```text
src/webcam_cnn.py
```

Ce script doit permettre d’utiliser le modèle entraîné avec une webcam.

Fonctionnement attendu :

```text
ouverture de la webcam
capture du flux vidéo
prétraitement de chaque image
envoi de l’image dans le CNN
récupération de la classe prédite
récupération du score de confiance
affichage du résultat sur l’image
affichage du flux vidéo annoté
```

Exemple attendu à l’écran :

```text
Objet reconnu : souris
Confiance : 94 %
```

Pipeline attendu :

```text
Webcam
↓
Image capturée
↓
Redimensionnement
↓
Normalisation
↓
CNN entraîné
↓
Classe prédite
↓
Affichage du nom de l’objet
```

Exemple concret :

```text
L’utilisateur montre un clavier devant la webcam
↓
Le modèle analyse l’image
↓
Le programme affiche : clavier
```

Attention : le CNN reconnaît uniquement les classes apprises pendant l’entraînement.

Exemple :

```text
Si le modèle a été entraîné sur clavier, souris et écran,
il ne pourra pas reconnaître correctement une bouteille.
```

---

## Étape 6 — Utiliser Faster R-CNN sur une image

Vous devez créer le script :

```text
src/detect_faster_rcnn.py
```

Ce script doit utiliser un modèle Faster R-CNN pré-entraîné.

Il doit :

```text
charger une image
charger un modèle Faster R-CNN
prétraiter l’image
envoyer l’image dans le modèle
récupérer les objets détectés
récupérer les scores de confiance
récupérer les coordonnées des boîtes
filtrer les détections faibles
dessiner les boîtes sur l’image
sauvegarder l’image annotée
```

Sortie attendue :

```text
outputs/faster_rcnn_detection.png
```

Exemple :

```text
Image contenant une souris et un clavier
↓
Faster R-CNN
↓
Détection 1 : mouse, score 0.89
Détection 2 : keyboard, score 0.84
↓
Image sauvegardée avec les boîtes
```

---

## Étape 7 — Utiliser Faster R-CNN avec la webcam

Vous pouvez créer le script :

```text
src/webcam_faster_rcnn.py
```

Ce script doit appliquer Faster R-CNN sur le flux webcam.

Fonctionnement attendu :

```text
ouverture de la webcam
capture d’une image
envoi de l’image dans Faster R-CNN
récupération des boîtes
récupération des classes
récupération des scores
affichage des boîtes sur le flux vidéo
```

Exemple à l’écran :

```text
mouse : 0.87
keyboard : 0.82
```

Avec Faster R-CNN, l’image affichée devra montrer :

```text
une boîte autour de chaque objet détecté
le nom de l’objet
le score de confiance
```

Attention : Faster R-CNN est plus lourd qu’un CNN simple.
Le flux vidéo peut donc être plus lent selon la puissance de l’ordinateur.

---

# 9. Rappels techniques à comprendre

## 9.1. CNN

Un CNN est un réseau de neurones spécialisé dans le traitement des images.

Il apprend automatiquement des caractéristiques visuelles.

Exemples :

```text
bords
formes
textures
motifs
parties d’objets
```

Il est utilisé ici pour reconnaître l’objet principal dans une image.

---

## 9.2. Convolution

La convolution sert à extraire des informations visuelles dans une image.

Exemple :

```text
bord vertical
bord horizontal
forme ronde
texture particulière
```

Plus on avance dans le réseau, plus les formes détectées deviennent complexes.

---

## 9.3. ReLU

ReLU est une fonction d’activation.

Formule :

```text
ReLU(x) = max(0, x)
```

Lecture :

```text
si x est négatif, le résultat vaut 0
si x est positif, le résultat vaut x
```

Elle permet au réseau d’apprendre des relations plus complexes.

---

## 9.4. MaxPooling

MaxPooling réduit la taille des cartes de caractéristiques.

Il permet de garder les informations importantes tout en réduisant les calculs.

Exemple :

```text
grande carte de caractéristiques
↓
MaxPooling
↓
carte plus petite
```

---

## 9.5. Flatten

Flatten transforme les cartes de caractéristiques en un vecteur.

Avant Flatten :

```text
plusieurs cartes en deux dimensions
```

Après Flatten :

```text
un vecteur de nombres
```

Ce vecteur est ensuite envoyé aux couches Fully Connected.

---

## 9.6. Fully Connected

Les couches Fully Connected servent à prendre la décision finale.

Elles utilisent les caractéristiques extraites par les convolutions pour prédire la classe.

Exemple :

```text
caractéristiques extraites
↓
Fully Connected
↓
probabilités par classe
↓
classe finale
```

---

# 10. Fonctionnement attendu de Faster R-CNN

Faster R-CNN fonctionne en plusieurs étapes.

```text
Image
↓
Backbone CNN
↓
Extraction de caractéristiques
↓
RPN
↓
Propositions de régions
↓
Classification des régions
↓
Correction des boîtes
↓
Résultat final
```

## 10.1. Backbone CNN

Le backbone CNN extrait les caractéristiques visuelles de l’image.

Il produit des cartes de caractéristiques contenant des informations sur :

```text
les contours
les formes
les textures
les parties d’objets
```

---

## 10.2. RPN

RPN signifie :

```text
Region Proposal Network
```

Son rôle est de proposer les zones de l’image où il pourrait y avoir un objet.

Exemple :

```text
zone 1 : objet possible
zone 2 : objet possible
zone 3 : arrière-plan
```

---

## 10.3. Tête de détection

La tête de détection analyse les régions proposées par le RPN.

Elle décide :

```text
quelle est la classe de l’objet
où se trouve l’objet
quel est le score de confiance
```

---

# 11. Comparaison attendue

## 11.1. CNN

Le CNN permet de classifier une image.

Exemple :

```text
Image
↓
CNN
↓
Classe : souris
```

Limite :

```text
il ne localise pas précisément l’objet
```

Avec la webcam :

```text
Webcam
↓
CNN
↓
Affichage : souris
```

Le CNN fonctionne bien si l’objet principal est bien visible dans l’image.

---

## 11.2. R-CNN

R-CNN est une ancienne méthode de détection.

Principe :

```text
proposer plusieurs régions dans l’image
analyser chaque région avec un CNN
classifier chaque région
```

Limite :

```text
R-CNN est lent car il applique le CNN plusieurs fois
```

---

## 11.3. Fast R-CNN

Fast R-CNN améliore R-CNN.

Principe :

```text
l’image passe une seule fois dans le CNN
les régions sont ensuite analysées à partir des mêmes caractéristiques
```

Avantage :

```text
plus rapide que R-CNN
```

Limite :

```text
les régions candidates viennent encore d’une méthode externe
```

---

## 11.4. Faster R-CNN

Faster R-CNN améliore Fast R-CNN.

Principe :

```text
le modèle apprend lui-même à proposer les régions intéressantes grâce au RPN
```

Avantage :

```text
il reconnaît les objets
il localise les objets
il peut détecter plusieurs objets dans une même image
```

Limite :

```text
il est plus lourd qu’un CNN simple
il peut être plus lent en temps réel
```

---

# 12. Tableau comparatif à intégrer dans le rapport

| Critère               | CNN                            | Faster R-CNN                           |
| --------------------- | ------------------------------ | -------------------------------------- |
| Tâche principale      | Classification                 | Détection                              |
| Question traitée      | Qu’est-ce que c’est ?          | Qu’est-ce que c’est et où ?            |
| Sortie                | Une classe globale             | Boîtes, classes, scores                |
| Localisation          | Non                            | Oui                                    |
| Plusieurs objets      | Non, pas directement           | Oui                                    |
| Usage webcam          | Oui, reconnaissance simple     | Oui, détection avec boîtes             |
| Vitesse               | Plus rapide                    | Plus lent                              |
| Données nécessaires   | Images classées                | Images annotées ou modèle pré-entraîné |
| Métriques principales | Accuracy, loss                 | IoU, précision, rappel                 |
| Usage typique         | Reconnaître un objet principal | Localiser plusieurs objets             |

---

# 13. Métriques à expliquer

## 13.1. Accuracy

L’accuracy mesure le pourcentage de bonnes prédictions.

Formule :

```text
accuracy = nombre de bonnes prédictions / nombre total de prédictions
```

Exemple :

```text
90 bonnes prédictions sur 100 images
↓
accuracy = 90 %
```

---

## 13.2. Loss

La loss mesure l’erreur du modèle pendant l’entraînement.

```text
loss élevée = modèle mauvais
loss faible = modèle meilleur
```

Attention :

```text
une loss très faible sur le train mais une mauvaise performance sur le test peut indiquer du surapprentissage
```

---

## 13.3. Matrice de confusion

La matrice de confusion permet de voir quelles classes sont bien reconnues et quelles classes sont confondues.

Exemple :

```text
le modèle confond souvent souris et clavier
le modèle reconnaît très bien les écrans
```

---

## 13.4. IoU

IoU signifie :

```text
Intersection over Union
```

Elle mesure la qualité d’une boîte prédite.

Formule :

```text
IoU = aire de l’intersection / aire de l’union
```

Lecture :

```text
IoU proche de 1 : la boîte est très bonne
IoU proche de 0 : la boîte est mauvaise
```

---

## 13.5. Score de confiance

Le score de confiance indique à quel point le modèle est sûr de sa prédiction.

Exemple :

```text
souris : 0.94
```

Cela signifie :

```text
le modèle estime fortement que l’objet détecté est une souris
```

Un seuil de confiance permet de filtrer les mauvaises détections.

Exemple :

```text
seuil = 0.70
```

Cela signifie :

```text
on garde uniquement les détections avec un score supérieur ou égal à 0.70
```

---

# 14. Questions obligatoires dans le rapport

Le rapport devra répondre aux questions suivantes :

```text
1. Quelle est la différence entre classification, détection et reconnaissance ?

2. À quoi sert un CNN ?

3. À quoi sert une convolution ?

4. À quoi sert ReLU ?

5. À quoi sert MaxPooling ?

6. À quoi sert Flatten ?

7. À quoi servent les couches Fully Connected ?

8. Pourquoi séparer le dataset en train, validation et test ?

9. Qu’est-ce que le surapprentissage ?

10. Pourquoi un CNN simple ne localise-t-il pas précisément un objet ?

11. Peut-on utiliser un CNN entraîné avec une webcam ?

12. Que se passe-t-il si on montre à la webcam un objet non présent dans le dataset d’entraînement ?

13. Quelle est la différence entre R-CNN, Fast R-CNN et Faster R-CNN ?

14. À quoi sert le RPN dans Faster R-CNN ?

15. Qu’est-ce qu’une bounding box ?

16. Qu’est-ce que l’IoU ?

17. Pourquoi utilise-t-on un seuil de confiance ?

18. Que se passe-t-il si le seuil de confiance est trop bas ?

19. Que se passe-t-il si le seuil de confiance est trop haut ?

20. Dans quel cas utiliser un CNN ?

21. Dans quel cas utiliser Faster R-CNN ?

22. Quelles sont les limites de votre projet ?
```

---

# 15. Livrables attendus

Le rendu devra contenir :

```text
code source complet
dataset utilisé ou lien vers le dataset
modèle CNN entraîné
résultats d’évaluation
matrice de confusion
script de prédiction sur image
script de reconnaissance avec webcam
images annotées avec Faster R-CNN
README.md
rapport.md
présentation orale courte
```

---

# 16. Contenu attendu du README.md

Le fichier `README.md` devra contenir :

```text
titre du projet
objectif du projet
description du dataset
structure du projet
installation
commandes d’exécution
préparation du dataset
entraînement du CNN
évaluation du CNN
test d’une image avec le CNN
test avec la webcam
détection avec Faster R-CNN
résultats obtenus
limites du projet
pistes d’amélioration
```

Exemple de commandes :

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python src/prepare_dataset.py
python src/train_cnn.py
python src/evaluate_cnn.py
python src/predict_cnn.py --image images/test.jpg
python src/webcam_cnn.py
python src/detect_faster_rcnn.py --image images/test.jpg
python src/webcam_faster_rcnn.py
```

Sous Windows :

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python src\prepare_dataset.py
python src\train_cnn.py
python src\evaluate_cnn.py
python src\predict_cnn.py --image images\test.jpg
python src\webcam_cnn.py
python src\detect_faster_rcnn.py --image images\test.jpg
python src\webcam_faster_rcnn.py
```

---

# 17. Contenu attendu du rapport.md

Le fichier `rapport.md` devra contenir :

```text
introduction
présentation du dataset
explication du CNN
architecture du CNN utilisé
résultats de classification
matrice de confusion
test sur image
test avec webcam
analyse des erreurs
explication de Faster R-CNN
résultats de détection
comparaison CNN et Faster R-CNN
limites rencontrées
améliorations possibles
conclusion
```

---

# 18. Contraintes techniques

Le projet doit être réalisé en Python.

Bibliothèques attendues :

```text
torch
torchvision
opencv-python
pillow
numpy
matplotlib
scikit-learn
```

Le fichier `requirements.txt` devra contenir les dépendances nécessaires.

Exemple :

```text
torch
torchvision
opencv-python
pillow
numpy
matplotlib
scikit-learn
```

---

# 19. Démonstration attendue

Lors de la démonstration, vous devrez montrer :

```text
la structure du projet
le dataset utilisé
l’entraînement du CNN
les résultats du CNN
une prédiction sur une image
une reconnaissance avec la webcam
une détection avec Faster R-CNN
une image annotée avec des boîtes
une comparaison claire entre CNN et Faster R-CNN
```

Exemple de démonstration :

```text
Image testée : test_souris.jpg

Résultat CNN :
classe prédite : souris
confiance : 94 %

Résultat webcam :
objet montré à la caméra : souris
résultat affiché : souris
confiance : 91 %

Résultat Faster R-CNN :
objet détecté : mouse
score : 0.88
boîte détectée autour de la souris

Analyse :
Le CNN reconnaît l’objet principal.
La webcam permet d’utiliser le CNN en temps réel.
Faster R-CNN reconnaît l’objet et indique sa position.
```

---

# 20. Conclusion attendue

Le projet doit montrer qu’un CNN permet de reconnaître un objet après entraînement.

Une fois entraîné, le CNN peut être utilisé sur :

```text
une image statique
```

ou :

```text
un flux webcam
```

Le modèle peut donc afficher en direct le nom de l’objet montré devant la caméra.

Cependant, un CNN simple ne localise pas précisément l’objet.

Pour localiser les objets, il faut utiliser une approche de détection comme :

```text
R-CNN
Fast R-CNN
Faster R-CNN
YOLO
SSD
```

Dans ce projet, la comparaison principale portera sur :

```text
CNN pour reconnaître un objet
```

et :

```text
Faster R-CNN pour reconnaître et localiser un objet
```

Le rendu final doit permettre de comprendre clairement la différence entre :

```text
dire ce qu’il y a dans l’image
```

et :

```text
dire ce qu’il y a dans l’image et où cela se trouve
```
