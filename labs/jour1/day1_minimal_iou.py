"""
Version minimale du Jour 1 — Calcul d'IoU de bout en bout
===========================================================

Objectif pédagogique :
    Ce script volontairement court isole la compétence centrale du Jour 1 :
    localiser un objet par seuillage, extraire sa boîte englobante, puis
    mesurer la qualité de cette localisation avec l'Intersection over Union
    (IoU).

Pourquoi une version minimale ?
    - Pour les étudiants qui veulent se concentrer UNIQUEMENT sur l'IoU sans
      la complexité de HOG, SIFT et des figures.
    - Pour servir de base de débogage : si l'IoU ne fonctionne pas ici,
      le problème vient du calcul ou du seuillage, pas du reste du pipeline.
    - Pour montrer que 30 lignes de code suffisent à faire de la détection
      basique.

Compétences travaillées :
    - Seuillage binaire avec cv2.threshold.
    - Extraction de boîte englobante avec cv2.boundingRect.
    - Calcul d'IoU entre deux boîtes.
    - Sauvegarde de métriques au format JSON.
"""

import json
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# make_synthetic_scene
# ---------------------------------------------------------------------------
def make_synthetic_scene(shape: str, shift: int = 0) -> np.ndarray:
    """Construit une scène simple avec une forme blanche sur fond noir.

    Le contraste maximal (objet blanc 255, fond noir 0) garantit qu'un
    simple seuillage binaire à 127 sépare parfaitement l'objet du fond.
    L'étudiant n'a donc pas à se soucier du bruit ou de l'éclairage, et
    peut se concentrer sur la logique de localisation et d'IoU.

    Paramètres
    ----------
    shape : str
        "rectangle" ou "circle". Le cercle est fourni pour une éventuelle
        extension, mais le script minimal utilise surtout le rectangle.
    shift : int
        Décalage horizontal en pixels. Un shift non nul simule une
        prédiction imparfaite, ce qui fait baisser l'IoU.

    Retourne
    --------
    np.ndarray (256, 256, 3) en uint8
        Image BGR avec fond noir et forme blanche.

    Exemple
    -------
    >>> make_synthetic_scene("rectangle", 0)   # GT parfaite
    >>> make_synthetic_scene("rectangle", 12)  # prédiction décalée
    """
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    if shape == "rectangle":
        # Rectangle de 140px de large, 130px de haut.
        # 'shift' décale horizontalement la position de la forme.
        cv2.rectangle(img, (40 + shift, 60), (180 + shift, 190), (255, 255, 255), -1)

    elif shape == "circle":
        # Cercle de rayon 60, centré en (120, 130).
        cv2.circle(img, (120 + shift, 130), 60, (255, 255, 255), -1)

    else:
        raise ValueError("shape must be 'rectangle' or 'circle'")

    return img


# ---------------------------------------------------------------------------
# iou  (Intersection over Union)
# ---------------------------------------------------------------------------
def iou(box_a, box_b):
    """Calcule l'Intersection over Union entre deux boîtes.

    Paramètres
    ----------
    box_a, box_b : tuple (x1, y1, x2, y2)
        Coordonnées des deux boîtes.

    Retourne
    --------
    float
        IoU entre 0 (aucun recouvrement) et 1 (superposition parfaite).

    Rappel du calcul :
        1. Coin haut-gauche de l'intersection = max(x1_a, x1_b), max(y1_a, y1_b)
        2. Coin bas-droit de l'intersection = min(x2_a, x2_b), min(y2_a, y2_b)
        3. Si les coordonnées sont inversées => pas d'intersection => IoU = 0
        4. Aire d'intersection = (x_right - x_left) * (y_bottom - y_top)
        5. Aire d'union = aire(A) + aire(B) - intersection
        6. IoU = intersection / union
    """
    # Coin haut-gauche de l'intersection : on prend le max des deux.
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])

    # Coin bas-droit de l'intersection : on prend le min des deux.
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    # Si x_right <= x_left, les boîtes ne se chevauchent pas horizontalement.
    # Si y_bottom <= y_top, elles ne se chevauchent pas verticalement.
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    # Aire de l'intersection
    inter = (x_right - x_left) * (y_bottom - y_top)

    # Aires individuelles des deux boîtes
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    # Union = somme des aires - intersection (pour ne pas compter deux fois
    # la zone commune).
    return inter / (area_a + area_b - inter)


# ---------------------------------------------------------------------------
# bbox_from_threshold
# ---------------------------------------------------------------------------
def bbox_from_threshold(gray):
    """Extrait la boîte englobante d'un objet par seuillage binaire.

    Principe :
        1. On seuille l'image en niveaux de gris (seuil = 127).
        2. On récupère les coordonnées des pixels blancs (objet).
        3. On calcule le rectangle minimal contenant tous ces pixels.

    Paramètres
    ----------
    gray : np.ndarray (H, W) en uint8
        Image en niveaux de gris (objet clair sur fond sombre).

    Retourne
    --------
    tuple (x1, y1, x2, y2)
        Boîte englobante au format (x_min, y_min, x_max, y_max).

    Lève
    -----
    ValueError
        Si aucun pixel blanc n'est détecté (image trop sombre ou sans objet).

    Notes
    -----
    cv2.boundingRect retourne (x, y, w, h). On convertit immédiatement en
    (x1, y1, x2, y2) pour être compatible avec la fonction iou().
    """
    # Seuillage binaire : pixels >= 127 deviennent 255, les autres 0.
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Récupération des coordonnées des pixels non nuls.
    points = cv2.findNonZero(th)
    if points is None:
        raise ValueError(
            "Aucun pixel détecté après seuillage. "
            "Vérifier l'image ou le seuil."
        )

    # Rectangle englobant minimal au format (x, y, largeur, hauteur).
    x, y, w, h = cv2.boundingRect(points)

    # Conversion au format (x1, y1, x2, y2) pour le calcul d'IoU.
    return (x, y, x + w, y + h)


# ===========================================================================
# Pipeline principal
# ===========================================================================
# Les étapes ci-dessous forment le pipeline complet en ~15 lignes de code
# exécutable. C'est le minimum vital pour produire une IoU.

# 1) Scène de référence (vérité terrain) : rectangle sans décalage.
img_gt = make_synthetic_scene("rectangle", 0)

# 2) Scène prédite : même rectangle, décalé de 12 pixels vers la droite.
#    Ce décalage simule une détection imparfaite.
img_pred = make_synthetic_scene("rectangle", 12)

# 3) Conversion en niveaux de gris.
#    Le seuillage et boundingRect travaillent sur une seule intensité
#    par pixel, pas sur les 3 canaux BGR.
gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
gray_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)

# 4) Extraction automatique des boîtes englobantes.
box_gt = bbox_from_threshold(gray_gt)
box_pred = bbox_from_threshold(gray_pred)

# 5) Calcul et sauvegarde de l'IoU.
metrics = {
    "iou_score": float(iou(box_pred, box_gt)),
    "bbox_gt": box_gt,
    "bbox_pred": box_pred,
}

# Sauvegarde JSON : format standard, lisible, réutilisable.
Path("outputs/jour1").mkdir(parents=True, exist_ok=True)
Path("outputs/jour1/metrics_minimal.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)

# Affichage direct dans le terminal.
print(json.dumps(metrics, indent=2))
