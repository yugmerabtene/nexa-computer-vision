"""
Lab complet du Jour 1 — Fondamentaux OpenCV, IoU, HOG et SIFT
================================================================

Objectif pédagogique :
    Construire une mini-chaîne de vision classique (pipeline) pour comprendre
    chaque étape de A à Z : génération d'images synthétiques, segmentation
    par seuillage, extraction de boîtes englobantes, évaluation par IoU,
    calcul de descripteurs globaux (HOG) et locaux (SIFT), et matching.

Compétences travaillées :
    - Manipuler une image sous forme de tableau NumPy (BGR, niveaux de gris).
    - Convertir, redimensionner, seuiller une image avec OpenCV.
    - Calculer l'Intersection over Union (IoU) entre deux boîtes.
    - Extraire un descripteur HOG et interpréter sa dimension.
    - Détecter des points clés SIFT et appliquer le ratio test de Lowe.
    - Produire des métriques reproductibles au format JSON.

Architecture du pipeline :
    1. ensure_dirs()         -> création des dossiers de sortie
    2. make_synthetic_scene() -> génération de formes contrôlées
    3. bbox_from_threshold() -> localisation par seuillage
    4. iou()                 -> mesure de la qualité de localisation
    5. hog_features()        -> descripteur global de forme
    6. sift_features()       -> points clés locaux
    7. ratio_match_count()   -> matching avec filtrage de Lowe
    8. run()                 -> orchestration + sauvegarde JSON + figure
"""

import json
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Backend non interactif pour Matplotlib
# ---------------------------------------------------------------------------
# En environnement serveur, Docker ou SSH, il n'y a pas d'écran pour afficher
# les figures. En passant le backend sur "Agg", on force Matplotlib à générer
# les images en mémoire et à les sauvegarder sur disque via savefig().
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# ensure_dirs
# ---------------------------------------------------------------------------
def ensure_dirs() -> tuple[Path, Path]:
    """Crée les dossiers de sortie organisés par type d'artefact.

    Retourne
    --------
    out_dir : Path
        Dossier racine des sorties du jour 1 (outputs/jour1).
        Contient le fichier metrics.json.
    figures_dir : Path
        Sous-dossier dédié aux figures (outputs/jour1/figures).
        Sépare les visuels des métriques pour une arborescence claire.

    Notes
    -----
    parents=True  -> crée toute la chaîne outputs/jour1/figures d'un coup.
    exist_ok=True -> ne lève pas d'erreur si le dossier existe déjà
                     (utile pour une ré-exécution).
    """
    out_dir = Path("outputs/jour1")
    out_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, figures_dir


# ---------------------------------------------------------------------------
# make_synthetic_scene
# ---------------------------------------------------------------------------
def make_synthetic_scene(shape: str, shift: int = 0) -> np.ndarray:
    """Génère une image synthétique composée d'une forme blanche sur fond noir.

    Pourquoi du synthétique ?
        Les formes géométriques permettent un contrôle parfait de la vérité
        terrain (GT). On connaît exactement la position, la taille et la
        forme de l'objet, ce qui est indispensable pour :
        - valider le calcul d'IoU sans ambiguïté ;
        - vérifier que les descripteurs HOG/SIFT séparent bien des formes
          différentes tout en restant stables pour des formes similaires.

    Paramètres
    ----------
    shape : str
        "rectangle" ou "circle". Détermine la forme géométrique tracée.
    shift : int
        Décalage horizontal en pixels appliqué à la forme. Sert à simuler
        une prédiction imparfaite (bounding box légèrement décalée).

    Retourne
    --------
    img : np.ndarray
        Image BGR de taille 256x256, fond noir (0), forme blanche (255).

    Exemples
    --------
    >>> make_synthetic_scene("rectangle", shift=0)   # référence
    >>> make_synthetic_scene("rectangle", shift=12)  # prédiction décalée
    >>> make_synthetic_scene("circle", shift=0)      # forme différente
    """
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    if shape == "rectangle":
        # Rectangle de 140x130 pixels. shift le décale horizontalement.
        cv2.rectangle(img, (40 + shift, 60), (180 + shift, 190), (255, 255, 255), -1)

    elif shape == "circle":
        # Cercle de rayon 60 centré en (120, 130). Décalable aussi.
        cv2.circle(img, (120 + shift, 130), 60, (255, 255, 255), -1)

    else:
        raise ValueError("shape must be 'rectangle' or 'circle'")

    return img


# ---------------------------------------------------------------------------
# iou  (Intersection over Union)
# ---------------------------------------------------------------------------
def iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Calcule l'Intersection over Union (IoU) entre deux boîtes.

    Rappel mathématique :
        IoU = Aire(Intersection) / Aire(Union)

    L'IoU est comprise entre 0.0 (aucun recouvrement) et 1.0 (superposition
    parfaite). C'est la métrique standard pour évaluer la qualité de
    localisation d'un détecteur d'objets.

    Paramètres
    ----------
    box_a, box_b : tuple (x1, y1, x2, y2)
        Coordonnées des deux boîtes au format (coin_haut_gauche_x,
        coin_haut_gauche_y, coin_bas_droit_x, coin_bas_droit_y).

    Retourne
    --------
    float
        Score IoU entre 0.0 et 1.0.

    Déroulement du calcul
    ---------------------
    1. Coin haut-gauche de l'intersection = max() des deux x1 et y1.
    2. Coin bas-droit de l'intersection   = min() des deux x2 et y2.
    3. Si les coordonnées sont inversées -> pas d'intersection -> retour 0.
    4. Aire d'intersection = largeur * hauteur.
    5. Aire d'union = somme des aires des deux boîtes - intersection.
    6. IoU = intersection / union.
    """
    # Étape 1-2 : coordonnées du rectangle d'intersection
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    # Étape 3 : si pas d'intersection, retourner 0
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    # Étape 4-6 : calcul de l'IoU
    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union


# ---------------------------------------------------------------------------
# bbox_from_threshold
# ---------------------------------------------------------------------------
def bbox_from_threshold(img_gray: np.ndarray) -> tuple[int, int, int, int]:
    """Segmente l'objet clair par seuillage et retourne sa boîte englobante.

    Principe :
        Sur une image où l'objet est blanc (255) et le fond noir (0), un
        seuillage binaire simple à 127 isole parfaitement l'objet.
        On récupère ensuite les coordonnées de tous les pixels non nuls
        et on calcule le rectangle minimal qui les contient tous.

    Paramètres
    ----------
    img_gray : np.ndarray (H, W) en uint8
        Image en niveaux de gris.

    Retourne
    --------
    tuple (x1, y1, x2, y2)
        Boîte englobante au format (x_min, y_min, x_max, y_max).

    Lève
    -----
    ValueError
        Si aucun pixel blanc n'est trouvé après seuillage. Cela peut arriver
        si l'image est trop sombre ou si le seuil est mal choisi.

    Notes
    -----
    cv2.boundingRect retourne (x, y, w, h). On convertit en (x, y, x+w, y+h)
    pour rester cohérent avec le format attendu par la fonction iou() et
    par les standards de détection (utilisés aux Jours 2 et 3).
    """
    # Seuil fixe à 127 : tout pixel >= 127 devient 255, les autres 0.
    _, th = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # findNonZero retourne les coordonnées (x, y) de chaque pixel blanc.
    points = cv2.findNonZero(th)
    if points is None:
        raise ValueError(
            "Aucun pixel détecté après seuillage. "
            "Vérifier que l'image contient bien un objet clair sur fond sombre."
        )

    # Boîte englobante minimale (rotation non prise en compte).
    x, y, w, h = cv2.boundingRect(points)

    # Conversion (x, y, w, h) -> (x1, y1, x2, y2) pour le calcul d'IoU.
    return (x, y, x + w, y + h)


# ---------------------------------------------------------------------------
# hog_features
# ---------------------------------------------------------------------------
def hog_features(gray_img: np.ndarray) -> np.ndarray:
    """Calcule le descripteur HOG d'une image redimensionnée en 128x64.

    Qu'est-ce que HOG (Histogram of Oriented Gradients) ?
        HOG décrit la structure globale des contours d'une image en comptant
        l'orientation des gradients dans des cellules locales (8x8 pixels),
        puis en normalisant par blocs (16x16) pour être robuste aux
        variations de luminosité.

    Pourquoi redimensionner en 128x64 ?
        Le descripteur HOG d'OpenCV travaille sur une fenêtre de taille fixe.
        Sans redimensionnement, deux images de tailles différentes produiraient
        des vecteurs de dimensions différentes, rendant toute comparaison
        impossible.

    Paramètres
    ----------
    gray_img : np.ndarray (H, W) en uint8
        Image en niveaux de gris, n'importe quelle taille.

    Retourne
    --------
    np.ndarray de forme (3780, 1)
        Vecteur colonne contenant le descripteur HOG.

    Détail des paramètres HOG
    -------------------------
    - _winSize  = (128, 64) : fenêtre de détection.
    - _blockSize = (16, 16) : un block = 4 cellules (2x2).
    - _blockStride = (8, 8) : chevauchement des blocks.
    - _cellSize = (8, 8)    : une cellule = 8x8 pixels.
    - _nbins = 9            : 9 orientations (0-180°).

    Calcul de la dimension :
        Nombre de blocks horizontaux = (128 - 16) / 8 + 1 = 15
        Nombre de blocks verticaux   = (64 - 16) / 8 + 1 = 7
        Total blocks = 15 * 7 = 105
        Descripteur par block = 4 cellules * 9 bins = 36
        Dimension totale = 105 * 36 = 3780
    """
    # Redimensionnement obligatoire pour obtenir une fenêtre fixe.
    # INTER_AREA est recommandé pour la réduction (anti-crénelage).
    resized = cv2.resize(gray_img, (128, 64), interpolation=cv2.INTER_AREA)

    hog = cv2.HOGDescriptor(
        _winSize=(128, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    return hog.compute(resized)


# ---------------------------------------------------------------------------
# sift_features
# ---------------------------------------------------------------------------
def sift_features(gray_img: np.ndarray) -> tuple[list, np.ndarray | None]:
    """Détecte les points clés SIFT et calcule leurs descripteurs locaux.

    Qu'est-ce que SIFT (Scale-Invariant Feature Transform) ?
        SIFT détecte des points d'intérêt (angles, blobs, coins) qui sont
        invariants à l'échelle et à la rotation. Chaque point clé est décrit
        par un vecteur de 128 dimensions calculé à partir du voisinage local.

    Différence avec HOG :
        - HOG produit un seul vecteur global pour toute l'image.
        - SIFT produit plusieurs descripteurs locaux, un par point clé.
        - HOG est sensible à l'échelle ; SIFT y est invariant.

    Paramètres
    ----------
    gray_img : np.ndarray (H, W) en uint8
        Image en niveaux de gris.

    Retourne
    --------
    kp : list[cv2.KeyPoint]
        Liste des points clés détectés.
    desc : np.ndarray (N, 128) ou None
        Matrice des descripteurs (N = nombre de points clés).
        None si aucun point clé n'est trouvé (image trop uniforme).

    Notes
    -----
    Les formes géométriques pleines (rectangle blanc sur fond noir)
    produisent très peu de points clés car elles manquent de texture.
    C'est normal et pédagogiquement intéressant : cela illustre la limite
    de SIFT sur des images synthétiques lisses.
    """
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray_img, None)


# ---------------------------------------------------------------------------
# ratio_match_count
# ---------------------------------------------------------------------------
def ratio_match_count(
    desc_a: np.ndarray | None,
    desc_b: np.ndarray | None,
    ratio: float = 0.75,
) -> int:
    """Compte le nombre de « bons matches » entre deux ensembles de descripteurs
    SIFT en appliquant le ratio test de Lowe.

    Principe du ratio test de Lowe :
        Pour chaque descripteur de l'image A, on trouve ses deux plus proches
        voisins dans l'image B. Si la distance au premier voisin (d1) est
        nettement plus petite que la distance au second (d2), le match est
        considéré fiable. Sinon, le match est ambigu et rejeté.

        Condition de validité : d1 < ratio * d2

    Pourquoi ce filtre ?
        Sans ce test, un descripteur pourrait être associé à n'importe quel
        voisin, même très différent, créant des faux matches. Le ratio de
        Lowe (0.75 est la valeur recommandée par Lowe) élimine les
        correspondances ambiguës.

    Paramètres
    ----------
    desc_a : np.ndarray (N, 128) ou None
        Descripteurs de la première image.
    desc_b : np.ndarray (M, 128) ou None
        Descripteurs de la seconde image.
    ratio : float
        Seuil du ratio test (0.75 par défaut). Plus il est bas, plus le
        filtre est strict.

    Retourne
    --------
    int
        Nombre de bons matches retenus après filtrage.

    Notes
    -----
    Si l'une des deux images n'a pas de descripteurs ou en a moins de 2,
    le matching est impossible et on retourne 0.
    """
    # Cas où le matching n'est pas possible
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) < 2:
        return 0

    # BFMatcher = Brute-Force Matcher : compare chaque descripteur de A
    # avec tous les descripteurs de B pour trouver les plus proches voisins.
    # NORM_L2 = distance euclidienne (L2).
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # k=2 demande les 2 meilleurs voisins pour appliquer le ratio test.
    matches = bf.knnMatch(desc_a, desc_b, k=2)

    good = 0
    for pair in matches:
        if len(pair) < 2:
            continue  # pas assez de voisins pour appliquer le test
        m, n = pair  # m = meilleur voisin, n = second meilleur
        if m.distance < ratio * n.distance:
            good += 1

    return good


# ---------------------------------------------------------------------------
# run  (fonction principale)
# ---------------------------------------------------------------------------
def run() -> dict:
    """Orchestre l'exécution complète du lab Jour 1.

    Étapes détaillées
    -----------------
    1.  ensure_dirs()          -> prépare outputs/jour1/ et outputs/jour1/figures/
    2.  make_synthetic_scene() -> génère 3 images (GT, prédiction, autre forme)
    3.  Conversion BGR -> Gray pour HOG, SIFT et seuillage
    4.  bbox_from_threshold()  -> extrait les boîtes des 3 images
    5.  iou()                  -> compare boîte GT et boîte prédite
    6.  hog_features()         -> descripteurs globaux des 3 images
    7.  sift_features()        -> points clés / descripteurs locaux
    8.  ratio_match_count()    -> matching entre GT/Pred et GT/Autre
    9.  Figure de synthèse     -> visualisation GT / Pred / Keypoints
    10. Sauvegarde JSON        -> métriques exploitables pour le rapport

    Retourne
    --------
    dict
        Dictionnaire contenant toutes les métriques (IoU, normes L2 HOG,
        nombre de points clés SIFT, bons matches) sérialisable en JSON.
    """
    # --- Étape 1 : dossiers de sortie ---
    out_dir, figures_dir = ensure_dirs()

    # --- Étape 2 : génération des 3 scènes ---
    # GT         : rectangle de référence, sans décalage.
    # Prédiction : même rectangle, décalé de 12 px à droite.
    # Autre      : un cercle, forme différente.
    img_gt = make_synthetic_scene("rectangle", shift=0)
    img_pred = make_synthetic_scene("rectangle", shift=12)
    img_other = make_synthetic_scene("circle", shift=0)

    # --- Étape 3 : conversion en niveaux de gris ---
    # Tous les traitements (seuillage, HOG, SIFT) travaillent sur l'intensité
    # lumineuse, pas sur la couleur. Un seul canal suffit.
    gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
    gray_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
    gray_other = cv2.cvtColor(img_other, cv2.COLOR_BGR2GRAY)

    # --- Étape 4-5 : IoU entre boîte GT et boîte prédite ---
    box_gt = bbox_from_threshold(gray_gt)
    box_pred = bbox_from_threshold(gray_pred)
    iou_score = iou(box_pred, box_gt)

    # --- Étape 6 : descripteurs HOG ---
    hog_gt = hog_features(gray_gt)
    hog_pred = hog_features(gray_pred)
    hog_other = hog_features(gray_other)

    # --- Étape 7 : descripteurs SIFT ---
    kp_gt, desc_gt = sift_features(gray_gt)
    kp_pred, desc_pred = sift_features(gray_pred)
    kp_other, desc_other = sift_features(gray_other)

    # --- Étape 8 : matching SIFT ---
    # On s'attend à plus de matches entre deux rectangles proches qu'entre
    # un rectangle et un cercle (forme différente).
    good_similar = ratio_match_count(desc_gt, desc_pred)
    good_different = ratio_match_count(desc_gt, desc_other)

    # --- Étape 9 : figure de synthèse ---
    # 3 sous-graphiques : GT / Prédiction / Points clés SIFT.
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Sous-figure 1 : scène de référence (GT)
    axs[0].imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Scene GT")
    axs[0].axis("off")

    # Sous-figure 2 : scène prédite (décalée)
    axs[1].imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Scene Pred")
    axs[1].axis("off")

    # Sous-figure 3 : points clés SIFT superposés
    kp_img = cv2.drawKeypoints(
        gray_gt, kp_gt, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # affiche taille + orientation
    )
    axs[2].imshow(kp_img, cmap="gray")
    axs[2].set_title("SIFT keypoints")
    axs[2].axis("off")

    plt.tight_layout()
    figure_path = figures_dir / "jour1_overview.png"
    plt.savefig(figure_path, dpi=130)
    plt.close(fig)

    # --- Étape 10 : dictionnaire de métriques ---
    results = {
        # IoU : qualité de la localisation (proche de 1 = excellent)
        "iou_score": float(iou_score),
        "bbox_gt": box_gt,
        "bbox_pred": box_pred,
        # HOG : dimension du descripteur et distances inter-formes
        "hog_dimension": int(hog_gt.shape[0]),
        "hog_shifted_l2": float(np.linalg.norm(hog_gt - hog_pred)),
        "hog_different_l2": float(np.linalg.norm(hog_gt - hog_other)),
        # SIFT : nombre de points clés par image
        "sift_kp_gt": len(kp_gt),
        "sift_kp_pred": len(kp_pred),
        "sift_kp_other": len(kp_other),
        # SIFT : bons matches (similaires doit être > différents)
        "sift_good_matches_similar": good_similar,
        "sift_good_matches_different": good_different,
        # Chemin vers la figure de synthèse
        "figure_path": str(figure_path),
    }

    # Sauvegarde des métriques au format JSON (lisible, standard, réutilisable)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Exécution du lab et affichage des métriques dans la console.
    # Les métriques sont aussi sauvegardées dans outputs/jour1/metrics.json.
    metrics = run()
    print(json.dumps(metrics, indent=2))
