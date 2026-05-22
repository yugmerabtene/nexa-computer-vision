#!/usr/bin/env python3
"""
Lab Jour 2 — CNN et Faster R-CNN
=================================

Objectif pédagogique :
    Ce lab illustre deux niveaux de vision par ordinateur basée sur
    l'apprentissage profond :

    1. Classification d'images avec un petit CNN entraîné from scratch
       sur un dataset synthétique (rectangles, cercles, triangles).
    2. Détection d'objets avec un Faster R-CNN pré-entraîné sur COCO,
       utilisé sur une image réelle (chien).

Compétences travaillées :
    - Construire un CNN avec PyTorch (nn.Conv2d, nn.MaxPool2d, nn.Linear).
    - Générer un dataset synthétique et le séparer en train/test (C3.2).
    - Entraîner un modèle et visualiser sa courbe de perte.
    - Charger un modèle torchvision pré-entraîné.
    - Exécuter une inférence Faster R-CNN et interpréter les sorties.
    - Calculer des métriques de détection (IoU, précision, rappel, AP).
    - Visualiser les feature maps d'un CNN.

Architecture du pipeline :
    1. SimpleCNN        -> modèle CNN pour la classification (3 classes)
    2. generate_dataset -> jeu de données synthétique
    3. train_test_split -> séparation contrôlée
    4. train_cnn        -> boucle d'entraînement
    5. evaluate_cnn     -> mesure de précision
    6. run_faster_rcnn_detection -> inférence avec Faster R-CNN
    7. detection_metrics_at_threshold -> calcul TP/FP/FN/précision/rappel
    8. average_precision_from_pr_rows -> approximation pédagogique de l'AP
    9. save_precision_recall_curve -> visualisation du compromis
    10. save_feature_maps -> visualisation des filtres appris
"""

import json
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
# Backend Agg : pas de fenêtre graphique. Les figures sont directement
# sauvegardées en PNG. Fonctionne en Docker, SSH, CI/CD, serveur.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)


# ===========================================================================
# CONSTANTES GLOBALES
# ===========================================================================

# Chemin vers l'image réelle d'un chien (licence libre, dataset COCO-like).
REAL_IMAGE_PATH = "labs/shared/assets/coco_dog.jpg"

# Image synthétique de secours si l'image réelle est absente.
SYNTHETIC_IMAGE_PATH = "labs/jour2/assets/test_detection.png"

# Boîte vérité terrain approximative du chien dans l'image réelle.
# Format (x1, y1, x2, y2). Utilisée pour calculer l'IoU avec les
# prédictions de Faster R-CNN.
REAL_DOG_GT_BOX = (50, 35, 645, 555)
RANDOM_SEED = 42


# ===========================================================================
# SimpleCNN — Architecture du modèle de classification
# ===========================================================================
class SimpleCNN(nn.Module):
    """Petit CNN pédagogique pour classifier des formes géométriques simples.

    Architecture :
        [Entrée : image RGB 64x64]
            -> Conv2d(3 -> 32, 3x3) + ReLU + MaxPool2d(2)   -> 32x32x32
            -> Conv2d(32 -> 64, 3x3) + ReLU + MaxPool2d(2)  -> 64x16x16
            -> Flatten() -> Linear(64*16*16 -> 128) -> ReLU
            -> Linear(128 -> num_classes)

    Pourquoi cette architecture ?
        - 2 blocs convolutionnels suffisent pour des formes simples.
        - Le max-pooling réduit rapidement la taille spatiale (64 -> 32 -> 16).
        - Les couches fully-connected en fin de réseau combinent les
          caractéristiques apprises pour prendre une décision.

    Forward :
        x: tenseur (B, 3, 64, 64) -> logits (B, num_classes)
    """

    def __init__(self, num_classes=3):
        super().__init__()

        # Bloc d'extraction de caractéristiques (features).
        # Les convolutions apprennent des motifs visuels locaux (bords,
        # textures, formes). Le pooling rend ces motifs invariants aux
        # petites translations et réduit la dimension.
        self.features = nn.Sequential(
            # Première convolution : 3 canaux RGB -> 32 filtres.
            # kernel_size=3, padding=1 conserve la taille spatiale.
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # MaxPooling 2x2 : divise la hauteur et la largeur par 2.
            # 64x64 -> 32x32.
            nn.MaxPool2d(2),
            # Deuxième convolution : 32 -> 64 filtres.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Deuxième pooling : 32x32 -> 16x16.
            nn.MaxPool2d(2),
        )

        # Bloc classificateur (fully-connected).
        # Après 2 max-pooling(2) : 64x64 -> 16x16.
        # Taille aplatie = 64 (canaux) * 16 * 16 = 16384.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),  # logits de sortie (pas de softmax)
        )

    def forward(self, x):
        # Passage avant : l'image traverse les features puis le classifier.
        # Les logits bruts sont retournés (CrossEntropyLoss applique softmax
        # en interne).
        return self.classifier(self.features(x))


# ===========================================================================
# generate_dataset — Création du jeu de données synthétique
# ===========================================================================
def generate_dataset(num_samples=200, img_size=64):
    """Génère un dataset synthétique de 3 classes : rectangle, cercle, triangle.

    Pourquoi un dataset synthétique ?
        - Permet un contrôle total des labels (pas d'annotation manuelle).
        - Les variations de position, taille et couleur évitent que le CNN
          se contente de mémoriser un seul cas.
        - 100% reproductible grâce à np.random.RandomState(42).

    Paramètres
    ----------
    num_samples : int
        Nombre total d'images à générer (doit être un multiple de 3 pour
        l'équilibre des classes).
    img_size : int
        Taille (hauteur = largeur) des images carrées générées.

    Retourne
    --------
    X : torch.Tensor (num_samples, 3, img_size, img_size)
        Images normalisées dans [0, 1] au format PyTorch (N, C, H, W).
    y : torch.Tensor (num_samples,)
        Labels : 0 = rectangle, 1 = cercle, 2 = triangle.

    Distribution des classes
    ------------------------
    Les labels sont attribués par rotation modulo 3 (i % 3), ce qui garantit
    un équilibre parfait entre les trois classes.
    """
    X = []
    y = []
    rng = np.random.RandomState(42)

    for i in range(num_samples):
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        label = i % 3  # 0=rectangle, 1=cercle, 2=triangle

        # Position et taille aléatoires.
        x1 = int(rng.randint(5, 20))
        y1 = int(rng.randint(5, 20))
        x2 = int(rng.randint(40, 59))
        y2 = int(rng.randint(40, 59))

        # Couleur aléatoire (RGB, valeurs entre 100 et 255).
        color = tuple(int(c) for c in rng.randint(100, 256, 3))

        if label == 0:
            # Rectangle plein.
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        elif label == 1:
            # Cercle inscrit dans le rectangle défini par (x1,y1)-(x2,y2).
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = min(x2 - x1, y2 - y1) // 2
            cv2.circle(img, (cx, cy), r, color, -1)
        else:
            # Triangle isocèle (pointe vers le haut).
            pts = np.array([[(x1 + x2) // 2, y1], [x1, y2], [x2, y2]], dtype=np.int32)
            cv2.fillPoly(img, [pts], color)

        X.append(img)
        y.append(label)

    # Conversion :
    #   Format OpenCV : (N, H, W, C) en uint8 [0, 255]
    #   Format PyTorch : (N, C, H, W) en float32 [0, 1]
    X = np.array(X, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
    y = torch.tensor(y, dtype=torch.long)
    return torch.tensor(X), y


# ===========================================================================
# set_reproducible_seed
# ===========================================================================
def set_reproducible_seed(seed=RANDOM_SEED):
    """Fixe les graines de NumPy et PyTorch pour une reproductibilité stricte.

    Sans cette fonction, chaque exécution produirait des poids initiaux
    et des splits différents, rendant la comparaison des résultats impossible.

    Notes
    -----
    - torch.set_determinism(True) n'est pas appelé ici pour rester compatible
      CPU/GPU sans pénalité de performance.
    - Les opérations sur GPU restent non-déterministes par nature.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


# ===========================================================================
# train_test_split
# ===========================================================================
def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Sépare le dataset en sous-ensembles d'entraînement et de test.

    Pourquoi un split explicite ?
        Compétence C3.2 : organiser les données en au moins deux
        sous-ensembles distincts (entraînement vs test/validation).
        Cela permet de vérifier que le modèle généralise et ne se contente
        pas de mémoriser les données vues.

    Paramètres
    ----------
    X : torch.Tensor
        Données (images).
    y : torch.Tensor
        Labels.
    test_ratio : float
        Proportion du dataset réservée au test (0.2 = 20%).
    seed : int
        Graine pour la reproductibilité du split.

    Retourne
    --------
    X_train, y_train, X_test, y_test : torch.Tensor
    """
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(X), generator=generator)
    test_size = int(len(X) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# ===========================================================================
# evaluate_cnn
# ===========================================================================
def evaluate_cnn(model, X, y):
    """Évalue la précision d'un CNN sur un sous-ensemble donné (train ou test).

    Paramètres
    ----------
    model : nn.Module
    X : torch.Tensor
    y : torch.Tensor

    Retourne
    --------
    float
        Précision (accuracy) entre 0 et 1.
    """
    model.eval()

    # Désactivation du gradient pour l'évaluation : moins de mémoire,
    # plus rapide, et pas de modification accidentelle des poids.
    with torch.no_grad():
        out = model(X)
        preds = out.argmax(dim=1)
        return (preds == y).float().mean().item()


# ===========================================================================
# train_cnn
# ===========================================================================
def train_cnn(model, X, y, epochs=15, lr=0.001, batch_size=32):
    """Entraîne le CNN et retourne l'historique des pertes et la précision.

    Déroulement d'un epoch :
        1. Mélange aléatoire des indices (torch.randperm).
        2. Découpage en mini-batches de taille batch_size.
        3. Pour chaque batch : forward, loss, backward, optimizer.step.
        4. Moyenne de la perte sur l'epoch.
        5. Évaluation de la précision sur l'ensemble du dataset.

    Paramètres
    ----------
    model : SimpleCNN
    X : torch.Tensor (N, C, H, W)
    y : torch.Tensor (N,)
    epochs : int
    lr : float
    batch_size : int

    Retourne
    --------
    losses : list[float]
        Perte moyenne à chaque epoch.
    accuracy : float
        Précision finale sur le dataset fourni.
    """
    # CrossEntropyLoss combine log-softmax et negative log-likelihood.
    # Elle attend les logits bruts en entrée.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    n = len(X)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Mélange des indices pour éviter que le modèle apprenne l'ordre
        # des échantillons plutôt que le contenu des images.
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            batch_x = X[idx]
            batch_y = y[idx]

            # Réinitialisation des gradients (ils s'accumulent par défaut).
            optimizer.zero_grad()

            # Forward : calcul des logits.
            out = model(batch_x)

            # Calcul de la perte.
            loss = criterion(out, batch_y)

            # Backward : calcul des gradients.
            loss.backward()

            # Mise à jour des poids.
            optimizer.step()

            epoch_loss += loss.item() * len(idx)

        avg_loss = epoch_loss / n
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Évaluation finale sur le dataset d'entraînement.
    accuracy = evaluate_cnn(model, X, y)
    print(f"  Précision finale : {accuracy:.3f}")
    return losses, accuracy


# ===========================================================================
# run_faster_rcnn_detection
# ===========================================================================
def run_faster_rcnn_detection(img_path, score_thresh=0.5):
    """Exécute Faster R-CNN pré-entraîné sur une image et retourne les détections.

    Faster R-CNN est un détecteur two-stage :
        1. RPN (Region Proposal Network) : propose des régions candidates.
        2. Fast R-CNN : classe chaque région et affine sa boîte.

    Le modèle utilisé est fasterrcnn_resnet50_fpn_v2, pré-entraîné sur COCO
    (80 classes d'objets courants : personne, chien, voiture, etc.).

    Paramètres
    ----------
    img_path : str
        Chemin vers l'image d'entrée.
    score_thresh : float
        Seuil de confiance : seules les détections avec un score >= ce seuil
        sont retournées. Plus le seuil est bas, plus on récupère de
        détections (mais aussi plus de faux positifs).

    Retourne
    --------
    predictions : dict
        Dictionnaire contenant 'boxes', 'labels', 'scores' (tenseurs).
    img : np.ndarray
        Image originale lue par OpenCV (format BGR).

    Lève
    -----
    FileNotFoundError
        Si l'image n'est pas trouvée.
    """
    # Chargement du modèle pré-entraîné depuis torchvision.
    # Les poids sont téléchargés automatiquement au premier appel.
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=score_thresh)
    model.eval()

    # Lecture de l'image.
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image non trouvée : {img_path}")

    # Conversion OpenCV (BGR, HWC) -> PyTorch (RGB, CHW) -> float [0,1].
    img_tensor = (
        torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    # Inférence : torch.no_grad() désactive le gradient pour économiser
    # de la mémoire et accélérer le calcul.
    with torch.no_grad():
        predictions = model([img_tensor])

    return predictions[0], img


# ===========================================================================
# draw_detections
# ===========================================================================
def draw_detections(img, boxes, labels, scores):
    """Dessine les boîtes détectées sur l'image avec leur label et score.

    Paramètres
    ----------
    img : np.ndarray (BGR)
    boxes : np.ndarray (N, 4)
    labels : np.ndarray (N,)
    scores : np.ndarray (N,)

    Retourne
    --------
    np.ndarray
        Image avec les boîtes et annotations dessinées.
    """
    # Dictionnaire de couleurs pour quelques classes COCO.
    # Les classes non listées reçoivent un gris neutre (128, 128, 128).
    COCO_COLORS = {
        1: (255, 0, 0),    # person -> rouge
        2: (0, 255, 0),    # bicycle -> vert
        3: (0, 0, 255),    # car -> bleu
        16: (255, 255, 0), # dog -> cyan
        17: (255, 0, 255), # horse -> magenta
        18: (0, 255, 255), # sheep -> jaune
        19: (200, 200, 0), # cow -> olive
        44: (200, 0, 200), # bottle -> violet
        62: (0, 200, 200), # tv -> turquoise
    }

    for box, label, score in zip(boxes, labels, scores):
        color = COCO_COLORS.get(int(label), (128, 128, 128))
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img, f"{int(label)}:{score:.2f}", (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )
    return img


# ===========================================================================
# compute_iou
# ===========================================================================
def compute_iou(box_a, box_b):
    """Calcule l'IoU entre deux boîtes (identique à la version du Jour 1).

    Cette fonction est dupliquée ici pour que le lab du Jour 2 reste
    autonome et ne dépende pas du code du Jour 1.

    Paramètres
    ----------
    box_a, box_b : tuple/list (x1, y1, x2, y2)

    Retourne
    --------
    float
        IoU entre 0.0 et 1.0.
    """
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


# ===========================================================================
# prepare_detection_image
# ===========================================================================
def prepare_detection_image():
    """Sélectionne l'image de test : réelle (COCO) ou synthétique (fallback).

    Stratégie :
        1. Si l'image réelle labs/shared/assets/coco_dog.jpg existe,
           l'utiliser (les modèles pré-entraînés répondent mieux).
        2. Sinon, générer une image synthétique de secours contenant un
           rectangle et un cercle.

    Retourne
    --------
    img_path : str
        Chemin vers l'image sélectionnée.
    gt_boxes : list[tuple]
        Boîtes vérité terrain correspondantes.
    image_source : str
        "real_coco_dog" ou "synthetic_shapes".
    """
    if os.path.exists(REAL_IMAGE_PATH):
        return REAL_IMAGE_PATH, [REAL_DOG_GT_BOX], "real_coco_dog"

    # Fallback synthétique : l'image réelle n'est pas disponible.
    os.makedirs(os.path.dirname(SYNTHETIC_IMAGE_PATH), exist_ok=True)
    test_img = np.zeros((400, 500, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (50, 60), (200, 220), (255, 255, 255), -1)
    cv2.circle(test_img, (350, 200), 70, (200, 200, 0), -1)
    cv2.imwrite(SYNTHETIC_IMAGE_PATH, test_img)
    return (
        SYNTHETIC_IMAGE_PATH,
        [(40, 50, 210, 230), (270, 120, 430, 280)],
        "synthetic_shapes",
    )


# ===========================================================================
# detection_metrics_at_threshold
# ===========================================================================
def detection_metrics_at_threshold(boxes, scores, gt_boxes, score_thresh, iou_thresh=0.5):
    """Calcule TP, FP, FN, précision et rappel pour un seuil de score donné.

    Principe :
        - TP (vrai positif) : boîte prédite avec IoU >= iou_thresh avec une GT.
        - FP (faux positif) : boîte prédite sans GT correspondante.
        - FN (faux négatif) : GT sans aucune prédiction correspondante.
        - Une GT ne peut être associée qu'à une seule prédiction (greedy).

    Paramètres
    ----------
    boxes : np.ndarray (N, 4)
    scores : np.ndarray (N,)
    gt_boxes : list[tuple]
    score_thresh : float
    iou_thresh : float

    Retourne
    --------
    dict avec clés : threshold, tp, fp, fn, precision, recall.
    """
    # Filtrage des prédictions par seuil de score.
    selected = [(box, score) for box, score in zip(boxes, scores) if score >= score_thresh]
    matched_gt = set()
    tp = 0
    fp = 0

    for pred_box, _ in selected:
        best_iou = 0.0
        best_idx = None

        # Recherche de la meilleure GT non encore associée.
        for idx, gt_box in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou_val = compute_iou(
                (pred_box[0], pred_box[1], pred_box[2], pred_box[3]), gt_box
            )
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = idx

        if best_iou >= iou_thresh and best_idx is not None:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": score_thresh,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
    }


# ===========================================================================
# average_precision_from_pr_rows
# ===========================================================================
def average_precision_from_pr_rows(pr_rows):
    """Calcule une approximation pédagogique de l'Average Precision (AP).

    Méthode :
        Pour chaque valeur de rappel unique, on prend la précision maximale
        atteinte. L'AP est l'aire sous la courbe précision-rappel ainsi
        construite (approximation trapézoïdale simplifiée).

    Attention :
        Ce n'est PAS le calcul officiel COCO/Pascal VOC. C'est une version
        simplifiée pour illustrer le principe de l'aire sous la courbe.
        L'AP officielle utilise 101 points d'interpolation et des seuils
        d'IoU multiples.

    Paramètres
    ----------
    pr_rows : list[dict]
        Chaque dict contient 'recall' et 'precision'.

    Retourne
    --------
    float
        AP approximative entre 0 et 1.
    """
    by_recall = {}
    for row in pr_rows:
        recall = float(row["recall"])
        precision = float(row["precision"])
        by_recall[recall] = max(by_recall.get(recall, 0.0), precision)

    points = sorted(by_recall.items(), key=lambda item: item[0])
    ap = 0.0
    prev_recall = 0.0
    for recall, precision in points:
        delta = max(0.0, recall - prev_recall)
        ap += precision * delta
        prev_recall = max(prev_recall, recall)
    return ap


# ===========================================================================
# save_precision_recall_curve
# ===========================================================================
def save_precision_recall_curve(boxes, scores, gt_boxes):
    """Trace l'évolution de la précision et du rappel selon le seuil de score.

    L'objectif est de montrer le compromis fondamental en détection :
    - Seuil bas  -> beaucoup de détections (rappel élevé, précision faible).
    - Seuil haut -> détections fiables (précision élevée, rappel faible).

    Paramètres
    ----------
    boxes : np.ndarray
    scores : np.ndarray
    gt_boxes : list[tuple]

    Retourne
    --------
    rows : list[dict]
        Métriques pour chaque seuil testé.
    path : str
        Chemin vers la figure sauvegardée.
    """
    thresholds = [0.05, 0.1, 0.15, 0.25, 0.5]
    rows = [detection_metrics_at_threshold(boxes, scores, gt_boxes, t) for t in thresholds]

    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, [r["precision"] for r in rows], marker="o", label="Précision")
    plt.plot(thresholds, [r["recall"] for r in rows], marker="o", label="Rappel")
    plt.xlabel("Seuil de score")
    plt.ylabel("Valeur")
    plt.ylim(-0.05, 1.05)
    plt.title("Précision/Rappel selon le seuil Faster R-CNN")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = "outputs/jour2/figures/precision_recall.png"
    plt.savefig(path, dpi=130)
    plt.close()
    return rows, path


# ===========================================================================
# save_feature_maps
# ===========================================================================
def save_feature_maps(model, X):
    """Visualise les 8 premières cartes d'activation du CNN entraîné.

    Les feature maps montrent comment chaque filtre de convolution répond
    à l'image d'entrée. Certains filtres détectent des bords horizontaux,
    d'autres des verticaux, d'autres des textures.

    Pédagogie :
        Cette visualisation rend concret le concept de « filtre appris ».
        Les étudiants voient que les premiers filtres du CNN ressemblent
        à des détecteurs de HOG ou SIFT, mais appris automatiquement.

    Paramètres
    ----------
    model : SimpleCNN
    X : torch.Tensor (batch d'images de test)

    Retourne
    --------
    path : str
        Chemin vers la figure sauvegardée.
    """
    model.eval()
    with torch.no_grad():
        # model.features[:2] = Conv2d + ReLU de la première couche.
        # On prend la première image du batch (X[:1]).
        activations = model.features[:2](X[:1]).squeeze(0).cpu().numpy()

    n_maps = min(8, activations.shape[0])
    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    for idx, ax in enumerate(axs.ravel()):
        if idx < n_maps:
            ax.imshow(activations[idx], cmap="viridis")
            ax.set_title(f"Filtre {idx + 1}")
        ax.axis("off")
    plt.suptitle("Premières feature maps du CNN")
    plt.tight_layout()
    path = "outputs/jour2/figures/feature_maps.png"
    plt.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ===========================================================================
# main — Fonction principale
# ===========================================================================
def main():
    """Orchestre tout le lab Jour 2.

    Déroulement :
        1.  Génération du dataset synthétique + split train/test (C3.2).
        2.  Création et entraînement du CNN.
        3.  Évaluation sur le test set.
        4.  Visualisation de la courbe de perte et des feature maps.
        5.  Chargement de l'image de test (réelle ou synthétique).
        6.  Inférence avec Faster R-CNN pré-entraîné.
        7.  Calcul des métriques de détection (IoU, précision, rappel, AP).
        8.  Génération des figures (détection, précision/rappel).
        9.  Sauvegarde du rapport JSON complet.
    """
    set_reproducible_seed()
    os.makedirs("outputs/jour2/figures", exist_ok=True)

    # ======================================================================
    # PARTIE 1 : CNN
    # ======================================================================
    print("=" * 50)
    print("PARTIE 1 : CNN — Entraînement sur données synthétiques")
    print("=" * 50)

    X, y = generate_dataset(num_samples=360, img_size=64)

    # Split train/test (compétence C3.2).
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2, seed=42)
    model = SimpleCNN(num_classes=3)
    losses, train_accuracy = train_cnn(model, X_train, y_train, epochs=15)
    test_accuracy = evaluate_cnn(model, X_test, y_test)
    print(f"  Précision test : {test_accuracy:.3f}")

    # Visualisation des feature maps.
    feature_maps_path = save_feature_maps(model, X_test)
    print(f"  Feature maps sauvegardées : {feature_maps_path}")

    # Courbe de perte : vérification visuelle de la convergence.
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=2, color="steelblue")
    plt.title("Perte d'entraînement du CNN")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/jour2/figures/cnn_training.png", dpi=130)
    plt.close()
    print(f"  Courbe sauvegardée : outputs/jour2/figures/cnn_training.png")

    # ======================================================================
    # PARTIE 2 : Faster R-CNN
    # ======================================================================
    print("\n" + "=" * 50)
    print("PARTIE 2 : Faster R-CNN — Détection et évaluation")
    print("=" * 50)

    test_img_path, gt_boxes, image_source = prepare_detection_image()
    print(f"  Image de test : {test_img_path} ({image_source})")

    # Exécution de Faster R-CNN avec un seuil bas (0.1) pour récupérer
    # un maximum de détections et analyser le compromis.
    pred, img_bgr = run_faster_rcnn_detection(test_img_path, score_thresh=0.1)

    # Conversion des tenseurs PyTorch en tableaux NumPy.
    boxes = pred["boxes"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()

    print(f"  Détections : {len(boxes)}")
    for box, label, score in zip(boxes, labels, scores):
        print(
            f"    Classe {label}: score={score:.3f}, "
            f"box=({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})"
        )

    # Sauvegarde de l'image avec les détections dessinées.
    img_result = draw_detections(img_bgr.copy(), boxes, labels, scores)
    result_path = "outputs/jour2/figures/detection_result.png"
    cv2.imwrite(result_path, img_result)
    print(f"  Résultat sauvegardé : {result_path}")

    # Calcul de l'IoU entre chaque GT et la meilleure prédiction.
    ious = []
    for gt_box in gt_boxes:
        best_iou = 0
        for pred_box in boxes:
            iou_val = compute_iou(
                (pred_box[0], pred_box[1], pred_box[2], pred_box[3]), gt_box
            )
            best_iou = max(best_iou, iou_val)
        ious.append(best_iou)

    avg_iou = float(np.mean(ious)) if ious else 0.0

    # Courbe précision-rappel et AP simplifiée.
    pr_rows, pr_path = save_precision_recall_curve(boxes, scores, gt_boxes)
    ap50 = average_precision_from_pr_rows(pr_rows)
    print(f"  Courbe précision-rappel sauvegardée : {pr_path}")

    # ======================================================================
    # Rapport JSON final
    # ======================================================================
    metrics = {
        "cnn_final_loss": round(losses[-1], 4),
        "dataset_split": {
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "test_ratio": 0.2,
        },
        "cnn_train_accuracy": round(train_accuracy, 4),
        "cnn_test_accuracy": round(test_accuracy, 4),
        "frcnn_num_detections": int(len(boxes)),
        "image_source": image_source,
        "gt_boxes": [list(box) for box in gt_boxes],
        "frcnn_detections": [
            {
                "label": int(l),
                "score": round(float(s), 3),
                "box": [round(float(b), 1) for b in box],
            }
            for box, l, s in zip(boxes, labels, scores)
        ],
        "avg_iou": round(avg_iou, 4),
        "iou_per_gt": [round(float(i), 4) for i in ious],
        "ap50_simplified": round(float(ap50), 4),
        "map50_simplified": round(float(ap50), 4),
        "metric_note": "AP/mAP simplifiés pour usage pédagogique sur une seule classe et quelques seuils, pas une évaluation COCO officielle.",
        "precision_recall": [
            {
                "threshold": round(float(row["threshold"]), 3),
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "precision": round(float(row["precision"]), 4),
                "recall": round(float(row["recall"]), 4),
            }
            for row in pr_rows
        ],
        "figures": {
            "cnn_training": "outputs/jour2/figures/cnn_training.png",
            "detection_result": result_path,
            "precision_recall": pr_path,
            "feature_maps": feature_maps_path,
        },
    }

    with open("outputs/jour2/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Métriques sauvegardées : outputs/jour2/metrics.json")
    print(json.dumps(metrics, indent=2))


# ===========================================================================
# Point d'entrée
# ===========================================================================
if __name__ == "__main__":
    main()
