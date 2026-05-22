#!/usr/bin/env python3
"""
Lab Jour 3 — YOLOv8, comparaison et optimisation
=================================================

Objectif pédagogique :
    Ce lab clôt le module en comparant deux grandes familles de détecteurs
    sur les mêmes images :

    1. **Faster R-CNN** (Jour 2) : détecteur two-stage, lent mais précis.
    2. **YOLOv8n** (Jour 3) : détecteur one-stage, rapide et adapté
       au temps réel.

    L'objectif est d'analyser le **compromis vitesse/précision** et de
    produire une **recommandation** motivée pour un cas d'usage donné.

Compétences travaillées :
    - Charger et exécuter YOLOv8 avec Ultralytics.
    - Mesurer le temps d'inférence de manière reproductible.
    - Calculer l'IoU pour comparer la qualité de localisation.
    - Analyser l'impact du seuil de confiance (threshold sweep).
    - Calculer un mAP@0.5 simplifié à partir de la courbe précision/rappel.
    - Produire un rapport comparatif structuré (JSON + figures).

Liens avec les jours précédents :
    - Jour 1 : IoU, pipeline de vision, descripteurs manuels.
    - Jour 2 : CNN, Faster R-CNN, métriques de détection.
    - Jour 3 : synthèse, comparaison, optimisation, recommandation.

Architecture du pipeline :
    1. set_reproducible_seed    -> reproductibilité
    2. create_test_image        -> image synthétique de secours
    3. load_detection_image     -> image réelle ou synthétique
    4. compute_iou              -> IoU entre deux boîtes
    5. run_frcnn / run_yolo     -> inférence avec les deux modèles
    6. benchmark                -> mesure de temps moyenne
    7. precision_recall_at_threshold -> TP/FP/FN par seuil
    8. average_precision_from_pr_rows -> AP simplifiée
    9. draw_detections          -> annotation visuelle
    10. best_ious               -> meilleur IoU par objet GT
    11. main                    -> orchestration + figures + JSON
"""

import json
import os
import time
import numpy as np
import cv2
import torch
import matplotlib
# Backend non interactif : pas de fenêtre, sauvegarde directe en PNG.
# Fonctionne sur tout environnement (serveur, Docker, SSH, CI).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)


# ===========================================================================
# CONSTANTES GLOBALES
# ===========================================================================

# Chemin vers l'image réelle d'un chien (licence libre, COCO-like).
REAL_IMAGE_PATH = "labs/shared/assets/coco_dog.jpg"

# Boîte vérité terrain du chien dans l'image réelle.
# Format (x1, y1, x2, y2). Utilisée pour comparer l'IoU des deux modèles.
REAL_DOG_GT_BOX = (50, 35, 645, 555)
RANDOM_SEED = 42


# ===========================================================================
# set_reproducible_seed
# ===========================================================================
def set_reproducible_seed(seed=RANDOM_SEED):
    """Fixe les graines pseudo-aléatoires pour garantir la reproductibilité.

    Sans cette fonction, les résultats numériques (notamment les temps
    d'inférence) peuvent varier d'une exécution à l'autre à cause de
    l'initialisation aléatoire et des bruits systèmes.

    Notes
    -----
    - Ne garantit pas une reproductibilité absolue sur GPU.
    - Les temps CPU restent sensibles à la charge système.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


# ===========================================================================
# create_test_image
# ===========================================================================
def create_test_image(path, seed=42):
    """Crée une image de test synthétique avec des formes géométriques variées.

    Cette image est un **fallback** utilisé uniquement si l'image réelle
    (coco_dog.jpg) n'est pas disponible. Elle contient des formes qui
    simulent grossièrement des objets de différentes tailles pour permettre
    de discuter la détection multi-échelle.

    Contenu :
        - Fond dégradé (évite un histogramme trivial).
        - 4 formes de tailles différentes : grand rectangle, cercle moyen,
          rectangle moyen, petit rectangle.

    Paramètres
    ----------
    path : str
        Chemin de sauvegarde de l'image générée.
    seed : int
        Graine aléatoire pour la reproductibilité.

    Retourne
    --------
    img : np.ndarray (480, 640, 3) en uint8 (format BGR).
    """
    rng = np.random.RandomState(seed)

    # Image BGR 480x640 (ratio 3:4, proche d'une photo standard).
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Fond dégradé vertical pour une scène plus réaliste qu'un fond noir.
    for y in range(480):
        img[y, :] = [
            int(30 + 20 * y / 480),
            int(30 + 15 * y / 480),
            int(50 + 25 * y / 480),
        ]

    # 4 formes de tailles différentes pour discuter la détection multi-objet.
    cv2.rectangle(img, (80, 100), (200, 350), (180, 120, 80), -1)     # ~personne (grand)
    cv2.circle(img, (450, 250), 80, (200, 150, 0), -1)                # ~ballon (moyen)
    cv2.rectangle(img, (300, 300), (500, 420), (100, 100, 100), -1)   # ~meuble (moyen)
    cv2.rectangle(img, (50, 50), (120, 80), (220, 220, 220), -1)      # ~petit objet

    cv2.imwrite(path, img)
    return img


# ===========================================================================
# load_detection_image
# ===========================================================================
def load_detection_image(path):
    """Charge l'image réelle si disponible, sinon génère un fallback synthétique.

    Stratégie de sélection :
        - L'image réelle est privilégiée car les modèles pré-entraînés sur
          COCO répondent beaucoup mieux à des photographies naturelles qu'à
          des formes géométriques abstraites.
        - Si l'image réelle est absente (première exécution, déploiement
          partiel), on crée une scène synthétique pour que le lab reste
          exécutable.

    Paramètres
    ----------
    path : str
        Chemin où stocker l'image de test (utilisé pour le fallback).

    Retourne
    --------
    img : np.ndarray BGR
    img_path : str
        Chemin de l'image effectivement utilisée.
    gt_boxes : list[tuple]
        Boîtes vérité terrain.
    image_source : str
        "real_coco_dog" ou "synthetic_shapes".
    """
    if os.path.exists(REAL_IMAGE_PATH):
        img = cv2.imread(REAL_IMAGE_PATH)
        if img is None:
            raise FileNotFoundError(f"Image non lisible : {REAL_IMAGE_PATH}")
        return img, REAL_IMAGE_PATH, [REAL_DOG_GT_BOX], "real_coco_dog"

    # Fallback : génération d'une scène synthétique.
    img = create_test_image(path)
    gt_boxes = [
        (70, 90, 210, 360),
        (360, 160, 540, 340),
        (290, 290, 510, 430),
        (40, 40, 130, 90),
    ]
    return img, path, gt_boxes, "synthetic_shapes"


# ===========================================================================
# compute_iou
# ===========================================================================
def compute_iou(box_a, box_b):
    """Calcule l'IoU entre deux boîtes (identique aux Jours 1 et 2).

    Fonction autonome :
        Cette fonction est délibérément dupliquée dans chaque lab pour que
        chaque script reste indépendant et exécutable sans importer les
        modules des autres jours. C'est un choix pédagogique : chaque lab
        doit être self-contained.

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
# run_frcnn
# ===========================================================================
def run_frcnn(img, model, score_thresh=0.25):
    """Exécute Faster R-CNN sur une image.

    Notes techniques :
        - La conversion BGR -> RGB est essentielle car OpenCV lit en BGR
          mais torchvision attend du RGB. Sans cette conversion, les couleurs
          seraient inversées et les performances du modèle dégradées.
        - Le tenseur doit être au format (C, H, W) avec valeurs dans [0, 1].
        - torch.no_grad() désactive le gradient pour l'inférence.

    Paramètres
    ----------
    img : np.ndarray BGR
    model : torchvision Faster R-CNN
    score_thresh : float
        Seuil de confiance minimal.

    Retourne
    --------
    boxes : np.ndarray (N, 4)
    scores : np.ndarray (N,)
    labels : np.ndarray (N,)
    """
    # Conversion OpenCV (HWC, BGR) -> PyTorch (CHW, RGB) -> float [0,1].
    tensor = (
        torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        .permute(2, 0, 1)
        .float()
        / 255.0
    )
    with torch.no_grad():
        pred = model([tensor])[0]

    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()

    # Filtrage manuel par score (le modèle applique déjà un seuil interne,
    # mais on renforce ici pour être explicite).
    mask = scores >= score_thresh
    return boxes[mask], scores[mask], labels[mask]


# ===========================================================================
# run_yolo
# ===========================================================================
def run_yolo(img, model, conf_thresh=0.25):
    """Exécute YOLOv8n sur une image.

    Différence clé avec Faster R-CNN :
        YOLO accepte directement une image NumPy OpenCV (BGR) sans conversion
        de format. Ultralytics gère la normalisation en interne.

    Paramètres
    ----------
    img : np.ndarray BGR
    model : ultralytics.YOLO
    conf_thresh : float
        Seuil de confiance (identique à score_thresh pour Faster R-CNN).

    Retourne
    --------
    boxes : np.ndarray (N, 4)
    scores : np.ndarray (N,)
    labels : np.ndarray (N,)
    """
    # verbose=False réduit les logs dans la console.
    results = model(img, conf=conf_thresh, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy().astype(int)
    return boxes, scores, labels


# ===========================================================================
# benchmark
# ===========================================================================
def benchmark(name, run_fn, img, num_runs=5):
    """Mesure le temps d'inférence moyen et l'écart-type.

    Pourquoi plusieurs runs ?
        Le premier appel peut inclure des temps d'initialisation (cache CPU,
        allocation mémoire). En répétant l'inférence plusieurs fois, on
        obtient une mesure plus stable et représentative.

    Paramètres
    ----------
    name : str
        Nom du détecteur (pour l'affichage).
    run_fn : callable
        Fonction d'inférence (sans argument, img est capturée via closure).
    img : np.ndarray
    num_runs : int
        Nombre de répétitions.

    Retourne
    --------
    dict {"mean": float, "std": float}
        Temps moyen et écart-type en secondes.
    """
    times = []
    for _ in range(num_runs):
        start = time.time()
        run_fn(img)
        times.append(time.time() - start)
    return {"mean": float(np.mean(times)), "std": float(np.std(times))}


# ===========================================================================
# precision_recall_at_threshold
# ===========================================================================
def precision_recall_at_threshold(boxes, scores, gt_boxes, score_thresh, iou_thresh=0.5):
    """Calcule précision et rappel pour un seuil de score donné.

    Principe (identique au Jour 2) :
        TP = prédiction avec IoU >= iou_thresh avec une GT non encore matchée.
        FP = prédiction sans GT correspondante.
        FN = GT non matchée par aucune prédiction.

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
    selected = [(box, score) for box, score in zip(boxes, scores) if score >= score_thresh]
    matched_gt = set()
    tp = 0
    fp = 0

    for pred_box, _ in selected:
        best_iou = 0.0
        best_idx = None
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
    """Calcule une approximation pédagogique de l'AP (Average Precision).

    Méthode :
        1. Pour chaque valeur de rappel unique, on prend la précision max.
        2. L'AP est la somme des aires des rectangles sous la courbe
           précision-rappel ainsi lissée.

    Attention pédagogique :
        Ce n'est ni l'AP officielle COCO (101 points, IoU multiple), ni
        l'AP Pascal VOC (interpolation à 11 points). C'est une simplification
        pour illustrer le PRINCIPE de l'aire sous la courbe.

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
# draw_detections
# ===========================================================================
def draw_detections(img, boxes, scores, labels, color, title):
    """Dessine les détections sur une copie de l'image.

    Paramètres
    ----------
    img : np.ndarray BGR
    boxes : np.ndarray (N, 4)
    scores : np.ndarray (N,)
    labels : np.ndarray (N,)  (non affichés pour éviter la surcharge)
    color : tuple BGR
        Couleur des rectangles (ex: (255, 0, 0) pour rouge).
    title : str
        Texte affiché en haut à gauche.

    Retourne
    --------
    np.ndarray
        Image annotée.
    """
    out = img.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out, f"{score:.2f}", (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )
    # Titre général en haut à gauche.
    cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return out


# ===========================================================================
# best_ious
# ===========================================================================
def best_ious(pred_boxes, gt_boxes):
    """Pour chaque objet GT, trouve le meilleur IoU parmi toutes les prédictions.

    Cette métrique simple est très lisible pour comparer deux détecteurs :
    pour chaque objet réel, on regarde si au moins une prédiction le
    recouvre bien.

    Paramètres
    ----------
    pred_boxes : np.ndarray (N, 4)
    gt_boxes : list[tuple]

    Retourne
    --------
    list[float]
        Meilleur IoU pour chaque GT. Longueur = len(gt_boxes).
    """
    ious_per_gt = []
    for gt_box in gt_boxes:
        best_iou = 0
        for pred_box in pred_boxes:
            iou_val = compute_iou(
                (pred_box[0], pred_box[1], pred_box[2], pred_box[3]), gt_box
            )
            best_iou = max(best_iou, iou_val)
        ious_per_gt.append(best_iou)
    return ious_per_gt


# ===========================================================================
# main — Fonction principale
# ===========================================================================
def main():
    """Orchestre le lab comparatif complet du Jour 3.

    Déroulement :
        1.  Chargement des deux modèles (Faster R-CNN + YOLOv8n).
        2.  Chargement de l'image de test (réelle si dispo, sinon synthétique).
        3.  Benchmark : temps d'inférence moyen des deux détecteurs.
        4.  Calcul de l'IoU moyen par objet pour chaque détecteur.
        5.  Visualisations :
            - speed_comparison.png    (barres de vitesse)
            - iou_comparison.png      (IoU par objet GT)
            - detection_overlay.png   (détections côte à côte)
            - threshold_sweep.png     (impact du seuil de confiance)
        6.  Sweep de seuil de confiance pour analyser le compromis.
        7.  Calcul d'un mAP@0.5 simplifié.
        8.  Sauvegarde du rapport JSON complet avec toutes les métriques.
    """
    set_reproducible_seed()
    os.makedirs("outputs/jour3/figures", exist_ok=True)

    # ======================================================================
    # CHARGEMENT DES MODÈLES
    # ======================================================================
    print("Chargement des modèles...")

    # Faster R-CNN avec ResNet-50 FPN, pré-entraîné COCO.
    # box_score_thresh=0.25 filtre les détections les moins confiantes.
    model_frcnn = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        box_score_thresh=0.25,
    )
    model_frcnn.eval()

    # YOLOv8n : version nano (~3M paramètres), rapide sur CPU.
    # Les poids sont téléchargés automatiquement au premier appel.
    model_yolo = YOLO("yolov8n.pt")

    # ======================================================================
    # CHARGEMENT DE L'IMAGE DE TEST
    # ======================================================================
    img_path = "labs/jour3/test_image.png"
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    img, img_path, gt_boxes, image_source = load_detection_image(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"Image de test : {img_path} ({img.shape[1]}x{img.shape[0]}, {image_source})")

    # ======================================================================
    # PARTIE 1 : COMPARAISON DE VITESSE
    # ======================================================================
    print("\nBenchmark Faster R-CNN...")

    # Le benchmark et l'inférence finale sont séparés pour éviter que le
    # benchmark n'affecte les résultats de l'inférence utilisée pour les
    # métriques.
    frcnn_time = benchmark(
        "Faster R-CNN", lambda im: run_frcnn(im, model_frcnn), img, num_runs=3
    )
    frcnn_boxes, frcnn_scores, frcnn_labels = run_frcnn(img, model_frcnn)
    print(f"  Temps : {frcnn_time['mean']:.3f}s ± {frcnn_time['std']:.3f}s")
    print(f"  Détections : {len(frcnn_boxes)}")

    print("\nBenchmark YOLOv8n...")
    yolo_time = benchmark(
        "YOLOv8n", lambda im: run_yolo(im, model_yolo), img, num_runs=3
    )
    yolo_boxes, yolo_scores, yolo_labels = run_yolo(img, model_yolo)
    print(f"  Temps : {yolo_time['mean']:.3f}s ± {yolo_time['std']:.3f}s")
    print(f"  Détections : {len(yolo_boxes)}")

    # Ratio de vitesse (combien de fois YOLO est plus rapide).
    speedup = frcnn_time["mean"] / yolo_time["mean"] if yolo_time["mean"] > 0 else float("inf")
    print(f"\nYOLOv8n est {speedup:.1f}x plus rapide que Faster R-CNN")

    # ---- Figure 1 : Comparaison de vitesse (barres) ----
    plt.figure(figsize=(8, 4))
    models = ["Faster R-CNN", "YOLOv8n"]
    times = [frcnn_time["mean"], yolo_time["mean"]]
    stds = [frcnn_time["std"], yolo_time["std"]]
    colors = ["steelblue", "coral"]
    bars = plt.bar(models, times, yerr=stds, color=colors, capsize=5, width=0.5)
    plt.ylabel("Temps d'inférence (s)")
    plt.title(f"Comparaison de vitesse (x{speedup:.1f} plus rapide avec YOLO)")
    plt.grid(axis="y", alpha=0.3)
    for bar, t in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{t:.3f}s",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    plt.tight_layout()
    plt.savefig("outputs/jour3/figures/speed_comparison.png", dpi=130)
    plt.close()

    # ======================================================================
    # PARTIE 2 : COMPARAISON IoU
    # ======================================================================
    frcnn_ious = best_ious(frcnn_boxes, gt_boxes)
    yolo_ious = best_ious(yolo_boxes, gt_boxes)

    print(f"\nIoU par GT (Faster R-CNN) : {[f'{i:.3f}' for i in frcnn_ious]}")
    print(f"IoU par GT (YOLOv8n)        : {[f'{i:.3f}' for i in yolo_ious]}")
    print(f"IoU moyen Faster R-CNN : {np.mean(frcnn_ious):.3f}")
    print(f"IoU moyen YOLOv8n        : {np.mean(yolo_ious):.3f}")

    # ---- Figure 2 : Comparaison IoU par objet GT ----
    plt.figure(figsize=(8, 4))
    x = np.arange(len(gt_boxes))
    width = 0.35
    plt.bar(x - width / 2, frcnn_ious, width, label="Faster R-CNN", color="steelblue")
    plt.bar(x + width / 2, yolo_ious, width, label="YOLOv8n", color="coral")
    plt.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="Seuil 0.5")
    plt.xlabel("Objet GT")
    plt.ylabel("Meilleur IoU")
    plt.title("Comparaison IoU par objet")
    plt.xticks(x, [f"GT {i+1}" for i in range(len(gt_boxes))])
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/jour3/figures/iou_comparison.png", dpi=130)
    plt.close()

    # ======================================================================
    # PARTIE 3 : VISUALISATION CÔTE À CÔTE
    # ======================================================================
    # On affiche les détections des deux modèles côte à côte pour une
    # comparaison visuelle directe.
    frcnn_vis = draw_detections(
        img, frcnn_boxes, frcnn_scores, frcnn_labels,
        (255, 0, 0), f"Faster R-CNN ({len(frcnn_boxes)} detections)",
    )
    yolo_vis = draw_detections(
        img, yolo_boxes, yolo_scores, yolo_labels,
        (0, 255, 0), f"YOLOv8n ({len(yolo_boxes)} detections)",
    )

    combined = np.hstack([frcnn_vis, yolo_vis])
    overlay_path = "outputs/jour3/figures/detection_overlay.png"
    cv2.imwrite(overlay_path, combined)
    print(f"\nVisualisation sauvegardée : {overlay_path}")

    # ======================================================================
    # PARTIE 4 : VARIATION DU SEUIL DE CONFIANCE (THRESHOLD SWEEP)
    # ======================================================================
    # On fait varier le seuil de confiance de YOLOv8n pour observer l'impact
    # sur le nombre de détections et l'IoU moyen.
    print("\nOptimisation — Variation du seuil de confiance (YOLOv8n)...")
    thresholds = [0.1, 0.25, 0.5, 0.75]
    threshold_results = []
    for thresh in thresholds:
        boxes, scores, labels = run_yolo(img, model_yolo, conf_thresh=thresh)
        ious = best_ious(boxes, gt_boxes)
        avg_iou = float(np.mean(ious)) if ious else 0.0
        pr = precision_recall_at_threshold(boxes, scores, gt_boxes, thresh)
        threshold_results.append({
            "threshold": thresh,
            "num_detections": len(boxes),
            "avg_iou": round(avg_iou, 4),
            "precision": round(float(pr["precision"]), 4),
            "recall": round(float(pr["recall"]), 4),
        })
        print(
            f"  Seuil {thresh}: {len(boxes)} détections, "
            f"IoU moyen={avg_iou:.3f}"
        )

    # AP simplifiée à partir du sweep.
    yolo_ap50 = average_precision_from_pr_rows(threshold_results)

    # ---- Figure 4 : Sweep de seuil (double axe Y) ----
    plt.figure(figsize=(8, 4))
    xs = [row["threshold"] for row in threshold_results]
    counts = [row["num_detections"] for row in threshold_results]
    avg_ious = [row["avg_iou"] for row in threshold_results]

    ax1 = plt.gca()
    ax1.plot(xs, counts, marker="o", color="steelblue", label="Détections")
    ax1.set_xlabel("Seuil de confiance")
    ax1.set_ylabel("Nombre de détections", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, avg_ious, marker="s", color="coral", label="IoU moyen")
    ax2.set_ylabel("IoU moyen", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")
    ax2.set_ylim(-0.05, 1.05)

    plt.title("Impact du seuil de confiance YOLOv8n")
    plt.tight_layout()
    threshold_path = "outputs/jour3/figures/threshold_sweep.png"
    plt.savefig(threshold_path, dpi=130)
    plt.close()
    print(f"  Courbe de seuil sauvegardée : {threshold_path}")

    # ======================================================================
    # RAPPORT JSON FINAL
    # ======================================================================
    metrics = {
        "speed": {
            "faster_rcnn_mean_s": round(frcnn_time["mean"], 4),
            "yolov8n_mean_s": round(yolo_time["mean"], 4),
            "speedup_yolo_vs_frcnn": round(speedup, 2),
        },
        "image_source": image_source,
        "gt_boxes": [list(box) for box in gt_boxes],
        "iou": {
            "faster_rcnn_avg": round(float(np.mean(frcnn_ious)), 4),
            "yolov8n_avg": round(float(np.mean(yolo_ious)), 4),
            "faster_rcnn_per_gt": [round(float(i), 4) for i in frcnn_ious],
            "yolov8n_per_gt": [round(float(i), 4) for i in yolo_ious],
        },
        "num_detections": {
            "faster_rcnn": int(len(frcnn_boxes)),
            "yolov8n": int(len(yolo_boxes)),
        },
        "threshold_sweep": threshold_results,
        "map50_simplified": round(float(yolo_ap50), 4),
        "metric_note": "mAP@0.5 simplifié pour usage pédagogique sur une seule classe et quelques seuils, pas une évaluation COCO officielle.",
        "figures": {
            "speed_comparison": "outputs/jour3/figures/speed_comparison.png",
            "iou_comparison": "outputs/jour3/figures/iou_comparison.png",
            "detection_overlay": overlay_path,
            "threshold_sweep": threshold_path,
        },
    }

    with open("outputs/jour3/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMétriques sauvegardées : outputs/jour3/metrics.json")
    print(json.dumps(metrics, indent=2))


# ===========================================================================
# Point d'entrée
# ===========================================================================
if __name__ == "__main__":
    main()
