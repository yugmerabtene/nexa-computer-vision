"""
Préparation des images de test pour le lab du Jour 1
=====================================================

Objectif :
    Générer (ou télécharger) les images nécessaires au bon déroulement du
    lab Jour 1. Ces images servent de support visuel pour illustrer les
    concepts du cours : lecture OpenCV, histogramme, égalisation, seuillage,
    contours, extraction de boîtes.

Stratégie :
    1. Essayer de télécharger une image publique simple (Wikipedia).
    2. Si le téléchargement échoue (pas de réseau, proxy, etc.), générer
       une scène synthétique localement.
    3. Générer systématiquement une image à faible contraste pour illustrer
       l'égalisation d'histogramme (equalizeHist).

Pourquoi deux sources ?
    - L'image réelle téléchargée est plus motivante pour l'étudiant.
    - L'image synthétique garantit que le lab fonctionne même hors ligne.
    - L'image low-contrast est dédiée à une démonstration pédagogique
      spécifique (amélioration de contraste).

Compatibilité :
    Les images générées sont identiques d'une machine à l'autre grâce à
    une graine aléatoire fixe (RNG_SEED = 42). Cela facilite la correction
    et la comparaison des résultats entre étudiants.
"""

from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Graine aléatoire globale
# ---------------------------------------------------------------------------
# En fixant RNG_SEED, on garantit que les images synthétiques sont
# identiques à chaque exécution, sur n'importe quelle machine.
RNG_SEED = 42


# ---------------------------------------------------------------------------
# download_test_image
# ---------------------------------------------------------------------------
def download_test_image() -> str:
    """Tente de télécharger une image d'exemple depuis Wikimedia Commons.

    Retourne
    --------
    str
        Chemin vers l'image téléchargée, ou chaîne vide si échec.

    Notes
    -----
    - L'image est optionnelle : le cours ne dépend pas d'elle.
    - L'échec est silencieux (pas de stack trace) pour ne pas dérouter
      l'étudiant. La fonction appelante doit utiliser l'opérateur `or`
      pour basculer vers la génération locale.
    - L'image choisie (PNG transparency demonstration) contient des zones
      transparentes, utiles pour discuter des canaux alpha même si le lab
      utilise principalement BGR.
    """
    url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
        "PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    )
    out = Path(__file__).parent / "assets" / "test_image.png"

    try:
        import urllib.request

        urllib.request.urlretrieve(url, str(out))

        # Vérifier que l'image est bien lisible par OpenCV.
        img = cv2.imread(str(out))
        if img is not None:
            print(f"Downloaded test image: {out}")
            return str(out)
    except Exception:
        # Échec silencieux : la génération locale prend le relais.
        pass

    return ""


# ---------------------------------------------------------------------------
# generate_test_image
# ---------------------------------------------------------------------------
def generate_test_image() -> str:
    """Crée une scène synthétique réaliste avec formes, texte et bruit.

    Contenu de l'image générée :
        - Fond dégradé (non uniforme) pour que l'histogramme ne soit pas
          trivial.
        - Grand rectangle blanc : facile à segmenter par seuillage.
        - Cercle orange : forme alternative pour HOG/SIFT et contours.
        - Rectangle orange (bords seulement) : illustration de contours.
        - Ellipse verte : forme asymétrique pour varier les descripteurs.
        - Texte "CV Lab" : ajoute des bords fins et de la texture.
        - Bruit gaussien léger : rend l'image moins artificielle.

    Paramètres
    ----------
    Aucun (configuration interne).

    Retourne
    --------
    str
        Chemin vers l'image générée.

    Notes
    -----
    La graine RNG_SEED = 42 garantit la reproductibilité.
    """
    rng = np.random.default_rng(RNG_SEED)

    # Image BGR de 300x400 pixels.
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # Fond en dégradé vertical : les intensités varient de [20,20,40] en haut
    # à [50,35,85] en bas. Cela produit un histogramme non trivial.
    for i in range(300):
        img[i, :] = [int(20 + 0.1 * i), int(20 + 0.05 * i), int(40 + 0.15 * i)]

    # Grand rectangle blanc (objet principal).
    cv2.rectangle(img, (30, 40), (180, 200), (255, 255, 255), -1)

    # Cercle orange (forme courbe pour varier).
    cv2.circle(img, (300, 150), 60, (255, 200, 0), -1)

    # Rectangle orange avec bordure seulement (épaisseur = 3).
    cv2.rectangle(img, (200, 20), (260, 70), (0, 165, 255), 3)

    # Ellipse verte pleine.
    cv2.ellipse(img, (100, 250), (50, 30), 0, 0, 360, (128, 255, 128), -1)

    # Texte : crée des bords fins (utile pour les descripteurs).
    cv2.putText(img, "CV Lab", (220, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Bruit additif léger (0-20) pour un aspect plus naturel.
    noise = rng.integers(0, 20, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    out = Path(__file__).parent / "assets" / "test_scene.png"
    cv2.imwrite(str(out), img)
    print(f"Generated test image: {out}")
    return str(out)


# ---------------------------------------------------------------------------
# generate_low_contrast_image
# ---------------------------------------------------------------------------
def generate_low_contrast_image() -> str:
    """Crée une image sombre et peu contrastée pour démontrer equalizeHist.

    Objectif pédagogique :
        Cette image est délibérément difficile à analyser visuellement.
        Tous les rectangles ont des niveaux de gris très proches (30 à 70),
        ce qui rend les détails presque invisibles.
        L'égalisation d'histogramme (equalizeHist) doit permettre de faire
        ressortir les structures cachées.

    Paramètres
    ----------
    Aucun.

    Retourne
    --------
    str
        Chemin vers l'image générée.

    Notes
    -----
    La graine est RNG_SEED + 1 pour obtenir un tirage différent de celui
    de generate_test_image().
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    img = np.zeros((200, 300, 3), dtype=np.uint8)

    # 5 rectangles avec des intensités entre 30 et 70 (sur 255).
    # Ces valeurs sont très proches les unes des autres -> faible contraste.
    for _ in range(5):
        x1 = int(rng.integers(10, 200))
        y1 = int(rng.integers(10, 150))
        x2 = int(rng.integers(x1 + 20, 280))
        y2 = int(rng.integers(y1 + 20, 180))
        val = int(rng.integers(30, 70))
        cv2.rectangle(img, (x1, y1), (x2, y2), (val, val, val), -1)

    # Bruit très faible pour rester dans la plage basse.
    noise = rng.integers(0, 10, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    out = Path(__file__).parent / "assets" / "test_low_contrast.png"
    cv2.imwrite(str(out), img)
    print(f"Generated low-contrast image: {out}")
    return str(out)


# ===========================================================================
# Point d'entrée
# ===========================================================================
if __name__ == "__main__":
    # S'assurer que le dossier assets/ existe.
    (Path(__file__).parent / "assets").mkdir(parents=True, exist_ok=True)

    # Téléchargement d'une image réelle, ou génération locale si échec.
    # L'opérateur `or` court-circuite : si download() réussit, on saute
    # generate().
    download_test_image() or generate_test_image()

    # L'image low-contrast est toujours générée (pas de téléchargement).
    generate_low_contrast_image()

    print("All test images ready in labs/jour1/assets/")
