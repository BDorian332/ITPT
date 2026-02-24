"""
Ajout de bruit sur un dataset COCO d'arbres phylogénétiques.

Ce script prend un dataset COCO en entrée et génère une version bruitée en sortie.
Les annotations restent identiques, seules les images sont modifiées.

Structure du dataset en entrée:
  <INPUT_DATASET>/
    images/
      *.png (ou jpg)
    annotations.json

Structure du dataset en sortie:
  <OUTPUT_DATASET>/
    images/            (copies bruitées)
    annotations.json   (copié tel quel)

Types de bruit appliqués:
  1. Formes noires (cercle/carré/triangle) sur les nœuds internes
  2. Bruit poivre (pixels noirs épars)
  3. Texte aléatoire avec rotation

Usage:
  python add_noise_coco_dataset.py
  python add_noise_coco_dataset.py --input_dataset out_dataset --output_dataset out_dataset_noisy
"""

import os
import json
import shutil
import random
import argparse
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np

# ==============================================================================
# PARAMÈTRES DE CONFIGURATION
# ==============================================================================

# --- Chemins des datasets (modifiables via ligne de commande) ---
INPUT_DATASET = "out_dataset"           # Dossier du dataset d'entrée
OUTPUT_DATASET = "out_dataset_noisy"    # Dossier du dataset bruité en sortie

# --- Catégorie COCO des nœuds à corrompre ---
INTERNAL_NODE_CATEGORY_ID = 1           # ID de catégorie pour les nœuds internes

# --- Formes noires sur les nœuds ---
PCT_NODES_WITH_SHAPE = 0.20             # Pourcentage de nœuds avec une forme noire (0-1)
SHAPE_SIZE_MIN_PX = 6                   # Taille minimale des formes (pixels)
SHAPE_SIZE_MAX_PX = 10                  # Taille maximale des formes (pixels)

# --- Bruit poivre (pixels noirs épars) ---
PCT_IMAGES_WITH_PEPPER = 0.50           # Pourcentage d'images avec bruit poivre (0-1)
PEPPER_AMOUNT_MIN = 0.001               # Fraction minimale de pixels touchés
PEPPER_AMOUNT_MAX = 0.010               # Fraction maximale de pixels touchés
PEPPER_DILATE_K_MIN = 1                 # Taille minimale du kernel de dilatation
PEPPER_DILATE_K_MAX = 3                 # Taille maximale du kernel de dilatation
PEPPER_DILATE_ITERS_MIN = 1             # Nombre minimal d'itérations de dilatation
PEPPER_DILATE_ITERS_MAX = 2             # Nombre maximal d'itérations de dilatation

# --- Texte aléatoire ---
PCT_IMAGES_WITH_TEXT = 0.50             # Pourcentage d'images avec texte (0-1)
TEXT_MIN_LEN = 3                        # Longueur minimale d'un mot
TEXT_MAX_LEN = 12                       # Longueur maximale d'un mot
TEXT_MIN_WORDS = 3                      # Nombre minimal de mots par image
TEXT_MAX_WORDS = 12                     # Nombre maximal de mots par image
TEXT_FONT_SCALE_MIN = 0.4               # Échelle minimale de la police
TEXT_FONT_SCALE_MAX = 1.1               # Échelle maximale de la police
TEXT_THICKNESS_MIN = 1                  # Épaisseur minimale du texte
TEXT_THICKNESS_MAX = 2                  # Épaisseur maximale du texte
TEXT_ROT_MIN_DEG = -45                  # Rotation minimale (degrés)
TEXT_ROT_MAX_DEG = 45                   # Rotation maximale (degrés)
TEXT_COLOR_MIN = 0                      # Couleur minimale (noir)
TEXT_COLOR_MAX = 40                     # Couleur maximale (gris foncé)
TEXT_MARGIN_PX = 10                     # Marge autour du texte (pixels)

# --- Graine aléatoire ---
SEED = 42                               # Graine pour reproductibilité (None = aléatoire)

# --- Extensions d'images supportées ---
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def ensure_dir(path: str) -> None:
    """Crée un répertoire s'il n'existe pas."""
    os.makedirs(path, exist_ok=True)


def clamp_int(v: int, lo: int, hi: int) -> int:
    """Restreint une valeur entière entre lo et hi."""
    return max(lo, min(hi, v))


def rand_text_word(n: int) -> str:
    """Génère un mot aléatoire de n caractères alphanumériques."""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(random.choice(alphabet) for _ in range(n))


def maybe(p: float) -> bool:
    """Retourne True avec une probabilité p."""
    return random.random() < p


def load_coco(path: str) -> Dict[str, Any]:
    """Charge un fichier COCO JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_coco(obj: Dict[str, Any], path: str) -> None:
    """Sauvegarde un objet COCO en JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def draw_filled_shape(img_bgr: np.ndarray, cx: int, cy: int, size_px: int) -> None:
    """
    Dessine une forme noire remplie (cercle, carré ou triangle) centrée sur (cx, cy).
    """
    h, w = img_bgr.shape[:2]
    cx = clamp_int(cx, 0, w - 1)
    cy = clamp_int(cy, 0, h - 1)
    size_px = max(1, int(size_px))

    shape_type = random.choice(("circle", "square", "triangle"))
    color = (0, 0, 0)

    if shape_type == "circle":
        cv2.circle(img_bgr, (cx, cy), size_px, color, -1)
        return

    if shape_type == "square":
        x1 = clamp_int(cx - size_px, 0, w - 1)
        y1 = clamp_int(cy - size_px, 0, h - 1)
        x2 = clamp_int(cx + size_px, 0, w - 1)
        y2 = clamp_int(cy + size_px, 0, h - 1)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, -1)
        return

    ang = random.uniform(0, 2 * np.pi)
    r = float(size_px)
    pts = []
    for k in range(3):
        a = ang + k * (2 * np.pi / 3.0)
        x = int(round(cx + r * np.cos(a)))
        y = int(round(cy + r * np.sin(a)))
        pts.append([clamp_int(x, 0, w - 1), clamp_int(y, 0, h - 1)])
    pts_np = np.array([pts], dtype=np.int32)
    cv2.fillPoly(img_bgr, pts_np, color)


def apply_shapes_on_internal_nodes(img_bgr: np.ndarray, anns_for_image: List[Dict[str, Any]]) -> None:
    """
    Dessine des formes noires sur un sous-ensemble de nœuds internes.
    """
    internal_anns = [a for a in anns_for_image if int(a.get("category_id", -1)) == INTERNAL_NODE_CATEGORY_ID]
    if not internal_anns:
        return

    k = int(round(PCT_NODES_WITH_SHAPE * len(internal_anns)))
    if k <= 0:
        return

    chosen = random.sample(internal_anns, k=min(k, len(internal_anns)))
    for ann in chosen:
        bbox = ann.get("bbox", None)
        if not bbox or len(bbox) != 4:
            continue
        x, y, bw, bh = bbox
        cx = int(round(x + bw / 2.0))
        cy = int(round(y + bh / 2.0))

        size_px = random.randint(SHAPE_SIZE_MIN_PX, SHAPE_SIZE_MAX_PX)
        draw_filled_shape(img_bgr, cx, cy, size_px)

# ==============================================================================
# BRUIT : POIVRE (PIXELS NOIRS)
# ==============================================================================

def add_pepper(img_bgr: np.ndarray) -> None:
    """
    Ajoute du bruit poivre (pixels noirs épars) sur l'image.
    Les pixels sont optionnellement épaissis via dilatation.
    Modifie l'image sur place.
    """
    h, w = img_bgr.shape[:2]
    amount = random.uniform(PEPPER_AMOUNT_MIN, PEPPER_AMOUNT_MAX)
    num = int(amount * h * w)
    if num <= 0:
        return

    mask = np.zeros((h, w), dtype=np.uint8)
    ys = np.random.randint(0, h, size=num)
    xs = np.random.randint(0, w, size=num)
    mask[ys, xs] = 255

    k = random.randint(PEPPER_DILATE_K_MIN, PEPPER_DILATE_K_MAX)
    iters = random.randint(PEPPER_DILATE_ITERS_MIN, PEPPER_DILATE_ITERS_MAX)
    if k > 1 and iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=iters)

    img_bgr[mask > 0] = (0, 0, 0)

# ==============================================================================
# BRUIT : TEXTE ALÉATOIRE AVEC ROTATION
# ==============================================================================

def paste_rgba(dst_bgr: np.ndarray, patch_bgr: np.ndarray, patch_alpha: np.ndarray, x0: int, y0: int) -> None:
    """
    Colle un patch avec canal alpha sur l'image de destination à la position (x0, y0).
    Modifie l'image sur place.
    """
    h, w = dst_bgr.shape[:2]
    ph, pw = patch_bgr.shape[:2]

    x1 = clamp_int(x0, 0, w)
    y1 = clamp_int(y0, 0, h)
    x2 = clamp_int(x0 + pw, 0, w)
    y2 = clamp_int(y0 + ph, 0, h)

    if x2 <= x1 or y2 <= y1:
        return

    roi = dst_bgr[y1:y2, x1:x2]
    px1 = x1 - x0
    py1 = y1 - y0
    px2 = px1 + (x2 - x1)
    py2 = py1 + (y2 - y1)

    pb = patch_bgr[py1:py2, px1:px2].astype(np.float32)
    pa = patch_alpha[py1:py2, px1:px2].astype(np.float32) / 255.0
    pa = pa[..., None]

    out = roi.astype(np.float32) * (1.0 - pa) + pb * pa
    roi[:, :] = np.clip(out, 0, 255).astype(np.uint8)


def make_rotated_text_patch(text: str, font_scale: float, thickness: int, rot_deg: float, color_bgr: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crée un patch contenant du texte avec rotation sur fond transparent.
    Retourne (patch_bgr, patch_alpha).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 8
    pw = max(1, tw + 2 * pad)
    ph = max(1, th + 2 * pad + baseline)

    patch = np.full((ph, pw, 3), 255, dtype=np.uint8)
    org = (pad, ph - pad - baseline)
    cv2.putText(patch, text, org, font, font_scale, color_bgr, thickness, cv2.LINE_AA)

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    alpha = (gray < 250).astype(np.uint8) * 255

    center = (pw / 2.0, ph / 2.0)
    M = cv2.getRotationMatrix2D(center, rot_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int(ph * sin + pw * cos)
    nh = int(ph * cos + pw * sin)
    M[0, 2] += (nw / 2.0) - center[0]
    M[1, 2] += (nh / 2.0) - center[1]

    rot_patch = cv2.warpAffine(patch, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    rot_alpha = cv2.warpAffine(alpha, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)

    rot_gray = cv2.cvtColor(rot_patch, cv2.COLOR_BGR2GRAY)
    rot_alpha[rot_gray >= 250] = 0

    return rot_patch, rot_alpha


def add_random_text(img_bgr: np.ndarray) -> None:
    """
    Ajoute plusieurs textes aléatoires avec rotation à des positions aléatoires.
    Modifie l'image sur place.
    """
    h, w = img_bgr.shape[:2]
    n_words = random.randint(TEXT_MIN_WORDS, TEXT_MAX_WORDS)

    for _ in range(n_words):
        ln = random.randint(TEXT_MIN_LEN, TEXT_MAX_LEN)
        word = rand_text_word(ln)

        font_scale = random.uniform(TEXT_FONT_SCALE_MIN, TEXT_FONT_SCALE_MAX)
        thickness = random.randint(TEXT_THICKNESS_MIN, TEXT_THICKNESS_MAX)
        rot_deg = random.uniform(TEXT_ROT_MIN_DEG, TEXT_ROT_MAX_DEG)

        c = random.randint(TEXT_COLOR_MIN, TEXT_COLOR_MAX)
        color = (c, c, c)

        patch_bgr, patch_alpha = make_rotated_text_patch(word, font_scale, thickness, rot_deg, color)

        ph, pw = patch_bgr.shape[:2]
        if pw + 2 * TEXT_MARGIN_PX >= w or ph + 2 * TEXT_MARGIN_PX >= h:
            x0 = random.randint(0, max(0, w - pw))
            y0 = random.randint(0, max(0, h - ph))
        else:
            x0 = random.randint(TEXT_MARGIN_PX, max(TEXT_MARGIN_PX, w - pw - TEXT_MARGIN_PX))
            y0 = random.randint(TEXT_MARGIN_PX, max(TEXT_MARGIN_PX, h - ph - TEXT_MARGIN_PX))

        paste_rgba(img_bgr, patch_bgr, patch_alpha, x0, y0)

# ==============================================================================
# FONCTION PRINCIPALE
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse les arguments de ligne de commande."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dataset", type=str, default=INPUT_DATASET, help="Chemin du dataset COCO d'entrée")
    ap.add_argument("--output_dataset", type=str, default=OUTPUT_DATASET, help="Chemin du dataset COCO de sortie")
    ap.add_argument("--seed", type=int, default=SEED if SEED is not None else -1, help="Graine aléatoire (-1 = pas de graine)")
    return ap.parse_args()


def main() -> None:
    """
    Fonction principale qui applique du bruit sur un dataset COCO.
    
    Processus:
    1. Charge le dataset COCO d'entrée
    2. Pour chaque image:
       - Applique des formes noires sur les nœuds internes
       - Ajoute du bruit poivre (pixels noirs)
       - Ajoute du texte aléatoire avec rotation
    3. Sauvegarde les images bruitées
    4. Copie le fichier annotations.json tel quel
    """
    args = parse_args()

    if args.seed is not None and args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)

    in_root = args.input_dataset
    out_root = args.output_dataset

    in_images = os.path.join(in_root, "images")
    in_ann = os.path.join(in_root, "annotations.json")

    if not os.path.isdir(in_images):
        raise SystemExit(f"Dossier manquant: {in_images}")
    if not os.path.isfile(in_ann):
        raise SystemExit(f"Fichier manquant: {in_ann}")

    out_images = os.path.join(out_root, "images")
    out_ann = os.path.join(out_root, "annotations.json")
    ensure_dir(out_images)

    coco = load_coco(in_ann)

    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        img_id = int(ann.get("image_id", -1))
        anns_by_img.setdefault(img_id, []).append(ann)

    images = coco.get("images", [])
    if not images:
        raise SystemExit("Le fichier COCO ne contient pas d'images.")

    for info in images:
        img_id = int(info["id"])
        fname = info["file_name"]
        ext = os.path.splitext(fname)[1].lower()

        if ext not in IMG_EXTS:
            pass

        src_path = os.path.join(in_images, fname)
        dst_path = os.path.join(out_images, fname)

        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Impossible de lire: {src_path} (ignoré)")
            continue

        apply_shapes_on_internal_nodes(img, anns_by_img.get(img_id, []))

        if maybe(PCT_IMAGES_WITH_PEPPER):
            add_pepper(img)

        if maybe(PCT_IMAGES_WITH_TEXT):
            add_random_text(img)

        ensure_dir(os.path.dirname(dst_path))
        ok = cv2.imwrite(dst_path, img)
        if not ok:
            print(f"Échec d'écriture: {dst_path}")

    shutil.copy2(in_ann, out_ann)

    print("Terminé.")
    print(f"Entrée : {in_root}")
    print(f"Sortie : {out_root}")
    print("Note   : annotations.json copié sans modification, seules les images ont été modifiées.")

if __name__ == "__main__":
    main()