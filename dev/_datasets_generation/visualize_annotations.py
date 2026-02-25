
import os
import json
import cv2
from tqdm import tqdm


# CONFIGURATION

DATASET_ROOT = "out_dataset"

IMAGE_DIR = os.path.join(DATASET_ROOT, "images")
ANN_FILE = os.path.join(DATASET_ROOT, "annotations.json")

OUTPUT_DIR = os.path.join(DATASET_ROOT, "images_verif")

# Couleurs (BGR pour OpenCV)
COLOR_LEAF = (0, 0, 255)        # Rouge
COLOR_INTERNAL = (255, 0, 0)    # Bleu
COLOR_CORNER = (0, 255, 255)  # Jaune

POINT_RADIUS = 4
POINT_THICKNESS = -1            # Cercle plein


def visualize_annotations():
    if not os.path.exists(ANN_FILE):
        print(f"Fichier annotations introuvable : {ANN_FILE}")
        return

    if not os.path.exists(IMAGE_DIR):
        print(f"Dossier images introuvable : {IMAGE_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(ANN_FILE, "r") as f:
        coco = json.load(f)

    annotations_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        annotations_by_image.setdefault(img_id, []).append(ann)

    images_by_id = {
        img["id"]: img
        for img in coco["images"]
    }

    print("=" * 80)
    print("VISUALISATION DES ANNOTATIONS")
    print("=" * 80)
    print(f"Images source  : {IMAGE_DIR}")
    print(f"Images output  : {OUTPUT_DIR}\n")

    for img in tqdm(coco["images"], desc="Visualisation"):

        img_id = img["id"]
        img_name = img["file_name"]

        img_path = os.path.join(IMAGE_DIR, img_name)
        img_data = cv2.imread(img_path)

        if img_data is None:
            print(f"Image listée dans COCO mais absente du disque : {img_name}")
            continue

        anns = annotations_by_image.get(img_id, [])

        for ann in anns:
            category_id = ann["category_id"]
            bbox = ann["bbox"]

            x = int(bbox[0] + bbox[2] / 2)
            y = int(bbox[1] + bbox[3] / 2)

            if category_id == 1:
                color = COLOR_LEAF
            elif category_id == 2:
                color = COLOR_INTERNAL
            elif category_id == 3:
                color = COLOR_CORNER
            else:
                continue

            cv2.circle(
                img_data,
                (x, y),
                POINT_RADIUS,
                color,
                POINT_THICKNESS
            )

        out_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(out_path, img_data)

    print("\n" + "=" * 80)
    print("VISUALISATION TERMINÉE")
    print("=" * 80)
    print(f"\nImages annotées disponibles dans : {OUTPUT_DIR}\n")


if __name__ == "__main__":
    visualize_annotations()
