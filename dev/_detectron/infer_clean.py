import os
import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer

try:
    from correction.corrector import correction as corr_fn
except Exception as e:
    raise SystemExit(f"❌ Impossible d'import correction.corrector.correction: {e}")

CONFIG_PATH  = "output_detectron2/config.yaml"
WEIGHTS_PATH = "output_detectron2/model_final.pth"
DEVICE       = "cuda"
MAX_DETS     = 400


def draw_predictions_on_image(image_path: str, leaves, nodes, corners, out_path: str):
    """
    Dessine uniquement les prédictions (leaves/nodes/corners) sur l'image et sauvegarde.
    Format attendu: List[(x, y, score)] pour chaque classe.
    Couleurs (BGR):
      leaf = vert, node = bleu, corner = rouge
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    col_leaf = (0, 220, 0)
    col_node = (220, 0, 0)
    col_corner = (0, 0, 220)

    def draw_list(pts, color, name):
        for (x, y, s) in pts:
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(img, f"{s:.2f}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        return len(pts)

    n_leaf = draw_list(leaves, col_leaf, "leaf")
    n_node = draw_list(nodes, col_node, "node")
    n_corner = draw_list(corners, col_corner, "corner")

    cv2.putText(img, f"leaf={n_leaf} node={n_node} corner={n_corner}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(img, "leaf=green node=blue corner=red", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)

    ok = cv2.imwrite(out_path, img)
    if not ok:
        raise RuntimeError(f"Failed to write: {out_path}")


def mask_centroid(mask01: np.ndarray):
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def instances_to_points(instances):
    if instances is None or len(instances) == 0:
        return []

    instances = instances.to("cpu")

    boxes   = instances.pred_boxes.tensor.numpy()
    scores  = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    has_masks = instances.has("pred_masks")
    masks = instances.pred_masks.numpy() if has_masks else None

    pts = []
    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes[i]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if masks is not None:
            c = mask_centroid(masks[i].astype(np.uint8))
            if c is not None:
                cx, cy = c

        pts.append((int(classes[i]), float(cx), float(cy), float(scores[i])))

    return pts


_cfg = get_cfg()
_cfg.merge_from_file(CONFIG_PATH)
_cfg.MODEL.WEIGHTS = WEIGHTS_PATH
_cfg.MODEL.DEVICE = DEVICE
_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
_cfg.TEST.DETECTIONS_PER_IMAGE = MAX_DETS

_predictor = DefaultPredictor(_cfg)
DetectionCheckpointer(_predictor.model).load(WEIGHTS_PATH)
_predictor.model.eval()


def infer_image(image_path: str, score_thresh: float):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    outputs = _predictor(img)
    instances = outputs.get("instances", None)

    pts = instances_to_points(instances)

    leaves = []
    internal_nodes = []
    corners = []

    for cls, x, y, score in pts:
        if score < score_thresh:
            continue

        if cls == 0:
            leaves.append((int(x), int(y), score))
        elif cls == 1:
            internal_nodes.append((int(x), int(y), score))
        elif cls == 2:
            corners.append((int(x), int(y), score))

    return leaves, internal_nodes, corners


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--out", default="", help="Chemin de sortie de l'image annotée (png). Default: <image>_pred.png")
    args = ap.parse_args()

    leaves, nodes, corners = infer_image(args.image, args.score)

    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    nodes2, corners2, leaves2 = corr_fn(nodes, corners, leaves, img_bgr=img, printlog=False)

    print(f"Image: {args.image}")
    print(f"Score >= {args.score}")
    print(f"Leaves         : {len(leaves2)}")
    print(f"Internal nodes : {len(nodes2)}")
    print(f"Corners        : {len(corners2)}")

    if args.out.strip():
        out_path = args.out
    else:
        root, ext = os.path.splitext(args.image)
        out_path = root + "_pred.png"

    draw_predictions_on_image(args.image, leaves2, nodes2, corners2, out_path)
    print(f"Saved viz: {out_path}")

    print("\nSample outputs:")
    print("Leaf    :", leaves2[:5])
    print("Node    :", nodes2[:5])
    print("Corner  :", corners2[:5])