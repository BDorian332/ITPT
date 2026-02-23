import os
import re
import sys
import json
import math
import time
import shutil
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Any

import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CORRECTION_DIR = os.path.join(SCRIPT_DIR, "correction")
CORRECTION_HEATMAP_DIR = os.path.join(CORRECTION_DIR, "heatmap")


def _add_sys_path(p: str) -> None:
    if p and os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


def _find_and_add_dir_containing(filename: str, start_dir: str, max_depth: int = 6) -> Optional[str]:
    start_dir = os.path.abspath(start_dir)
    for root, dirs, files in os.walk(start_dir):
        rel = os.path.relpath(root, start_dir)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirs[:] = []
            continue
        if filename in files:
            _add_sys_path(root)
            return root
    return None


_add_sys_path(SCRIPT_DIR)
_add_sys_path(CORRECTION_DIR)
_add_sys_path(CORRECTION_HEATMAP_DIR)

_find_and_add_dir_containing("heatmap_infer_api.py", SCRIPT_DIR, max_depth=8)
_find_and_add_dir_containing("corrector.py", SCRIPT_DIR, max_depth=8)
_find_and_add_dir_containing("corrector_leaves.py", SCRIPT_DIR, max_depth=8)
_find_and_add_dir_containing("corrector_vote.py", SCRIPT_DIR, max_depth=8)


from contextlib import contextmanager

@contextmanager
def pushd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


_CORRECTOR_IMPORT_ERROR = None
correction = None
corrector_mod = None

try:
    if os.path.isdir(CORRECTION_DIR):
        import corrector as corrector_mod
        correction = corrector_mod.correction
except Exception as e:
    corrector_mod = None
    correction = None
    _CORRECTOR_IMPORT_ERROR = repr(e)


def progress_iter(items: Iterable, desc: str, unit: str = "it"):
    try:
        from tqdm import tqdm
        return tqdm(items, desc=desc, unit=unit, dynamic_ncols=True)
    except Exception:
        return items


def extract_tree_index(filename: str) -> Optional[int]:
    base = os.path.basename(filename)
    m = re.match(r"^tree_(\d+)\.(png|jpg|jpeg|bmp|webp)$", base, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def load_coco(ann_path: str):
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    img_by_id = {int(im["id"]): im for im in images}

    id_by_filename = {}
    for im in images:
        fn = im.get("file_name")
        if fn:
            id_by_filename[fn] = int(im["id"])

    ann_by_img: Dict[int, List[dict]] = {}
    for a in anns:
        ann_by_img.setdefault(int(a["image_id"]), []).append(a)

    cat_name = {int(c["id"]): c.get("name", str(c["id"])) for c in cats}

    return img_by_id, id_by_filename, ann_by_img, cat_name


def resolve_image_id(filename: str, id_by_filename: Dict[str, int]) -> Optional[int]:
    if filename in id_by_filename:
        return int(id_by_filename[filename])

    base = os.path.basename(filename)
    for k, v in id_by_filename.items():
        if os.path.basename(k) == base:
            return int(v)
    return None


def bbox_center_xywh(bbox_xywh):
    x, y, w, h = bbox_xywh
    return float(x + w / 2.0), float(y + h / 2.0)


def mask_centroid(mask01: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def instances_to_points(instances, use_masks: bool = True) -> List[dict]:
    if instances is None or len(instances) == 0:
        return []

    instances = instances.to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()

    has_masks = use_masks and instances.has("pred_masks")
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

        pts.append({"cls": int(classes[i]), "score": float(scores[i]), "x": float(cx), "y": float(cy)})
    return pts


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    distances: List[float]
    pred_is_tp: List[bool]
    gt_is_matched: List[bool]


def greedy_match_points(
    gt_pts: List[Tuple[float, float]],
    pred_pts: List[Tuple[float, float, float]],
    dist_thresh: float,
) -> MatchResult:
    if len(gt_pts) == 0:
        return MatchResult(
            tp=0,
            fp=len(pred_pts),
            fn=0,
            distances=[],
            pred_is_tp=[False] * len(pred_pts),
            gt_is_matched=[],
        )

    gt_used = [False] * len(gt_pts)
    pred_is_tp = [False] * len(pred_pts)
    dists: List[float] = []

    tp = fp = 0

    for i, (px, py, _sc) in enumerate(pred_pts):
        best_j = -1
        best_d = 1e18
        for j, (gx, gy) in enumerate(gt_pts):
            if gt_used[j]:
                continue
            d = math.hypot(px - gx, py - gy)
            if d < best_d:
                best_d = d
                best_j = j

        if best_j >= 0 and best_d <= dist_thresh:
            gt_used[best_j] = True
            pred_is_tp[i] = True
            tp += 1
            dists.append(float(best_d))
        else:
            fp += 1

    fn = int(gt_used.count(False))
    return MatchResult(tp=tp, fp=fp, fn=fn, distances=dists, pred_is_tp=pred_is_tp, gt_is_matched=gt_used)


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_ap(pr_curve: List[Tuple[float, float]]) -> float:
    if not pr_curve:
        return 0.0

    pr_curve = sorted(pr_curve, key=lambda t: t[0])
    recalls = np.array([r for r, _ in pr_curve], dtype=np.float32)
    precs = np.array([p for _, p in pr_curve], dtype=np.float32)

    for i in range(len(precs) - 2, -1, -1):
        precs[i] = max(precs[i], precs[i + 1])

    area = 0.0
    prev_r = 0.0
    for r, p in zip(recalls, precs):
        dr = float(r - prev_r)
        if dr > 0:
            area += dr * float(p)
            prev_r = float(r)
    return float(max(0.0, min(1.0, area)))


def draw_cross(img: np.ndarray, x: float, y: float, color, size: int = 7, thickness: int = 2) -> None:
    x_i, y_i = int(round(x)), int(round(y))
    cv2.line(img, (x_i - size, y_i), (x_i + size, y_i), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x_i, y_i - size), (x_i, y_i + size), color, thickness, cv2.LINE_AA)


def draw_circle(img: np.ndarray, x: float, y: float, color, radius: int = 6, thickness: int = 2) -> None:
    x_i, y_i = int(round(x)), int(round(y))
    cv2.circle(img, (x_i, y_i), radius, color, thickness, cv2.LINE_AA)


def annotate_failure_image_only_errors(
    img_bgr: np.ndarray,
    gt_by_name: Dict[str, List[Tuple[float, float]]],
    pred_by_name: Dict[str, List[Tuple[float, float, float]]],
    wanted_names: List[str],
    dist_thresh: float,
    score_thresh: float,
) -> np.ndarray:
    out = img_bgr.copy()

    class_color = {
        "leaf": (0, 255, 0),
        "internal_node": (0, 0, 255),
        "corner": (255, 0, 0),
    }

    for nm in wanted_names:
        color = class_color.get(nm, (255, 255, 255))

        gt_list = list(gt_by_name.get(nm, []))
        preds_all = pred_by_name.get(nm, [])
        preds = [p for p in preds_all if p[2] >= float(score_thresh)]
        preds = sorted(preds, key=lambda t: t[2], reverse=True)

        res = greedy_match_points(gt_list, preds, float(dist_thresh))

        for (is_tp, (px, py, _sc)) in zip(res.pred_is_tp, preds):
            if not is_tp:
                draw_circle(out, px, py, color, radius=7, thickness=2)

        for matched, (gx, gy) in zip(res.gt_is_matched, gt_list):
            if not matched:
                draw_cross(out, gx, gy, color, size=8, thickness=2)

    cv2.putText(
        out,
        "ONLY ERRORS: circle=FP(extra pred) | cross=FN(missing GT)",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _json_gt_points(gt_by_name: Dict[str, List[Tuple[float, float]]]) -> Dict[str, List[Dict[str, float]]]:
    return {k: [{"x": float(x), "y": float(y)} for (x, y) in v] for k, v in gt_by_name.items()}


def _json_pred_points(pred_by_name: Dict[str, List[Tuple[float, float, float]]]) -> Dict[str, List[Dict[str, float]]]:
    return {k: [{"x": float(x), "y": float(y), "score": float(s)} for (x, y, s) in v] for k, v in pred_by_name.items()}


def save_out_eval_for_nonperfect(
    *,
    fn: str,
    images_dir: str,
    out_images_failed: str,
    out_gt_points: str,
    out_pred_points: str,
    out_vis: str,
    gt_by_name: Dict[str, List[Tuple[float, float]]],
    pred_by_name: Dict[str, List[Tuple[float, float, float]]],
    wanted_names: List[str],
    dist_thresh: float,
    score_thresh: float,
) -> Dict[str, Optional[str]]:
    base = os.path.splitext(os.path.basename(fn))[0]

    src_img = os.path.join(images_dir, fn)
    dst_img = os.path.join(out_images_failed, os.path.basename(fn))
    copied = None
    if os.path.isfile(src_img):
        shutil.copy2(src_img, dst_img)
        copied = dst_img

    gt_path = os.path.join(out_gt_points, f"{base}_gt.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump({"file": fn, "gt_by_class": _json_gt_points(gt_by_name)}, f, indent=2)

    pred_path = os.path.join(out_pred_points, f"{base}_pred.json")
    pred_all = _json_pred_points(pred_by_name)
    pred_filt = {k: [p for p in v if p.get("score", 0.0) >= float(score_thresh)] for k, v in pred_all.items()}
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "file": fn,
                "score_threshold": float(score_thresh),
                "pred_by_class": pred_all,
                "pred_by_class_filtered": pred_filt,
            },
            f,
            indent=2,
        )

    vis_path = None
    img = cv2.imread(src_img, cv2.IMREAD_COLOR)
    if img is not None:
        ann = annotate_failure_image_only_errors(
            img_bgr=img,
            gt_by_name=gt_by_name,
            pred_by_name=pred_by_name,
            wanted_names=wanted_names,
            dist_thresh=float(dist_thresh),
            score_thresh=float(score_thresh),
        )
        vis_path = os.path.join(out_vis, f"{base}_vis.png")
        cv2.imwrite(vis_path, ann)

    return {"file": fn, "copied_image": copied, "gt_json": gt_path, "pred_json": pred_path, "vis_png": vis_path}


def build_predictor(config_path: str, weights_path: str, device: str, max_dets: int):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = device

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    cfg.TEST.DETECTIONS_PER_IMAGE = int(max_dets)

    predictor = DefaultPredictor(cfg)
    DetectionCheckpointer(predictor.model).load(weights_path)
    predictor.model.eval()
    return predictor


def _median_leaf_step(leaves_xy: List[Tuple[float, float]]) -> Optional[float]:
    if len(leaves_xy) < 2:
        return None
    ys = sorted([float(y) for (_, y) in leaves_xy])
    diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    diffs = [d for d in diffs if d > 0.0]
    if not diffs:
        return None
    return float(np.median(np.array(diffs, dtype=np.float32)))


def _min_abs_dx_leaf_to_nonleaf(leaves_xy: List[Tuple[float, float]], nonleaf_xy: List[Tuple[float, float]]) -> Optional[float]:
    if not leaves_xy or not nonleaf_xy:
        return None
    best = None
    for lx, _ly in leaves_xy:
        for nx, _ny in nonleaf_xy:
            d = abs(float(lx) - float(nx))
            if best is None or d < best:
                best = d
    return best


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(v)))))


def make_dynamic_extract_patch_fn(
    *,
    half_w: int,
    half_h: int,
):
    def _extract_patch_bgr_dyn(img_bgr: np.ndarray, cx: int, cy: int, size: int = 96) -> np.ndarray:
        h, w = img_bgr.shape[:2]

        x1 = int(cx - half_w)
        x2 = int(cx + half_w)
        y1 = int(cy - half_h)
        y2 = int(cy + half_h)

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)

        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)

        patch = img_bgr[y1c:y2c, x1c:x2c].copy()

        if any(p > 0 for p in (pad_left, pad_top, pad_right, pad_bottom)):
            patch = cv2.copyMakeBorder(
                patch,
                pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        if patch.size == 0:
            patch = np.zeros((size, size, 3), dtype=np.uint8)
            return patch

        if patch.shape[0] != size or patch.shape[1] != size:
            patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)

        return patch

    return _extract_patch_bgr_dyn


def main():
    start_time = time.perf_counter()

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to Detectron2 .pth")
    ap.add_argument("--config", required=True, help="Path to Detectron2 config.yaml")
    ap.add_argument("--images-dir", required=True, help="Folder containing images")
    ap.add_argument("--annotations", required=True, help="COCO annotations.json")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Run device")

    ap.add_argument("--score", type=float, default=0.5, help="Score threshold for summary + perfect-images")

    ap.add_argument("--dist", type=float, default=8.0, help="Distance threshold (px) for matching (default: 8)")

    ap.add_argument("--max-dets", type=int, default=400, help="Max detections per image")
    ap.add_argument("--use-masks", action="store_true", help="Use mask centroid if masks exist (otherwise bbox center)")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only N images (0 = all after filtering)")
    ap.add_argument("--report", default="eval_report.json", help="Output report JSON path")
    ap.add_argument("--pr-steps", type=int, default=30, help="Number of score thresholds for PR/AP curve")

    ap.add_argument("--range-start", type=int, default=4001)
    ap.add_argument("--range-end", type=int, default=5000)

    ap.add_argument("--save-nonperfect", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--nonperfect-dir", default="eval_nonperfect_png")

    ap.add_argument("--out-eval", default="out_eval")
    ap.add_argument("--save-eval-artifacts", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--post", action="store_true", help="Apply ./correction/corrector.py correction()")

    args = ap.parse_args()

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(args.weights)
    if not os.path.isfile(args.config):
        raise FileNotFoundError(args.config)
    if not os.path.isdir(args.images_dir):
        raise FileNotFoundError(args.images_dir)
    if not os.path.isfile(args.annotations):
        raise FileNotFoundError(args.annotations)
    if args.range_start > args.range_end:
        raise ValueError("--range-start must be <= --range-end")

    out_eval_root = os.path.abspath(args.out_eval)
    images_failed_dir = os.path.join(out_eval_root, "images_failed")
    pred_points_dir = os.path.join(out_eval_root, "pred_points")
    gt_points_dir = os.path.join(out_eval_root, "gt_points")
    vis_dir = os.path.join(out_eval_root, "vis")

    if args.save_eval_artifacts:
        os.makedirs(images_failed_dir, exist_ok=True)
        os.makedirs(pred_points_dir, exist_ok=True)
        os.makedirs(gt_points_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

    if args.post:
        if correction is None or corrector_mod is None:
            msg = (
                "Cannot import/use ./correction/corrector.py correction().\n"
                f"Expected folder: {CORRECTION_DIR}\n"
                f"sys.path head: {sys.path[:6]}\n"
            )
            if _CORRECTOR_IMPORT_ERROR is not None:
                msg += f"Import error: {_CORRECTOR_IMPORT_ERROR}\n"
            msg += (
                "Fix:\n"
                "- Ensure ./correction/corrector.py exists\n"
                "- Ensure ./correction/heatmap/heatmap_model.pt exists\n"
                "- Ensure ./correction/heatmap/heatmap_infer_api.py exists\n"
            )
            raise RuntimeError(msg)

    _img_by_id, id_by_filename, ann_by_img, cat_name = load_coco(args.annotations)

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    disk_files = sorted([f for f in os.listdir(args.images_dir) if f.lower().endswith(exts)])

    r0, r1 = int(args.range_start), int(args.range_end)
    all_files: List[str] = []
    for f in disk_files:
        idx = extract_tree_index(f)
        if idx is None:
            continue
        if r0 <= idx <= r1:
            all_files.append(f)

    if not all_files:
        raise RuntimeError(f"No images matched range tree_{r0:06d}..tree_{r1:06d} in {args.images_dir}")

    if args.limit and args.limit > 0:
        all_files = all_files[: int(args.limit)]

    predictor = build_predictor(args.config, args.weights, args.device, args.max_dets)

    wanted_names = ["leaf", "internal_node", "corner"]

    name_to_catid: Dict[str, int] = {}
    for cid, nm in cat_name.items():
        name_to_catid[str(nm).strip().lower()] = int(cid)

    coco_cat_ids: List[int] = []
    missing = []
    for nm in wanted_names:
        cid = name_to_catid.get(nm.lower())
        if cid is None:
            missing.append(nm)
        else:
            coco_cat_ids.append(int(cid))

    if missing:
        fallback = {"leaf": 1, "internal_node": 2, "corner": 3}
        print(f"[WARN] Missing categories in COCO: {missing}. Falling back to ids {fallback}.")
        coco_cat_ids = [fallback[nm] for nm in wanted_names]

    cache: List[Tuple[str, Dict[str, List[Tuple[float, float]]], Dict[str, List[Tuple[float, float, float]]]]] = []
    cache_map: Dict[str, Tuple[Dict[str, List[Tuple[float, float]]], Dict[str, List[Tuple[float, float, float]]]]] = {}

    for fn in progress_iter(all_files, "Inference (cache preds)", unit="img"):
        img_path = os.path.join(args.images_dir, fn)

        image_id = resolve_image_id(fn, id_by_filename)
        if image_id is None:
            continue

        gt_anns = ann_by_img.get(int(image_id), [])

        gt_by_name: Dict[str, List[Tuple[float, float]]] = {nm: [] for nm in wanted_names}
        for a in gt_anns:
            cid = int(a.get("category_id", -1))
            bbox = a.get("bbox", None)
            if bbox is None:
                continue
            gx, gy = bbox_center_xywh(bbox)
            if cid in coco_cat_ids:
                nm = wanted_names[coco_cat_ids.index(cid)]
                gt_by_name[nm].append((gx, gy))

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        outputs = predictor(img)
        inst = outputs.get("instances", None)
        pred_pts = instances_to_points(inst, use_masks=args.use_masks)

        pred_leaves: List[Tuple[float, float, float]] = []
        pred_nodes: List[Tuple[float, float, float]] = []
        pred_corners: List[Tuple[float, float, float]] = []

        for p in pred_pts:
            cls = int(p["cls"])
            if cls == 0:
                pred_leaves.append((p["x"], p["y"], p["score"]))
            elif cls == 1:
                pred_nodes.append((p["x"], p["y"], p["score"]))
            elif cls == 2:
                pred_corners.append((p["x"], p["y"], p["score"]))

        if args.post:
            leaves_xy = [(float(x), float(y)) for (x, y, _s) in pred_leaves]
            nonleaf_xy = [(float(x), float(y)) for (x, y, _s) in (pred_nodes + pred_corners)]

            dx = _min_abs_dx_leaf_to_nonleaf(leaves_xy, nonleaf_xy)
            ystep = _median_leaf_step(leaves_xy)

            half_w = _clamp_int((dx * 1.5) if dx is not None else 48.0, lo=16, hi=220)
            half_h = _clamp_int(((ystep * 1.5) / 2.0) if ystep is not None else 48.0, lo=16, hi=220)

            orig_extract = getattr(corrector_mod, "extract_patch_bgr", None)
            dyn_extract = make_dynamic_extract_patch_fn(half_w=half_w, half_h=half_h)

            with pushd(CORRECTION_DIR):
                try:
                    if orig_extract is not None:
                        setattr(corrector_mod, "extract_patch_bgr", dyn_extract)

                    pred_nodes = [(int(p[0]), int(p[1]), float(p[2])) for p in pred_nodes if p[2] >= float(args.score)]
                    pred_corners = [(int(p[0]), int(p[1]), float(p[2])) for p in pred_corners if p[2] >= float(args.score)]
                    pred_leaves = [(int(p[0]), int(p[1]), float(p[2])) for p in pred_leaves if p[2] >= float(args.score)]


                    nodes2, corners2, leaves2 = correction(
                        nodes=pred_nodes,
                        corners=pred_corners,
                        leaves=pred_leaves,
                        img_bgr=img,
                    )
                finally:
                    if orig_extract is not None:
                        setattr(corrector_mod, "extract_patch_bgr", orig_extract)

            pred_nodes, pred_corners, pred_leaves = nodes2, corners2, leaves2

        pred_by_name: Dict[str, List[Tuple[float, float, float]]] = {
            "leaf": list(pred_leaves),
            "internal_node": list(pred_nodes),
            "corner": list(pred_corners),
        }

        cache.append((fn, gt_by_name, pred_by_name))
        cache_map[fn] = (gt_by_name, pred_by_name)

    if not cache:
        raise RuntimeError(
            "No images matched COCO annotations (cache is empty). "
            "Check file_name in annotations.json vs your folder filenames."
        )

    per_class = {nm: {"tp": 0, "fp": 0, "fn": 0, "dists": [], "gt_count": 0, "pred_count": 0} for nm in wanted_names}
    all_gt_counts: List[int] = []
    all_pred_counts: List[int] = []

    perfect_files: List[str] = []
    nonperfect_info: List[dict] = []

    for (fn, gt_by_name, pred_by_name) in cache:
        img_fp_total = 0
        img_fn_total = 0
        img_tp_total = 0

        per_img = {"file": fn, "per_class": {}, "tp": 0, "fp": 0, "fn": 0}

        gt_total_img = 0
        pred_total_img = 0

        for nm in wanted_names:
            gt_list = gt_by_name[nm]
            preds = [p for p in pred_by_name[nm] if p[2] >= float(args.score)]
            preds = sorted(preds, key=lambda t: t[2], reverse=True)

            gt_total_img += len(gt_list)
            pred_total_img += len(preds)

            res = greedy_match_points(gt_list, preds, float(args.dist))

            per_class[nm]["tp"] += res.tp
            per_class[nm]["fp"] += res.fp
            per_class[nm]["fn"] += res.fn
            per_class[nm]["dists"].extend(res.distances)
            per_class[nm]["gt_count"] += len(gt_list)
            per_class[nm]["pred_count"] += len(preds)

            img_tp_total += res.tp
            img_fp_total += res.fp
            img_fn_total += res.fn

            per_img["per_class"][nm] = {"tp": res.tp, "fp": res.fp, "fn": res.fn, "gt": len(gt_list), "pred": len(preds)}

        per_img["tp"] = img_tp_total
        per_img["fp"] = img_fp_total
        per_img["fn"] = img_fn_total

        all_gt_counts.append(gt_total_img)
        all_pred_counts.append(pred_total_img)

        if img_fp_total == 0 and img_fn_total == 0:
            perfect_files.append(fn)
        else:
            nonperfect_info.append(per_img)

    perfect_count = len(perfect_files)
    nonperfect_count = len(nonperfect_info)
    total_eval = len(cache)
    perfect_ratio = (perfect_count / total_eval) if total_eval else 0.0

    # Save NON-perfect
    out_eval_exports: List[dict] = []
    saved_files_legacy: List[str] = []

    if nonperfect_count > 0:
        if args.save_nonperfect:
            os.makedirs(args.nonperfect_dir, exist_ok=True)

        for item in progress_iter(nonperfect_info, "Saving NON-perfect artifacts", unit="img"):
            fn = item["file"]
            img_path = os.path.join(args.images_dir, fn)

            pair = cache_map.get(fn)
            if pair is None:
                continue
            gt_by_name, pred_by_name = pair

            if args.save_eval_artifacts:
                exp = save_out_eval_for_nonperfect(
                    fn=fn,
                    images_dir=args.images_dir,
                    out_images_failed=images_failed_dir,
                    out_gt_points=gt_points_dir,
                    out_pred_points=pred_points_dir,
                    out_vis=vis_dir,
                    gt_by_name=gt_by_name,
                    pred_by_name=pred_by_name,
                    wanted_names=wanted_names,
                    dist_thresh=float(args.dist),
                    score_thresh=float(args.score),
                )
                out_eval_exports.append(exp)

            if args.save_nonperfect:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                ann = annotate_failure_image_only_errors(
                    img_bgr=img,
                    gt_by_name=gt_by_name,
                    pred_by_name=pred_by_name,
                    wanted_names=wanted_names,
                    dist_thresh=float(args.dist),
                    score_thresh=float(args.score),
                )
                base = os.path.splitext(os.path.basename(fn))[0]
                out_path = os.path.join(args.nonperfect_dir, f"{base}_fail.png")
                ok = cv2.imwrite(out_path, ann)
                if ok:
                    saved_files_legacy.append(out_path)

        if args.save_eval_artifacts:
            out_eval_index = os.path.join(out_eval_root, "out_eval_index.json")
            with open(out_eval_index, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "score": float(args.score),
                        "dist": float(args.dist),
                        "images_evaluated": int(len(cache)),
                        "perfect_count": int(perfect_count),
                        "nonperfect_count": int(nonperfect_count),
                        "exports": out_eval_exports,
                        "folders": {
                            "root": out_eval_root,
                            "images_failed": images_failed_dir,
                            "gt_points": gt_points_dir,
                            "pred_points": pred_points_dir,
                            "vis": vis_dir,
                        },
                    },
                    f,
                    indent=2,
                )

    thr_list = np.linspace(0.0, 1.0, max(5, int(args.pr_steps))).tolist()
    ap_by_class: Dict[str, float] = {}
    pr_by_class: Dict[str, List[Tuple[float, float]]] = {}

    for nm in wanted_names:
        pr_curve: List[Tuple[float, float]] = []
        for thr in progress_iter(thr_list, f"PR/AP thresholds [{nm}]", unit="thr"):
            TP = FP = FN = 0
            thr = float(thr)

            for (_fn, gt_by_name, pred_by_name) in cache:
                gt_list = gt_by_name[nm]
                preds = [p for p in pred_by_name[nm] if p[2] >= thr]
                preds = sorted(preds, key=lambda t: t[2], reverse=True)

                res = greedy_match_points(gt_list, preds, float(args.dist))
                TP += res.tp
                FP += res.fp
                FN += res.fn

            prec, rec, _f1 = precision_recall_f1(TP, FP, FN)
            pr_curve.append((rec, prec))

        ap_by_class[nm] = compute_ap(pr_curve)
        pr_by_class[nm] = pr_curve

    summary: Dict[str, Any] = {"per_class": {}, "global": {}}

    for nm in wanted_names:
        tp = int(per_class[nm]["tp"])
        fp = int(per_class[nm]["fp"])
        fn = int(per_class[nm]["fn"])
        prec, rec, f1v = precision_recall_f1(tp, fp, fn)

        d = per_class[nm]["dists"]
        d_arr = np.array(d, dtype=np.float32) if len(d) else np.array([], dtype=np.float32)

        dist_stats = {
            "mean": float(d_arr.mean()) if d_arr.size else None,
            "median": float(np.median(d_arr)) if d_arr.size else None,
            "p90": float(np.percentile(d_arr, 90)) if d_arr.size else None,
            "min": float(d_arr.min()) if d_arr.size else None,
            "max": float(d_arr.max()) if d_arr.size else None,
            "count": int(d_arr.size),
        }

        summary["per_class"][nm] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1v),
            "dist_thresh_px": float(args.dist),
            "localization_error_px": dist_stats,
            "gt_points": int(per_class[nm]["gt_count"]),
            "pred_points": int(per_class[nm]["pred_count"]),
            "ap": float(ap_by_class.get(nm, 0.0)),
        }

    gt_counts = np.array(all_gt_counts, dtype=np.float32) if all_gt_counts else np.array([], dtype=np.float32)
    pr_counts = np.array(all_pred_counts, dtype=np.float32) if all_pred_counts else np.array([], dtype=np.float32)
    if gt_counts.size:
        diff = pr_counts - gt_counts
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff * diff)))
    else:
        mae = None
        rmse = None

    TP = sum(int(per_class[nm]["tp"]) for nm in wanted_names)
    FP = sum(int(per_class[nm]["fp"]) for nm in wanted_names)
    FN = sum(int(per_class[nm]["fn"]) for nm in wanted_names)
    gprec, grec, gf1 = precision_recall_f1(TP, FP, FN)

    summary["global"] = {
        "images_evaluated": int(len(cache)),
        "device": str(args.device),
        "range_start": int(args.range_start),
        "range_end": int(args.range_end),
        "score_threshold_summary": float(args.score),
        "dist_thresh_px": float(args.dist),
        "micro_tp": int(TP),
        "micro_fp": int(FP),
        "micro_fn": int(FN),
        "micro_precision": float(gprec),
        "micro_recall": float(grec),
        "micro_f1": float(gf1),
        "count_mae": mae,
        "count_rmse": rmse,
        "ap_mean": float(np.mean([ap_by_class[nm] for nm in wanted_names])) if wanted_names else 0.0,
        "perfect_images": int(perfect_count),
        "nonperfect_images": int(nonperfect_count),
        "perfect_ratio": float(perfect_ratio),
        "nonperfect_dir": os.path.abspath(args.nonperfect_dir) if args.save_nonperfect else None,
        "out_eval": out_eval_root if args.save_eval_artifacts else None,
        "post_enabled": bool(args.post),
        "correction_dir": os.path.abspath(CORRECTION_DIR) if os.path.isdir(CORRECTION_DIR) else None,
    }

    report = {
        "weights": os.path.abspath(args.weights),
        "config": os.path.abspath(args.config),
        "images_dir": os.path.abspath(args.images_dir),
        "annotations": os.path.abspath(args.annotations),
        "selection": {
            "range_start": int(args.range_start),
            "range_end": int(args.range_end),
            "requested_files": int(len(all_files)),
            "matched_in_coco": int(len(cache)),
        },
        "post": {
            "enabled": bool(args.post),
            "correction_dir": os.path.abspath(CORRECTION_DIR) if os.path.isdir(CORRECTION_DIR) else None,
            "note": "Heatmap patch extraction is dynamically cropped (x/y) then resized to 96x96.",
        },
        "perfect_images": {
            "definition": "Perfect = FP==0 and FN==0 on the image (across all classes) using --score and --dist.",
            "score": float(args.score),
            "dist": float(args.dist),
            "count": int(perfect_count),
            "ratio": float(perfect_ratio),
        },
        "nonperfect_images": {
            "count": int(nonperfect_count),
            "saved_png_dir": os.path.abspath(args.nonperfect_dir) if args.save_nonperfect else None,
            "files": [x["file"] for x in nonperfect_info],
        },
        "out_eval": {
            "enabled": bool(args.save_eval_artifacts),
            "root": out_eval_root if args.save_eval_artifacts else None,
            "images_failed": images_failed_dir if args.save_eval_artifacts else None,
            "gt_points": gt_points_dir if args.save_eval_artifacts else None,
            "pred_points": pred_points_dir if args.save_eval_artifacts else None,
            "vis": vis_dir if args.save_eval_artifacts else None,
            "vis_note": "Visualization shows ONLY errors: FP circles + FN crosses.",
        },
        "summary": summary,
        "pr_curves": pr_by_class,
        "nonperfect_detail": nonperfect_info,
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n================ EVAL (POINTS) ================")
    print(f"Images evaluated (matched COCO): {summary['global']['images_evaluated']}")
    print(f"Range: tree_{args.range_start:06d}..tree_{args.range_end:06d}")
    print(f"Device: {summary['global']['device']}")
    print(f"Score threshold (perfect + summary): {args.score}")
    print(f"Distance threshold: {args.dist}px")
    print(f"Post(correction): {bool(args.post)}")
    print("-----------------------------------------------")
    for nm in wanted_names:
        m = summary["per_class"][nm]
        loc_mean = m["localization_error_px"]["mean"]
        print(
            f"[{nm}]  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  AP={m['ap']:.3f}  "
            f"TP={m['tp']} FP={m['fp']} FN={m['fn']}  loc_mean={loc_mean}"
        )
    print("-----------------------------------------------")
    print(
        f"[GLOBAL] micro_P={summary['global']['micro_precision']:.3f}  "
        f"micro_R={summary['global']['micro_recall']:.3f}  micro_F1={summary['global']['micro_f1']:.3f}  "
        f"mAP={summary['global']['ap_mean']:.3f}"
    )
    print("-----------------------------------------------")
    print(f"[PERFECT] 100% images (0 errors): {perfect_count}/{total_eval}  ({perfect_ratio * 100:.2f}%)")
    if args.save_nonperfect:
        print(f"[NON-PERFECT] legacy PNGs in: {os.path.abspath(args.nonperfect_dir)}  (count={len(saved_files_legacy)})")
    if args.save_eval_artifacts:
        print(f"[OUT_EVAL] folder: {out_eval_root}  (exports={len(out_eval_exports)})")
    print("-----------------------------------------------")
    print(f"Report saved: {args.report}")
    print("===============================================\n")

    elapsed = time.perf_counter() - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()