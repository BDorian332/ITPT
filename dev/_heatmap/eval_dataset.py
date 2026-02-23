# eval_dataset.py
import os, json, math, argparse
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch)
    def forward(self, x): return self.block(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([skip, x], dim=1))

class TinyUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, base=48):
        super().__init__()
        self.in0 = ConvBlock(in_ch, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.u2 = Up(base * 8 + base * 4, base * 4)
        self.u1 = Up(base * 4 + base * 2, base * 2)
        self.u0 = Up(base * 2 + base, base)
        self.head = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        s0 = self.in0(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        x = self.u2(s3, s2)
        x = self.u1(x, s1)
        x = self.u0(x, s0)
        return self.head(x)


def extract_peaks(hm_np: np.ndarray, threshold=0.3, k=200, nms=3) -> List[Tuple[int,int,float]]:
    hm = torch.from_numpy(hm_np)[None, None].float()
    pooled = F.max_pool2d(hm, kernel_size=nms, stride=1, padding=nms//2)
    is_peak = (hm == pooled) & (hm >= threshold)
    ys, xs = torch.where(is_peak[0, 0])
    if len(xs) == 0:
        return []
    scores = hm[0, 0, ys, xs]
    if len(scores) > k:
        topk = torch.topk(scores, k=min(k, len(scores))).indices
        xs, ys, scores = xs[topk], ys[topk], scores[topk]
    pts = [(int(xs[i]), int(ys[i]), float(scores[i])) for i in range(len(xs))]
    # tri score desc (utile pour AP)
    pts.sort(key=lambda t: t[2], reverse=True)
    return pts


def match_points(
    pred_xy: List[Tuple[float,float,float]],
    gt_xy: List[Tuple[float,float]],
    dist_thresh: float
) -> Tuple[int,int,int, List[int]]:
    if len(pred_xy) == 0:
        return 0, 0, len(gt_xy), []

    gt_used = [False] * len(gt_xy)
    matched = [0] * len(pred_xy)
    tp = 0

    for i, (px, py, _s) in enumerate(pred_xy):
        best_j = -1
        best_d = 1e18
        for j, (gx, gy) in enumerate(gt_xy):
            if gt_used[j]:
                continue
            d = math.hypot(px - gx, py - gy)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d <= dist_thresh:
            gt_used[best_j] = True
            matched[i] = 1
            tp += 1

    fp = len(pred_xy) - tp
    fn = len(gt_xy) - tp
    return tp, fp, fn, matched

def average_precision(pred_xy, gt_xy, dist_thresh):
    if len(gt_xy) == 0:
        return 1.0 if len(pred_xy) == 0 else 0.0
    tp, fp = 0, 0
    precisions = []
    recalls = []
    gt_used = [False] * len(gt_xy)

    for (px, py, score) in pred_xy:
        best_j = -1
        best_d = 1e18
        for j, (gx, gy) in enumerate(gt_xy):
            if gt_used[j]:
                continue
            d = math.hypot(px - gx, py - gy)
            if d < best_d:
                best_d, best_j = d, j

        if best_j >= 0 and best_d <= dist_thresh:
            gt_used[best_j] = True
            tp += 1
        else:
            fp += 1

        prec = tp / max(1, tp + fp)
        rec = tp / len(gt_xy)
        precisions.append(prec)
        recalls.append(rec)

    precisions = np.array(precisions, dtype=np.float64)
    recalls = np.array(recalls, dtype=np.float64)

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precisions, recalls):
        if r > prev_r:
            ap += p * (r - prev_r)
            prev_r = r
    return float(ap)

def prf(tp, fp, fn):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2*p*r / max(1e-12, (p + r))
    return p, r, f1


def load_annotations(points_path: str) -> List[dict]:
    with open(points_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "images" in data:
        return data["images"]
    if isinstance(data, list):
        return data
    raise ValueError("annotations.json must be a list or a dict with 'images'")

def rec_to_points(rec: dict) -> Dict[str, List[List[float]]]:
    pts = rec.get("points", None)
    if (pts is None) or (isinstance(pts, dict) and len(pts) == 0):
        pts = {}
        for kp in rec.get("keypoints", []):
            name = kp.get("name", "")
            if name:
                pts[name] = kp.get("points", [])
    if not isinstance(pts, dict):
        pts = {}
    return pts


def main():
    apg = argparse.ArgumentParser()
    apg.add_argument("--dataset_dir", default="out_dataset", help="Folder with images/ and annotations.json")
    apg.add_argument("--model", default="model_checkpoints/heatmap_model.pt", help="Path to checkpoint .pt")
    apg.add_argument("--out_dir", default="out_eval", help="Output folder")
    apg.add_argument("--start", type=int, default=4751, help="Start index (1-based, inclusive)")
    apg.add_argument("--end", type=int, default=5000, help="End index (1-based, inclusive)")
    apg.add_argument("--img_size", type=int, default=1500)
    apg.add_argument("--hm_size", type=int, default=1000)
    apg.add_argument("--threshold", type=float, default=0.3, help="Score threshold for peaks")
    apg.add_argument("--nms_size", type=int, default=3)
    apg.add_argument("--max_points", type=int, default=200)
    apg.add_argument("--dist_thresh_px", type=float, default=2.0, help="Matching tolerance in IMAGE pixels")
    apg.add_argument("--device", default="cpu", help="cuda or cpu (or cuda:0, etc)")
    apg.add_argument("--save_images", action="store_true", help="Save annotated images")
    args = apg.parse_args()

    dataset_dir = args.dataset_dir
    img_dir = os.path.join(dataset_dir, "images")
    ann_path = os.path.join(dataset_dir, "annotations.json")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(args.out_dir, "images_annotated"), exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: cuda requested but not available, fallback to cpu")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.model, map_location=device)
    class_order = ckpt.get("class_order", ["junction", "corner"])
    model_base = ckpt.get("cfg", {}).get("model_base", 48)

    model = TinyUNet(in_ch=1, out_ch=len(class_order), base=model_base)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    items = load_annotations(ann_path)

    start0 = max(0, args.start - 1)
    end0 = min(len(items) - 1, args.end - 1)
    if end0 < start0:
        raise ValueError("Invalid range: end < start")

    print("="*70)
    print(f"Device: {device}")
    print(f"Model: {args.model} | base={model_base} | classes={class_order}")
    print(f"Dataset: {dataset_dir} | items={len(items)}")
    print(f"Range: {args.start}..{args.end} (clamped to {start0+1}..{end0+1})")
    print(f"Infer: img_size={args.img_size} hm_size={args.hm_size} thr={args.threshold} nms={args.nms_size} max_points={args.max_points}")
    print(f"Eval: dist_thresh_px={args.dist_thresh_px}")
    print("="*70)

    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])


    totals = {cls: {"tp":0,"fp":0,"fn":0,"ap_list":[]} for cls in class_order}
    perfect = 0
    predictions_out = []

    for idx in tqdm(range(start0, end0 + 1), desc="Eval"):
        rec = items[idx]
        fname = rec.get("file_name") or rec.get("image")
        if not fname:
            continue
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        pts = rec_to_points(rec)
        gt_corner = pts.get("corner", [])
        gt_leaf = pts.get("leaf", [])
        gt_node = pts.get("node", [])
        gt_junction = gt_leaf + gt_node

        gt_by = {
            "corner": [(float(x), float(y)) for (x,y) in gt_corner if isinstance([x,y], list) or True],
            "junction": [(float(x), float(y)) for (x,y) in gt_junction if isinstance([x,y], list) or True],
        }

        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            logits = F.interpolate(logits, size=(args.hm_size, args.hm_size), mode="bilinear", align_corners=False)
            heatmaps = torch.sigmoid(logits)[0].cpu().numpy()

        sx = orig_w / args.hm_size
        sy = orig_h / args.hm_size

        pred_by = {}
        pred_by_scored = {}

        for ci, cls in enumerate(class_order):
            hm = heatmaps[ci]
            peaks = extract_peaks(hm, threshold=args.threshold, k=args.max_points, nms=args.nms_size)
            pred_scored = [(xh*sx, yh*sy, score) for (xh,yh,score) in peaks]
            pred_xy = [(x,y) for (x,y,_s) in pred_scored]
            pred_by[cls] = pred_xy
            pred_by_scored[cls] = pred_scored

        image_ok = True
        image_stats = {"image": fname, "points": []}

        for cls in class_order:
            gt_xy = gt_by.get(cls, [])
            pred_scored = pred_by_scored.get(cls, [])
            tp, fp, fn, matched_flags = match_points(pred_scored, gt_xy, args.dist_thresh_px)
            totals[cls]["tp"] += tp
            totals[cls]["fp"] += fp
            totals[cls]["fn"] += fn
            totals[cls]["ap_list"].append(average_precision(pred_scored, gt_xy, args.dist_thresh_px))

            n_pred = len(pred_scored)
            n_gt = len(gt_xy)

            if n_pred != n_gt:
                msg = f"[COUNT MISMATCH] {fname} | {cls}: pred={n_pred} gt={n_gt}"
                print(msg)

            if fp != 0 or fn != 0:
                image_ok = False

            pts_out = [[round(x, 2), round(y, 2)] for (x,y,_s) in pred_scored]
            image_stats["points"].append({"name": cls, "points": pts_out})

        if image_ok:
            perfect += 1

        predictions_out.append(image_stats)

        if args.save_images:
            draw = ImageDraw.Draw(img)
            colors = {"corner": "blue", "junction": "red"}
            r = 3
            for cls in class_order:
                for (x,y,_s) in pred_by_scored.get(cls, []):
                    c = colors.get(cls, "yellow")
                    draw.ellipse((x-r, y-r, x+r, y+r), fill=c, outline="white")
            out_img_path = os.path.join(args.out_dir, "images_annotated", fname)
            img.save(out_img_path)

    out_json_path = os.path.join(args.out_dir, "predictions.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(predictions_out, f, indent=2)

    print("\n" + "="*70)
    print(f"Saved predictions: {out_json_path}")
    if args.save_images:
        print(f"Saved images: {os.path.join(args.out_dir, 'images_annotated')}")
    print("="*70)

    for cls in class_order:
        tp = totals[cls]["tp"]; fp = totals[cls]["fp"]; fn = totals[cls]["fn"]
        p, r, f1 = prf(tp, fp, fn)
        ap_mean = float(np.mean(totals[cls]["ap_list"])) if len(totals[cls]["ap_list"]) else 0.0
        print(f"{cls:8s} | TP={tp:6d} FP={fp:6d} FN={fn:6d} | P={p:.4f} R={r:.4f} F1={f1:.4f} | AP={ap_mean:.4f}")

    tp_all = sum(totals[c]["tp"] for c in class_order)
    fp_all = sum(totals[c]["fp"] for c in class_order)
    fn_all = sum(totals[c]["fn"] for c in class_order)
    p_all, r_all, f1_all = prf(tp_all, fp_all, fn_all)

    print("-"*70)
    print(f"MICRO    | TP={tp_all:6d} FP={fp_all:6d} FN={fn_all:6d} | P={p_all:.4f} R={r_all:.4f} F1={f1_all:.4f}")
    print(f"PERFECT images: {perfect}")
    print("="*70)

if __name__ == "__main__":
    main()