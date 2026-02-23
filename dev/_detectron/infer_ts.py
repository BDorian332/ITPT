import os
import argparse
import cv2
import numpy as np
import torch

try:
    from correction.corrector import correction as corr_fn
except Exception:
    corr_fn = None


def mask_centroid(mask01: np.ndarray):
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def outputs_to_points(boxes_xyxy, scores, classes, masks_u8):
    n = int(scores.shape[0]) if scores is not None else 0
    if n == 0:
        return []

    pts = []
    has_masks = masks_u8 is not None and len(masks_u8) == n

    for i in range(n):
        x1, y1, x2, y2 = boxes_xyxy[i]
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)

        if has_masks:
            m = masks_u8[i]
            if m.dtype != np.uint8:
                m = m.astype(np.uint8)
            if m.max() > 1:
                m = (m > 0).astype(np.uint8)
            c = mask_centroid(m)
            if c is not None:
                cx, cy = c

        pts.append((int(classes[i]), cx, cy, float(scores[i])))

    return pts


def load_ts_model(ts_path: str, device: str):
    import torchvision
    import torchvision.extension
    torchvision.extension._assert_has_ops()
    dev = torch.device(device)
    model = torch.jit.load(ts_path, map_location=dev).eval()
    return model, dev


def infer_image(ts_model, device, image_path: str, score_thresh: float, force_size: int):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    if force_size > 0:
        img_rs = cv2.resize(img, (force_size, force_size), interpolation=cv2.INTER_LINEAR)
    else:
        img_rs = img

    x = torch.from_numpy(img_rs).to(device=device)

    with torch.no_grad():
        out = ts_model(x)

    boxes = out["boxes"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()
    classes = out["classes"].detach().cpu().numpy()
    masks = out["masks"].detach().cpu().numpy() if "masks" in out else None

    keep = scores >= float(score_thresh)
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]
    if masks is not None:
        masks = masks[keep]

    pts = outputs_to_points(boxes, scores, classes, masks)

    leaves, internal_nodes, corners = [], [], []
    for cls, xpt, ypt, sc in pts:
        if cls == 0:
            leaves.append((int(round(xpt)), int(round(ypt)), float(sc)))
        elif cls == 1:
            internal_nodes.append((int(round(xpt)), int(round(ypt)), float(sc)))
        elif cls == 2:
            corners.append((int(round(xpt)), int(round(ypt)), float(sc)))

    return img_rs, leaves, internal_nodes, corners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", required=True, help="TorchScript model (.pt)")
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--score", type=float, default=0.5, help="Score threshold")
    args = ap.parse_args()

    ts_model, dev = load_ts_model(args.ts, args.device)

    img_rs, leaves, nodes, corners = infer_image(
        ts_model=ts_model,
        device=dev,
        image_path=args.image,
        score_thresh=args.score,
        force_size=1500,
    )

    nodes2, corners2, leaves2 = corr_fn(nodes, corners, leaves, img_bgr=img_rs, printlog=False)
    leaves, nodes, corners = leaves2, nodes2, corners2

    print(f"Image: {args.image}")
    print(f"Score >= {args.score}")
    print(f"Leaves         : {len(leaves)}")
    print(f"Internal nodes : {len(nodes)}")
    print(f"Corners        : {len(corners)}")
    print("Sample leaf   :", leaves[:5])
    print("Sample node   :", nodes[:5])
    print("Sample corner :", corners[:5])

if __name__ == "__main__":
    main()