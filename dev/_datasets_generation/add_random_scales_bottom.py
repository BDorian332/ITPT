# python3 add_random_scales_bottom.py --input_dataset ./out_dataset_noisy/ --extra_h_min 20 --extra_h_max 20 --seed 42 --overwrite --scale_y_mode at_image_bottom
import argparse
import random
from pathlib import Path
import os

import cv2
import numpy as np


def list_images(images_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def draw_scale_style_simple(canvas, y_mid, x0, x1, thickness, tick_h, color, tick_color):
    cv2.line(canvas, (x0, y_mid), (x1, y_mid), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(canvas, (x0, y_mid - tick_h), (x0, y_mid + tick_h), tick_color, thickness, lineType=cv2.LINE_AA)
    cv2.line(canvas, (x1, y_mid - tick_h), (x1, y_mid + tick_h), tick_color, thickness, lineType=cv2.LINE_AA)


def draw_scale_style_segmented(canvas, y_mid, x0, x1, segments, thickness, tick_h, color):
    length = x1 - x0
    seg_w = max(1, length // segments)
    for i in range(segments):
        sx0 = x0 + i * seg_w
        sx1 = x0 + (i + 1) * seg_w if i < segments - 1 else x1
        if i % 2 == 0:
            cv2.rectangle(canvas, (sx0, y_mid - thickness), (sx1, y_mid + thickness), color, -1, lineType=cv2.LINE_AA)
    cv2.line(canvas, (x0, y_mid - tick_h), (x0, y_mid + tick_h), color, max(1, thickness), lineType=cv2.LINE_AA)
    cv2.line(canvas, (x1, y_mid - tick_h), (x1, y_mid + tick_h), color, max(1, thickness), lineType=cv2.LINE_AA)


def draw_scale_style_bracket(canvas, y_mid, x0, x1, thickness, tick_h, color):
    cv2.line(canvas, (x0, y_mid), (x1, y_mid), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(canvas, (x0, y_mid), (x0 + tick_h, y_mid - tick_h), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(canvas, (x0, y_mid), (x0 + tick_h, y_mid + tick_h), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(canvas, (x1, y_mid), (x1 - tick_h, y_mid - tick_h), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(canvas, (x1, y_mid), (x1 - tick_h, y_mid + tick_h), color, thickness, lineType=cv2.LINE_AA)


def draw_scale_style_multitick(canvas, y_mid, x0, x1, n_ticks, thickness, tick_h, color):
    cv2.line(canvas, (x0, y_mid), (x1, y_mid), color, thickness, lineType=cv2.LINE_AA)
    for i in range(n_ticks):
        t = i / (n_ticks - 1) if n_ticks > 1 else 0.0
        xt = int(round(x0 + t * (x1 - x0)))
        h = tick_h if i in (0, n_ticks - 1) else int(round(tick_h * random.uniform(0.4, 0.9)))
        cv2.line(canvas, (xt, y_mid - h), (xt, y_mid + h), color, max(1, thickness), lineType=cv2.LINE_AA)


def add_random_scale_bar(img_bgr: np.ndarray, rng: random.Random, extra_h: int, scale_y_mode: str):
    h, w = img_bgr.shape[:2]
    extra_h = max(30, extra_h)

    canvas = np.full((h + extra_h, w, 3), 255, dtype=np.uint8)
    canvas[:h, :w] = img_bgr

    pad_x = int(round(w * rng.uniform(0.04, 0.10)))
    bar_len = int(round(w * rng.uniform(0.18, 0.45)))
    thickness = rng.choice([1, 2, 2, 3, 3, 4])
    tick_h = rng.choice([6, 8, 10, 12, 14])
    if scale_y_mode == "at_image_bottom":
        # pile sur le bas de l'image originale
        y_mid = h
    else:
        # dans la zone ajoutée (comportement actuel)
        y_mid = h + int(round(extra_h * rng.uniform(0.35, 0.70)))

    placement = rng.choice(["left", "center", "right"])
    if placement == "left":
        x0 = pad_x + int(round(rng.uniform(-0.03, 0.03) * w))
    elif placement == "center":
        x0 = (w - bar_len) // 2 + int(round(rng.uniform(-0.05, 0.05) * w))
    else:
        x0 = w - pad_x - bar_len + int(round(rng.uniform(-0.03, 0.03) * w))

    x0 = clamp(x0, pad_x, w - pad_x - bar_len)
    x1 = x0 + bar_len

    g = int(round(rng.uniform(0, 60)))
    color = (g, g, g)
    tick_color = (int(round(rng.uniform(0, 40))),) * 3

    style = rng.choice(["simple", "segmented", "bracket", "multitick"])
    if style == "simple":
        draw_scale_style_simple(canvas, y_mid, x0, x1, thickness, tick_h, color, tick_color)
    elif style == "segmented":
        segments = rng.choice([4, 5, 6, 8])
        draw_scale_style_segmented(canvas, y_mid, x0, x1, segments, max(1, thickness), tick_h, color)
    elif style == "bracket":
        draw_scale_style_bracket(canvas, y_mid, x0, x1, thickness, tick_h, color)
    else:
        n_ticks = rng.choice([3, 4, 5, 6, 7])
        draw_scale_style_multitick(canvas, y_mid, x0, x1, n_ticks, thickness, tick_h, color)

    if rng.random() < 0.85:
        units = rng.choice(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
        suffix = rng.choice(["", " substitutions/site", " bp", " cm", " mm", ""])
        label = f"{units}{suffix}"

        font = rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX])
        font_scale = rng.uniform(0.45, 0.85)
        text_th = rng.choice([1, 1, 2])
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_th)

        where = rng.choice(["under", "under", "above"])
        tx = int(round((x0 + x1) / 2 - tw / 2))
        tx = clamp(tx, 5, w - tw - 5)
        if where == "under":
            ty = clamp(y_mid + tick_h + th + 6, h + th + 2, h + extra_h - 6)
        else:
            ty = clamp(y_mid - tick_h - 6, h + th + 2, h + extra_h - 6)

        if rng.random() < 0.35:
            pad = 3
            cv2.rectangle(canvas, (tx - pad, ty - th - pad), (tx + tw + pad, ty + pad), (255, 255, 255), -1)
            cv2.rectangle(canvas, (tx - pad, ty - th - pad), (tx + tw + pad, ty + pad), (0, 0, 0), 1)

        cv2.putText(canvas, label, (tx, ty), font, font_scale, color, text_th, lineType=cv2.LINE_AA)

    if rng.random() < 0.30:
        sep_y = h + int(round(extra_h * rng.uniform(0.10, 0.25)))
        sep_col = int(round(rng.uniform(180, 230)))
        cv2.line(canvas, (0, sep_y), (w - 1, sep_y), (sep_col, sep_col, sep_col), 1, lineType=cv2.LINE_AA)

    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dataset", required=True, help="Dataset input (contient images/)")
    ap.add_argument("--extra_h_min", type=int, default=70, help="Hauteur min ajoutée en bas (px)")
    ap.add_argument("--extra_h_max", type=int, default=180, help="Hauteur max ajoutée en bas (px)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output images if exist")
    ap.add_argument(
        "--scale_y_mode",
        choices=["bottom", "at_image_bottom"],
        default="bottom",
        help="Position verticale de l'echelle"
    )
    args = ap.parse_args()

    in_dir = Path(args.input_dataset)
    out_dir = Path(os.path.join(in_dir,"images_with_scaled"))
    in_images = in_dir / "images"
    out_images = out_dir

    if not in_images.exists():
        raise SystemExit(f"Missing folder: {in_images}")

    out_images.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    imgs = list_images(in_images)
    if not imgs:
        raise SystemExit(f"No images found in {in_images}")

    for p in imgs:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] cannot read: {p.name}")
            continue

        extra_h = rng.randint(args.extra_h_min, args.extra_h_max)
        out_img = add_random_scale_bar(img, rng, extra_h, args.scale_y_mode)


        out_path = out_images / p.name
        if out_path.exists() and not args.overwrite:
            continue

        ok = cv2.imwrite(str(out_path), out_img)
        if not ok:
            print(f"[WARN] cannot write: {out_path}")

    print(f"OK -> wrote images to: {out_images}")


if __name__ == "__main__":
    main()
