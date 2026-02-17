import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import cv2
from ..preprocessing.denoising import img_to_gray, img_to_tensor


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class NodesDetectionModel(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 2, base: int = 16):
        super().__init__()
        self.in0 = ConvBlock(in_ch, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.u2 = Up(base * 8 + base * 4, base * 4)
        self.u1 = Up(base * 4 + base * 2, base * 2)
        self.u0 = Up(base * 2 + base, base)
        self.head = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        s0 = self.in0(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        x = self.u2(s3, s2)
        x = self.u1(x, s1)
        x = self.u0(x, s0)
        return self.head(x)


def auto_device(device_str=None):
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, device, model_base=64):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    state = torch.load(model_path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError("Expected a state_dict (dict).")
    model = NodesDetectionModel(in_ch=1, out_ch=2, base=model_base).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def extract_peaks(hm_np, threshold=0.3, nms=3):
    hm = torch.from_numpy(hm_np)[None, None].float()
    pooled = F.max_pool2d(hm, kernel_size=nms, stride=1, padding=nms // 2)
    is_peak = (hm == pooled) & (hm >= threshold)
    ys, xs = torch.where(is_peak[0, 0])
    if len(xs) == 0:
        return []
    scores = hm[0, 0, ys, xs]
    return [(int(xs[i]), int(ys[i]), float(scores[i])) for i in range(len(xs))]


@torch.no_grad()
def infer_heatmap_points(
    img_tensor,
    model,
    device="cpu",
    img_size=1500,
    hm_size=1000,
    threshold=0.3,
    nms_size=3,
):
    if img_tensor.ndim != 3:
        raise ValueError(f"Expected img_tensor shape [C, H, W], got {img_tensor.shape}")
    
    print("Preparing image tensor")
    C, H, W = img_tensor.shape
    
    if C == 1:
        img_gray = img_tensor[0]
    elif C == 3:
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        gray_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_gray = torch.from_numpy(gray_np).float() / 255.0
    else:
        raise ValueError(f"Expected 1 or 3 channels, got {C}")
    
    print(f"Running heatmap model (H={H}, W={W})")
    x = img_gray[None, None]
    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    x = x.to(device)
    
    model.eval()
    logits = model(x)
    if logits.shape[-2:] != (hm_size, hm_size):
        logits = F.interpolate(logits, size=(hm_size, hm_size), mode="bilinear", align_corners=False)
    
    heatmaps = torch.sigmoid(logits)[0].detach().cpu().numpy()
    
    if heatmaps.shape[0] != 2:
        raise ValueError(f"Expected 2 channels (node, corner), got {heatmaps.shape[0]}")
    
    print("Extracting peaks")
    hm_node = heatmaps[0]
    hm_corner = heatmaps[1]
    
    node_hm_pts = extract_peaks(hm_node, threshold=threshold, nms=nms_size)
    corner_hm_pts = extract_peaks(hm_corner, threshold=threshold, nms=nms_size)
    
    sx = 1500 / float(hm_size) # si vouloir coord normalisé remplacer 1500 par 1
    sy = 1500 / float(hm_size)
    
    node_pts = [(xh * sx, yh * sy, score) for (xh, yh, score) in node_hm_pts]
    corner_pts = [(xh * sx, yh * sy, score) for (xh, yh, score) in corner_hm_pts]
    
    print(f"Found {len(node_pts)} nodes and {len(corner_pts)} corners")
    
    return node_pts, corner_pts


@torch.no_grad()
def detect_nodes_batch(
    imgs_rgb,
    model,
    device="cpu",
    img_size=1500,
    hm_size=1000,
    threshold=0.3,
    nms_size=3,
):
    """
    Traite une liste d'images RGB et retourne les points détectés pour chacune.

    imgs_rgb  : list of numpy arrays [H, W, 3] uint8
    model     : NodesDetectionModel
    return    : nodes_by_image (list of lists), corners_by_image (list of lists)
    """
    print(f"Detecting nodes on {len(imgs_rgb)} image(s)...")
    nodes_by_image = []
    corners_by_image = []

    for i, tree_img in enumerate(imgs_rgb):
        tree_img = cv2.resize(tree_img, (1500, 1500), interpolation=cv2.INTER_LINEAR)

        if tree_img.ndim == 3 and tree_img.shape[2] == 3:
            img_gray = img_to_gray(tree_img, threshold=None, out_channels=1)
        else:
            img_gray = tree_img

        img_tensor = img_to_tensor(img_gray)  # [1, H, W]

        node_pts, corner_pts = infer_heatmap_points(
            img_tensor=img_tensor,
            model=model,
            device=device,
            img_size=img_size,
            hm_size=hm_size,
            threshold=threshold,
            nms_size=nms_size,
        )
        nodes_by_image.append(node_pts)
        corners_by_image.append(corner_pts)

    return nodes_by_image, corners_by_image


def draw_points_on_image(img_pil, node_pts, corner_pts, point_size=3):
    draw = ImageDraw.Draw(img_pil)
    for x, y, _ in node_pts:
        draw.ellipse((x - point_size, y - point_size, x + point_size, y + point_size), fill="red", outline="white")
    for x, y, _ in corner_pts:
        draw.ellipse((x - point_size, y - point_size, x + point_size, y + point_size), fill="blue", outline="white")
    return img_pil


def save_points_json(out_json, image_path, node_pts, corner_pts):
    payload = {
        "image": image_path,
        "node": [[float(x), float(y), float(s)] for (x, y, s) in node_pts],
        "corner": [[float(x), float(y), float(s)] for (x, y, s) in corner_pts],
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Infer node/corner points from NodesDetectionModel checkpoint.")
    ap.add_argument("--model", required=True, help="Checkpoint .pth")
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--out", default="output_points.png", help="Output overlay image")
    ap.add_argument("--out_json", default=None, help="Save points as JSON")
    ap.add_argument("--device", default="cuda", choices=[None, "cpu", "cuda"])
    ap.add_argument("--img_size", type=int, default=1500)
    ap.add_argument("--hm_size", type=int, default=1000)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--nms_size", type=int, default=3)
    ap.add_argument("--point_size", type=int, default=3)
    ap.add_argument("--model_base", type=int, default=8)
    args = ap.parse_args()

    device = auto_device(args.device)
    model = load_model(args.model, device=device, model_base=args.model_base)

    if not os.path.exists(args.image):
        raise SystemExit(f"ERROR: Image not found: {args.image}")

    img_pil = Image.open(args.image).convert("RGB")
    img_rgb = np.array(img_pil)
    img_gray_3d = img_to_gray(img_rgb, threshold=None, out_channels=1)
    img_tensor = img_to_tensor(img_gray_3d)

    node_pts, corner_pts = infer_heatmap_points(
        img_tensor=img_tensor,
        model=model,
        device=device,
        img_size=args.img_size,
        hm_size=args.hm_size,
        threshold=args.threshold,
        nms_size=args.nms_size,
    )

    if args.out_json is not None:
        save_points_json(args.out_json, args.image, node_pts, corner_pts)
        print(f"Saved points JSON to {args.out_json}")

    draw_points_on_image(img_pil, node_pts, corner_pts, point_size=args.point_size)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    img_pil.save(args.out)
    print(f"Saved overlay image to {args.out}")


if __name__ == "__main__":
    main()
