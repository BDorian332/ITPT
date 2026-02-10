import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import cv2

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
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch)
    def forward(self, x):
        return self.block(self.pool(x))

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
    def __init__(self, in_ch=1, out_ch=2, base=64):
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

def extract_peaks(hm_np, threshold=0.3, k=200, nms=3):
    hm = torch.from_numpy(hm_np)[None, None].float()
    pooled = F.max_pool2d(hm, kernel_size=nms, stride=1, padding=nms//2)
    is_peak = (hm == pooled) & (hm >= threshold)
    ys, xs = torch.where(is_peak[0, 0])

    if len(xs) == 0:
        return []

    scores = hm[0, 0, ys, xs]
    if len(scores) > k:
        topk_indices = torch.topk(scores, k=k).indices
        xs, ys, scores = xs[topk_indices], ys[topk_indices], scores[topk_indices]

    return [(int(xs[i]), int(ys[i]), float(scores[i])) for i in range(len(xs))]

@torch.no_grad()
def infer_points(
    model,
    device,
    img_gray,
    orig_size=None,
    img_size=1500,
    hm_size=1000,
    threshold=0.3,
    max_points=200,
    nms_size=3,
    return_heatmaps=False,
):
    """
    Détecte les points junction et corner dans une image grayscale

    Args:
        model: TinyUNet en eval mode
        device: torch.device
        img_gray: numpy array 2D (H, W) grayscale, uint8 ou float
        orig_size: (orig_w, orig_h) ou None (utilise img_gray.shape)
        img_size: taille de resize pour le modèle
        hm_size: résolution de la heatmap
        threshold: score minimum
        max_points: nombre max de points par classe
        nms_size: taille kernel NMS
        return_heatmaps: si True, retourne aussi les heatmaps

    Returns:
        junction_pts: liste de (x, y, score) en coordonnées image originale
        corner_pts: liste de (x, y, score) en coordonnées image originale
        [heatmaps]: numpy (2, hm_size, hm_size) si return_heatmaps=True
    """
    # Vérifications
    if isinstance(img_gray, torch.Tensor):
        img_gray = img_gray.detach().cpu().numpy()

    if img_gray.ndim != 2:
        raise ValueError(f"img_gray must be 2D, got shape {img_gray.shape}")

    # Dimensions
    in_h, in_w = img_gray.shape
    if orig_size is None:
        orig_w, orig_h = in_w, in_h
    else:
        orig_w, orig_h = int(orig_size[0]), int(orig_size[1])

    # Normalisation [0, 1]
    img = img_gray.astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0

    # Préparation tensor
    x = torch.from_numpy(img)[None, None]  # (1, 1, H, W)
    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    x = x.to(device)

    # Inférence
    model.eval()
    logits = model(x)
    logits = F.interpolate(logits, size=(hm_size, hm_size), mode="bilinear", align_corners=False)
    heatmaps = torch.sigmoid(logits)[0].detach().cpu().numpy()  # (2, hm, hm)

    # Extraction des pics
    hm_junction = heatmaps[0]
    hm_corner = heatmaps[1]

    junction_hm_pts = extract_peaks(hm_junction, threshold=threshold, k=max_points, nms=nms_size)
    corner_hm_pts = extract_peaks(hm_corner, threshold=threshold, k=max_points, nms=nms_size)

    # Scaling vers coordonnées image originale
    sx = orig_w / float(hm_size)
    sy = orig_h / float(hm_size)

    junction_pts = [(xh * sx, yh * sy, score) for (xh, yh, score) in junction_hm_pts]
    corner_pts = [(xh * sx, yh * sy, score) for (xh, yh, score) in corner_hm_pts]

    if return_heatmaps:
        return junction_pts, corner_pts, heatmaps

    return junction_pts, corner_pts
