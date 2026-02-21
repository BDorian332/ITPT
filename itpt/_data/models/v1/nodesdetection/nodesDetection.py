import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from ..utils import img_to_gray, img_to_tensor

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

def extract_peaks(hm_np, threshold, nms):
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
    model_input_size,
    device,
    hm_size,
    threshold,
    nms_size
):
    print("Preparing image tensor")
    C, H, W = img_tensor.shape

    print(f"Running heatmap model (H={H}, W={W})")
    x = img_tensor.unsqueeze(0)
    x = F.interpolate(x, size=model_input_size, mode="bilinear", align_corners=False)
    x = x.to(device)

    model.eval()
    logits = model(x)
    if logits.shape[-2:] != (hm_size, hm_size):
        logits = F.interpolate(logits, size=(hm_size, hm_size), mode="bilinear", align_corners=False)

    heatmaps = torch.sigmoid(logits)[0].cpu().numpy()

    if heatmaps.shape[0] != 2:
        raise ValueError(f"Expected 2 channels (node, corner), got {heatmaps.shape[0]}")

    print("Extracting peaks")
    hm_node = heatmaps[0]
    hm_corner = heatmaps[1]

    node_hm_pts = extract_peaks(hm_node, threshold, nms_size)
    corner_hm_pts = extract_peaks(hm_corner, threshold, nms_size)

    sx = 1.0 / hm_size
    sy = 1.0 / hm_size

    nodes = [Point(x*sx, y*sy, "node") for (x, y, score) in node_hm_pts]
    corners = [Point(x*sx, y*sy, "corner") for (x, y, score) in corner_hm_pts]

    print(f"Found {len(nodes)} nodes and {len(corners)} corners")

    return nodes + corners

@torch.no_grad()
def detect_nodes(
    imgs_rgb,
    model,
    model_input_size=(1500, 1500),
    device="cpu",
    hm_size=1000,
    threshold=0.3,
    nms_size=3,
):
    print(f"Detecting nodes on {len(imgs_rgb)} image(s)...")
    nodes_by_image = []
    for i, tree_img in enumerate(imgs_rgb):
        img_bw1 = img_to_gray(tree_img, threshold=None, out_channels=1)
        img_tensor = img_to_tensor(img_bw1) # [1, H, W]

        nodes = infer_heatmap_points(
            img_tensor,
            model,
            model_input_size,
            device,
            hm_size,
            threshold,
            nms_size,
        )
        nodes_by_image.append(nodes)

    return nodes_by_image
