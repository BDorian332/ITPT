import os
import json
import random
import pickle
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

import matplotlib.pyplot as plt


def draw_gaussian(heatmap: torch.Tensor, x: float, y: float, sigma: float) -> None:
    """
    heatmap: (H, W) float tensor on CPU
    x, y: pixel coords in heatmap space
    sigma: gaussian std in pixels
    We write into heatmap using max(heatmap, gaussian)
    """
    H, W = heatmap.shape
    if sigma <= 0:
        return

    # 3*sigma window (standard)
    radius = int(3 * sigma)
    cx = int(round(x))
    cy = int(round(y))
    if cx < -radius or cy < -radius or cx >= W + radius or cy >= H + radius:
        return

    x0 = max(0, cx - radius)
    x1 = min(W - 1, cx + radius)
    y0 = max(0, cy - radius)
    y1 = min(H - 1, cy + radius)

    # Build gaussian patch
    xs = torch.arange(x0, x1 + 1, dtype=torch.float32)
    ys = torch.arange(y0, y1 + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    g = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma))

    # Max-blend into heatmap
    patch = heatmap[y0 : y1 + 1, x0 : x1 + 1]
    heatmap[y0 : y1 + 1, x0 : x1 + 1] = torch.maximum(patch, g)


def points_to_heatmaps(
    points_by_class: Dict[str, List[List[float]]],
    out_h: int,
    out_w: int,
    sigma: float,
    class_order: List[str],
) -> torch.Tensor:
    heatmaps = torch.zeros((len(class_order), out_h, out_w), dtype=torch.float32)
    for ci, cls in enumerate(class_order):
        pts = points_by_class.get(cls, [])
        hm = heatmaps[ci]
        for p in pts:
            if not isinstance(p, list) or len(p) != 2:
                continue
            x, y = float(p[0]), float(p[1])
            draw_gaussian(hm, x, y, sigma)
    return heatmaps


# Model: TinyUNet
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
        # pad if needed
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class TinyUNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 2, base: int = 64):
        """
        GRAYSCALE VERSION: in_ch=1 par défaut
        """
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


# Dataset avec augmentation zoom/padding
class HeatmapDataset(Dataset):
    IMG_EXTS = (".png", ".jpg", ".jpeg")

    def __init__(self, dataset_dir, img_size=1500, hm_size=1000, sigma=1.0, 
                 shrink_min=0.5, augment=True, id_list=None):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.hm_size = hm_size
        self.sigma = sigma
        self.shrink_min = shrink_min
        self.augment = augment
        self.class_order = ["node", "corner"]

        img_dir = os.path.join(dataset_dir, "images")
        points_file = os.path.join(dataset_dir, "annotations.json")

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Dossier images introuvable : {img_dir}")
        if not os.path.isfile(points_file):
            raise FileNotFoundError(f"Fichier annotations.json introuvable : {points_file}")

        with open(points_file, 'r') as f:
            self.points_by_image = json.load(f)

        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(self.IMG_EXTS)]

        def extract_id(name):
            m = re.findall(r"\d+", os.path.splitext(name)[0])
            if not m:
                raise ValueError(f"Pas d'ID numérique dans: {name}")
            return int(m[-1])

        imgs_by_id = {extract_id(f): f for f in img_files}

        points_ids = set(int(k) for k in self.points_by_image.keys())

        common_ids = sorted(set(imgs_by_id.keys()) & points_ids)

        if id_list is not None:
            id_set = set(int(x) for x in id_list)
            common_ids = [i for i in common_ids if i in id_set]

        if not common_ids:
            raise RuntimeError(f"[{dataset_dir}] Aucun ID commun entre images/ et annotations.json après filtres.")

        self.items = [
            {
                "img": os.path.join(img_dir, imgs_by_id[i]),
                "id": i
            }
            for i in common_ids
        ]

        print(f"[{os.path.basename(dataset_dir)}] {len(self.items)} images trouvées. augment={self.augment}")

    @staticmethod
    def _load_gray(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Impossible de lire : {path}")
        return img

    def _scale_points(self, pts: List[List[float]], in_w: int, in_h: int, out_w: int, out_h: int) -> List[List[float]]:
        if in_w <= 0 or in_h <= 0:
            return []
        sx = out_w / float(in_w)
        sy = out_h / float(in_h)
        scaled = []
        for p in pts:
            if not isinstance(p, list) or len(p) != 2:
                continue
            x, y = float(p[0]), float(p[1])
            scaled.append([x * sx, y * sy])
        return scaled

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_id = item["id"]
        img = self._load_gray(item["img"])

        h, w = img.shape

        points = self.points_by_image.get(str(img_id), [])

        pts_dict = {}
        for pt in points:
            if not isinstance(pt, dict):
                continue
            
            pt_type = pt.get("type", "node")
            x = pt.get("x")
            y = pt.get("y")
            
            if x is None or y is None:
                continue
                
            if pt_type not in pts_dict:
                pts_dict[pt_type] = []
            pts_dict[pt_type].append([float(x), float(y)])

        # Augmentation: zoom/padding
        if self.augment:
            scale = random.uniform(self.shrink_min, 1.0)
            new_w = int(w / scale)
            new_h = int(h / scale)

            pad_w = new_w - w
            pad_h = new_h - h

            off_x = random.randint(0, pad_w)
            off_y = random.randint(0, pad_h)

            canvas_img = np.full((new_h, new_w), 255, dtype=np.uint8)
            canvas_img[off_y:off_y + h, off_x:off_x + w] = img

            img = canvas_img
            h, w = new_h, new_w

            for key in pts_dict:
                adjusted = []
                for p in pts_dict[key]:
                    adjusted.append([p[0] + off_x, p[1] + off_y])
                pts_dict[key] = adjusted

        img_resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        pts_hm = {}
        for key in self.class_order:
            pts_in_current_space = pts_dict.get(key, [])
            pts_hm[key] = self._scale_points(pts_in_current_space, w, h, self.hm_size, self.hm_size)

        heatmaps = points_to_heatmaps(pts_hm, self.hm_size, self.hm_size, self.sigma, self.class_order)

        x = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0) / 255.0  # (1, H, W)
        y = heatmaps  # (2, hm, hm)

        return x, y


# Config
@dataclass
class HeatmapTrainConfig:
    dataset_dir: str = "./merged_dataset"
    img_size: int = 1500
    hm_size: int = 1000
    sigma: float = 1.0
    shrink_min: float = 0.5  # zoom
    batch_size: int = 1
    epochs: int = 60
    lr: float = 2e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    num_workers: int = 2
    seed: int = 42
    save_path: str = "model_checkpoints/heatmap_model.pth"
    model_base: int = 64
    scheduler_type: str = "cosine"  # "cosine", "plateau", or "none"


# ---------------------------
# Debug function
# ---------------------------
def save_debug_heatmap(ds, idx, save_path="debug_true_heatmaps.png"):
    """Sauvegarde une visualisation des heatmaps ground truth"""
    images_moi, heatmap_moi = ds[idx]

    img = images_moi[0].numpy()
    heatmaps = heatmap_moi.numpy()

    names = ["node", "corner"]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Image originale (grayscale)")
    plt.axis("off")

    for i in range(2):
        plt.subplot(1, 3, i + 2)
        plt.imshow(heatmaps[i], cmap="hot", vmin=0, vmax=1)
        plt.title(f"{names[i]} (max={heatmaps[i].max():.3f})")
        plt.colorbar(fraction=0.046)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Debug heatmap sauvegardé: {save_path}")


# Train
def train_single_dataset(cfg: HeatmapTrainConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)

    dataset_dir = cfg.dataset_dir

    # Charger le dataset complet pour split train/val
    base_ds = HeatmapDataset(
        dataset_dir,
        img_size=cfg.img_size,
        hm_size=cfg.hm_size,
        sigma=cfg.sigma,
        shrink_min=cfg.shrink_min,
        augment=False
    )

    n_total = len(base_ds)
    idxs = list(range(n_total))
    rng = random.Random(cfg.seed)
    rng.shuffle(idxs)

    val_len = max(1, int(n_total * cfg.val_ratio))
    train_len = n_total - val_len
    train_idxs = idxs[:train_len]
    val_idxs = idxs[train_len:]

    train_ids = [base_ds.items[i]["id"] for i in train_idxs]
    val_ids   = [base_ds.items[i]["id"] for i in val_idxs]

    # Créer les datasets train/val
    train_ds = HeatmapDataset(
        dataset_dir,
        img_size=cfg.img_size,
        hm_size=cfg.hm_size,
        sigma=cfg.sigma,
        shrink_min=cfg.shrink_min,
        augment=True,
        id_list=train_ids
    )

    val_ds = HeatmapDataset(
        dataset_dir,
        img_size=cfg.img_size,
        hm_size=cfg.hm_size,
        sigma=cfg.sigma,
        shrink_min=cfg.shrink_min,
        augment=False,
        id_list=val_ids
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = TinyUNet(in_ch=1, out_ch=2, base=cfg.model_base).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    loss_fn = nn.BCEWithLogitsLoss()
    print("Loss: BCEWithLogitsLoss")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = None
    if cfg.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        print("Scheduler: CosineAnnealingLR")
    elif cfg.scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        print("Scheduler: ReduceLROnPlateau")

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": [], "lr": []}
    history_path = os.path.join(os.path.dirname(cfg.save_path) or ".", "training_history.pkl")

    print(f"\nDevice={device} | Train={len(train_ds)} | Val={len(val_ds)}")
    print("="*60)
    print("Starting training...")
    print("="*60)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, n = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                if logits.shape[-2:] != y.shape[-2:]:
                    logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            running += loss.item() * bs
            n += bs
            pbar.set_postfix(loss=f"{(running/max(1,n)):.5f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

        train_loss = running / max(1, n)

        model.eval()
        total, m = 0.0, 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Val {epoch}/{cfg.epochs}", leave=False)
            for x, y in pbar_val:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(x)
                    if logits.shape[-2:] != y.shape[-2:]:
                        logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
                    vloss = loss_fn(logits, y)

                bs = x.size(0)
                total += vloss.item() * bs
                m += bs
                pbar_val.set_postfix(val=f"{(total/max(1,m)):.5f}")

        val_loss = total / max(1, m)

        if scheduler is not None:
            if cfg.scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = opt.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch:03d}/{cfg.epochs} | train={train_loss:.6f} | val={val_loss:.6f} | lr={current_lr:.2e}")

        base, ext = os.path.splitext(cfg.save_path)
        epoch_path = f"{base}_epoch{epoch:03d}{ext}"
        torch.save(model.state_dict(), epoch_path)
        print(f"  ✓ Saved: {epoch_path}")

        if val_loss < best_val:
            best_val = val_loss
            best_path = f"{base}_best{ext}"
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Best model: {best_path}")

        with open(history_path, "wb") as f:
            pickle.dump(history, f)

    print("="*60)
    print(f"Training complete! Best val loss: {best_val:.6f}")
    print(f"History saved to: {history_path}")

    return model, history


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="./merged_dataset", help="Dossier contenant images/ et annotations.json")
    p.add_argument("--img_size", type=int, default=1500)
    p.add_argument("--hm_size", type=int, default=1000)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--shrink_min", type=float, default=0.5, help="Facteur minimum pour zoom augmentation")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_path", default="model_checkpoints/heatmap_model.pth")
    p.add_argument("--model_base", type=int, default=64)
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "plateau", "none"])
    p.add_argument("--debug_sample", action="store_true", help="Save a debug heatmap and exit")

    args = p.parse_args()

    cfg = HeatmapTrainConfig(
        dataset_dir=args.dataset_dir,
        img_size=args.img_size,
        hm_size=args.hm_size,
        sigma=args.sigma,
        shrink_min=args.shrink_min,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        save_path=args.save_path,
        model_base=args.model_base,
        scheduler_type=args.scheduler,
    )

    print("="*60)
    print("HEATMAP TRAINING (MERGED DATASET)")
    print("="*60)
    print(f"Dataset dir: {cfg.dataset_dir}")
    print(f"Image size: {cfg.img_size}x{cfg.img_size}")
    print(f"Heatmap size: {cfg.hm_size}x{cfg.hm_size}")
    print(f"Sigma: {cfg.sigma}")
    print(f"Augmentation: zoom/padding (shrink_min={cfg.shrink_min})")
    print(f"Model: TinyUNet (base={cfg.model_base})")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Epochs: {cfg.epochs}")
    print(f"LR: {cfg.lr}")
    print("="*60)

    # Mode debug: charger un dataset et sauvegarder des samples
    if args.debug_sample:
        print("\nMODE DEBUG - Génération des samples de heatmaps...")
        
        try:
            debug_ds = HeatmapDataset(
                cfg.dataset_dir,
                img_size=cfg.img_size,
                hm_size=cfg.hm_size,
                sigma=cfg.sigma,
                shrink_min=cfg.shrink_min,
                augment=False
            )
            
            if len(debug_ds) > 0:
                save_debug_heatmap(debug_ds, 0, "debug_true_heatmaps_sample0.png")
                
            if len(debug_ds) > 1:
                save_debug_heatmap(debug_ds, 1, "debug_true_heatmaps_sample1.png")
            
            print("\nMode debug terminé. Vérifiez les fichiers debug_true_heatmaps_*.png")
            return
        except Exception as e:
            print(f"\nErreur en mode debug : {e}")
            import traceback
            traceback.print_exc()
            return

    train_single_dataset(cfg)


if __name__ == "__main__":
    main()
