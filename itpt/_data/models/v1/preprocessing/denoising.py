import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from .cropping import tensor_to_img

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DenoisingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = ConvBlock(64 + 32, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = nn.functional.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = nn.functional.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out(d1))

def img_to_gray(img_rgb, threshold=None, out_channels=1):
    """
    Convert an RGB image to a binary black/white grayscale image.

    img_rgb : numpy array [H, W, 3] uint8
    threshold : binarization threshold [0,255] or None
    out_channels : number of channels for the output image
    return : [H, W, out_channels] uint8
    """

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if threshold is not None:
        gray = (gray > threshold).astype("uint8") * 255
    gray = gray.reshape(gray.shape[0], gray.shape[1], 1)

    if out_channels > 1:
        gray = np.tile(gray, (1, 1, out_channels))

    return gray

def img_to_tensor(img):
    """
    img : numpy array [H, W, C] uint8
    return : tensor [C, H, W] normalized
    """
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

def load_and_preprocess_image(path, size=(512, 512)):
    """
    Load an image from disk and preprocess it.

    return:
    - img_rgb : numpy array [H, W, 3] uint8
    - img_tensor : tensor [1, 1, H, W] normalized
    - (H, W) : original size
    """
    try:
        pil_img = Image.open(path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(path)

    img_rgb = np.array(pil_img)
    H, W, _ = img_rgb.shape

    img_rgb = cv2.resize(img_rgb, size, interpolation=cv2.INTER_LINEAR)
    img_bw = img_to_gray(img_rgb, threshold=200, out_channels=1)
    img_tensor = img_to_tensor(img_bw) # [1, H, W]

    return img_rgb, img_tensor, (H, W)

def denoise_image(imgs_rgb, model, model_input_size, device="cpu"):
    """
    imgs_rgb : list of numpy arrays [H, W, 3] uint8
    model : denoising model
    model_input_size : prefered model input size
    return : list of numpy arrays [H, W, 3] uint8
    """
    img_tensors_list = []
    for img_rgb in imgs_rgb:
        img_resized = cv2.resize(img_rgb, model_input_size, interpolation=cv2.INTER_LINEAR)
        img_bw = img_to_gray(img_resized, threshold=200, out_channels=1) # [H, W, 1]
        tensor = img_to_tensor(img_bw).unsqueeze(0) # [1, 1, H, W]
        img_tensors_list.append(tensor)

    img_tensors = torch.cat(img_tensors_list, dim=0).to(device) # [B, 1, H, W]

    model.eval()
    with torch.no_grad():
        preds = model(img_tensors) # [B, 1, H, W]

    preds = preds.cpu()

    imgs_out = []
    for i in range(preds.shape[0]):
        out_tensor = preds[i] # [1, H, W]
        out_gray = tensor_to_img(out_tensor) # [H, W, 1]
        out_rgb = np.repeat(out_gray, 3, axis=2) # [H, W, 3]
        imgs_out.append(out_rgb)

    return imgs_out
