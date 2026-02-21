import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import tensor_to_img

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

def denoise_image(imgs_rgb, model, model_input_size=(512, 512), device="cpu"):
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
        img_tensor = img_to_tensor(img_bw).unsqueeze(0) # [1, 1, H, W]
        img_tensors_list.append(img_tensor)

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
