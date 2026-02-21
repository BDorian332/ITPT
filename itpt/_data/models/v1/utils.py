import torch
import numpy as np
import cv2
from PIL import Image

def tensor_to_img(tensor):
    """
    Convert a PyTorch tensor [C, H, W] normalized back to a numpy image [H, W, C] uint8
    """
    img_np = tensor.permute(1, 2, 0).cpu().numpy() # [H, W, C]
    img_np = (img_np * 255).astype(np.uint8)
    return img_np

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

def load_and_preprocess_image(path_or_array, size=(512, 512)):
    """
    Load an image from disk or use a numpy array and preprocess it.

    path_or_array : str (path) or numpy array [H, W, 3] uint8
    size : tuple (width, height)
    return :
        img_rgb_resized : numpy array [height, width, 3] uint8
        img_tensor : tensor [1, H, W] normalized
        (H, W) : original size
    """
    if isinstance(path_or_array, str):
        try:
            pil_img = Image.open(path_or_array).convert("RGB")
            img_rgb = np.array(pil_img)
        except FileNotFoundError:
            raise FileNotFoundError(path_or_array)
    elif isinstance(path_or_array, np.ndarray):
        img_rgb = path_or_array.copy()

    H, W, _ = img_rgb.shape

    img_rgb_resized = cv2.resize(img_rgb, size, interpolation=cv2.INTER_LINEAR)
    img_bw = img_to_gray(img_rgb_resized, threshold=200, out_channels=1)
    img_tensor = img_to_tensor(img_bw) # [1, H, W] normalized

    return img_rgb_resized, img_tensor, (H, W)
