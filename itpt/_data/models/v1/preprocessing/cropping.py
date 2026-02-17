import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import cv2

class CroppingModel(nn.Module):
    def __init__(self):
        super().__init__()

        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net

        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        feats = self.backbone(x)
        bbox = self.regressor(feats)
        return bbox

def crop_image_with_bbox(img, bbox):
    """
    Crop an image using a bounding box.

    img : numpy array [H, W, C] uint8
    bbox : [x_min, y_min, x_max, y_max] in pixels
    """
    H, W = img.shape[:2]

    x_min = max(0, int(bbox[0]))
    y_min = max(0, int(bbox[1]))
    x_max = min(W, int(bbox[2]))
    y_max = min(H, int(bbox[3]))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid BBox after clipping: {x_min},{y_min},{x_max},{y_max}")

    return img[y_min:y_max, x_min:x_max].copy()

def denormalize_bbox(bbox_norm, img_w, img_h):
    """
    Convert normalized BBox to pixel coordinates.

    bbox_norm : tensor [4]
    """
    x_min = bbox_norm[0].item() * img_w
    y_min = bbox_norm[1].item() * img_h
    x_max = bbox_norm[2].item() * img_w
    y_max = bbox_norm[3].item() * img_h
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

def expand_bbox(bbox, expand_ratio=0.05):
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min

    x_min_exp = x_min - w * expand_ratio / 2
    x_max_exp = x_max + w * expand_ratio / 2
    y_min_exp = y_min - h * expand_ratio / 2
    y_max_exp = y_max + h * expand_ratio / 2

    return [x_min_exp, y_min_exp, x_max_exp, y_max_exp]

def tensor_to_img(tensor):
    """
    Convert a PyTorch tensor [C, H, W] normalized back to a numpy image [H, W, C] uint8
    """
    img_np = tensor.permute(1, 2, 0).cpu().numpy() # [H, W, C]
    img_np = (img_np * 255).astype(np.uint8)
    return img_np

def extract_tree_from_image(
    imgs_rgb,
    model,
    model_input_size,
    device="cpu",
    return_bboxes=False
):
    """
    Extract tree crops from N image tensors using a BBox model.

    imgs_rgb : list of numpy arrays [H, W, 3] uint8
    model : BBox prediction model
    model_input_size : prefered model input size
    return : list of numpy arrays [H, W, 3] uint8, optional BBoxes
    """
    from .denoising import img_to_gray, img_to_tensor
    img_tensors_list = []
    for img_rgb in imgs_rgb:
        img_resized = cv2.resize(img_rgb, model_input_size, interpolation=cv2.INTER_LINEAR)
        img_bw3 = img_to_gray(img_resized, threshold=200, out_channels=3) # [H, W, 3]
        tensor = img_to_tensor(img_bw3).unsqueeze(0) # [1, 3, H, W]
        img_tensors_list.append(tensor)

    img_tensors = torch.cat(img_tensors_list, dim=0).to(device) # [N, 3, H, W]

    # forward pass through the model
    model.eval()
    with torch.no_grad():
        pred_bboxes_norm = model(img_tensors).cpu() # [N, 4]

    trees = []
    global_bboxes = []

    for i in range(len(pred_bboxes_norm)):
        img_rgb = imgs_rgb[i] # [H, W, 3]
        H, W, _ = img_rgb.shape

        global_bbox = denormalize_bbox(expand_bbox(pred_bboxes_norm[i]), W, H)

        tree = crop_image_with_bbox(img_rgb, global_bbox)
        tree = cv2.resize(tree, (H, W), interpolation=cv2.INTER_LINEAR)

        trees.append(tree)
        global_bboxes.append(global_bbox)

    if return_bboxes:
        return trees, global_bboxes
    return trees
