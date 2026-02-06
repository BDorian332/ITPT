import numpy as np
import torch
import torch.nn as nn
from torchvision import models

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

    img : numpy array (H, W, C)
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
    Convert a PyTorch tensor [C,H,W] normalized back to a numpy image [H,W,C] uint8
    """
    img_np = tensor.permute(1, 2, 0).cpu().numpy() # [H,W,C]
    img_np = (img_np * 255).astype(np.uint8)
    return img_np

def extract_tree_crops_from_images(
    img_tensors,
    model,
    device="cpu",
    return_bboxes=False
):
    """
    Extract tree crops from N image tensors using a BBox model.

    img_tensors : torch tensor [N, 3, H, W] normalized
    model : BBox prediction model
    return : list of cropped images (numpy), optional BBoxes
    """
    N, C, H, W = img_tensors.shape

    img_tensors = img_tensors.to(device)

    model.eval()
    with torch.no_grad():
        pred_bboxes_norm = model(img_tensors).cpu() # [N, 4]

    crops = []
    global_bboxes = []

    for i in range(N):
        img_np = tensor_to_img(img_tensors[i])

        global_bbox = denormalize_bbox(expand_bbox(pred_bboxes_norm[i]), W, H)
        cropped_tree = crop_image_with_bbox(img_np, global_bbox)
        cropped_tree = cv2.resize(cropped_tree, (H, W), interpolation=cv2.INTER_LINEAR)

        crops.append(cropped_tree)
        global_bboxes.append(global_bbox)

    if return_bboxes:
        return crops, global_bboxes
    return crops
