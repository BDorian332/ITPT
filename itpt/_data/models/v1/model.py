import os
import torch
import cv2
from itpt.core import Model
from .preprocessing.cropping import extract_tree_from_image, CroppingModel
from .preprocessing.denoising import denoise_image, load_and_preprocess_image, DenoisingModel, img_to_tensor, img_to_gray
from .postprocessing.newick import build_newick
from .postprocessing.ocr import detect_texts, get_texts_detector_model

class v1(Model):
    def __init__(self):
        super().__init__()
        self._metadata["name"] = "V1"
        self._metadata["description"] = "Version 1"
        self._metadata["version"] = 1
        self.cropping_model = CroppingModel()
        self.denoising_model = DenoisingModel()
        self.texts_detector_model = None

    def load(self, device="cpu"):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        self.cropping_model.load_state_dict(torch.load(os.path.join(current_dir, "weights/cropping_model.pth"), map_location=device))
        self.denoising_model.load_state_dict(torch.load(os.path.join(current_dir, "weights/denoising_model.pth"), map_location=device))

        self.cropping_model.eval()
        self.denoising_model.eval()
        self.texts_detector_model = get_texts_detector_model()

        print(f"Models loaded")
        self._loaded = True

    def convert(self, image_path):
        self.ensure_loaded()

        img_rgb, img_tensor, (H, W) = self.load_and_preprocess(image_path)
        cropped_tree = self.extract_tree(img_rgb)
        cleaned_tree = self.clean_tree(cropped_tree)
        #nodes, corners = self.detect_nodes(cleaned_tree)
        #nodes, corners = self.correct_nodes(nodes, corners)
        texts = self.detect_texts(img_rgb)
        newick = self.build_newick(nodes, corners, texts)

        print(f"Conversion finished")
        return newick.to_string()

    def load_and_preprocess(self, image_path):
        print("Loading and Preprocessing image...")
        img_rgb, img_tensor, (H, W) = load_and_preprocess_image(image_path, size=(1500, 1500))
        print(f"Image loaded: original size (H={H}, W={W}), tensor shape: {img_tensor.shape}")
        return img_rgb, img_tensor, (H, W)

    def extract_tree(self, img_rgb):
        print("Extracting tree...")
        trees = extract_tree_from_image([img_rgb], self.cropping_model, (500, 500))
        tree = trees[0]
        print("Tree obtained, shape:", tree.shape)
        return tree

    def clean_tree(self, cropped_tree):
        print("Cleaning tree...")
        cleaned_trees = denoise_image([cropped_tree], self.denoising_model, (512, 512))
        cleaned_tree = cleaned_trees[0]
        print("Cleaned tree obtained, shape:", cleaned_tree.shape)
        return cleaned_tree

    def detect_nodes(self, cleaned_tree):
        print("Detecting nodes...")
        #nodes, corners = ?
        #print(f"Detected: {len(nodes)} nodes, {len(corners)} corners")
        return nodes, corners

    def correct_nodes(self, nodes, corners):
        print("Correcting nodes...")
        #nodes, corners = ?
        #print(f"Now: {len(nodes)} nodes, {len(corners)} corners")
        return nodes, corners

    def detect_texts(self, img_rgb):
        print("Detecting texts...")
        texts_by_image = detect_texts(img_rgb.unsqueeze(0), self.texts_detector_model)
        texts = texts_by_image[0]
        print("Found texts: ", texts)
        return texts

    def build_newick(self, nodes, corners, texts):
        print("Building Newick...")
        newick = build_newick(nodes, corners, texts=texts)
        print("Newick built: ", newick.to_string())
        return newick
