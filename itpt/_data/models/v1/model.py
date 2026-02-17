import torch
import cv2
from pathlib import Path
from itpt.core import Model
from itpt.core import Point, build_newick
from .preprocessing.cropping import extract_tree_from_image, CroppingModel
from .preprocessing.denoising import denoise_image, load_and_preprocess_image, DenoisingModel
from .postprocessing.ocr import detect_texts, get_textsDetector_model
from .nodesdetection.nodesDetection import NodesDetectionModel, detect_nodes_batch

import numpy as np

class v1(Model):
    def __init__(self):
        super().__init__()
        self._metadata.update({
            "name": "V1",
            "description": "Cropping + Cleaning + Heatmap",
            "version": 1,
            "weights_urls": {
                "Cropping Model": "https://example.com/cropping.pth",
                "Denoising Model": "https://example.com/denoising.pth",
                "Nodes Detection Model": "https://example.com/nodesDetection.pth"
            }
        })
        self.cropping_model = CroppingModel()
        self.denoising_model = DenoisingModel()
        self.nodesDetection_model = NodesDetectionModel(base=16) # j'ai entrainer 8 / 16 / 32 / 64 à tester lequels marchent le mieux
        self.textsDetector_model = None

    def load(self, cropping_model_weights_path_or_url=None, denoising_model_weights_path_or_url=None, nodesdetection_model_weights_path_or_url=None):
        current_dir = Path(__file__).resolve().parent
        weights_dir = current_dir / "weights"
        device = "cpu"

        if cropping_model_weights_path_or_url is None:
            cropping_model_weights_path = self.ensure_weights(weights_dir / "cropping_model.pth", self.get_metadata()["weights_urls"]["Cropping Model"])
        elif cropping_model_weights_path_or_url.startswith(("http://", "https://")):
            cropping_model_weights_path = self.download_weights(cropping_model_weights_path_or_url, self.get_model_cache_path() / "cropping_model.pth")
        else:
            cropping_model_weights_path = Path(cropping_model_weights_path_or_url)

        if denoising_model_weights_path_or_url is None:
            denoising_model_weights_path = self.ensure_weights(weights_dir / "denoising_model.pth", self.get_metadata()["weights_urls"]["Denoising Model"])
        elif denoising_model_weights_path_or_url.startswith(("http://", "https://")):
            denoising_model_weights_path = self.download_weights(denoising_model_weights_path_or_url, self.get_model_cache_path() / "denoising_model.pth")
        else:
            denoising_model_weights_path = Path(denoising_model_weights_path_or_url)

        if nodesdetection_model_weights_path_or_url is None:
            nodesdetection_model_weights_path = self.ensure_weights(weights_dir / "nodesDetection_model.pth", self.get_metadata()["weights_urls"]["Nodes Detection Model"])
        elif nodesdetection_model_weights_path_or_url.startswith(("http://", "https://")):
            nodesdetection_model_weights_path = self.download_weights(nodesdetection_model_weights_path_or_url, self.get_model_cache_path() / "nodesDetection_model.pth")
        else:
            nodesdetection_model_weights_path = Path(nodesdetection_model_weights_path_or_url)

        self.cropping_model.load_state_dict(torch.load(cropping_model_weights_path, map_location=device))
        self.denoising_model.load_state_dict(torch.load(denoising_model_weights_path, map_location=device))
        self.nodesDetection_model.load_state_dict(torch.load(nodesdetection_model_weights_path, map_location=device))

        self.cropping_model.eval()
        self.denoising_model.eval()
        self.nodesDetection_model.eval()
        self.textsDetector_model = get_textsDetector_model()

        print("Models loaded")
        self._loaded = True

    def convert(self, path_or_array):
        self.ensure_loaded()

        img_rgb_resized, img_tensor, (H, W) = self.load_and_preprocess(path_or_array)

        cropped_trees = self.extract_tree([img_rgb_resized])
        cleaned_trees = self.clean_tree(cropped_trees)
        nodes_by_image, corners_by_image = self.detect_nodes(cleaned_trees)
        texts_by_image = self.detect_texts([img_rgb_resized])

        newick = self.build_newick(nodes_by_image[0], corners_by_image[0], texts_by_image[0])

        print(f"Conversion finished")
        return newick

    def load_and_preprocess(self, path_or_array):
        print("Loading and Preprocessing image...")
        img_rgb, img_tensor, (H, W) = load_and_preprocess_image(path_or_array, size=(1500, 1500))
        print(f"Image loaded: original size (H={H}, W={W}), tensor shape: {img_tensor.shape}")
        return img_rgb, img_tensor, (H, W)

    def extract_tree(self, imgs_rgb):
        print("Extracting trees...")
        trees = extract_tree_from_image(imgs_rgb, self.cropping_model, (500, 500))
        print(f"Trees shapes obtained: {[t.shape for t in trees]}")
        self._save_debug_images(trees, prefix="cropped")
        return trees

    def clean_tree(self, cropped_trees):
        print("Cleaning trees...")
        cleaned_trees = denoise_image(cropped_trees, self.denoising_model, (512, 512))
        print(f"Cleaned trees shapes obtained: {[t.shape for t in cleaned_trees]}")
        self._save_debug_images(cleaned_trees, prefix="cleaned")
        return cleaned_trees

    def _save_debug_images(self, imgs, prefix="debug"):
        debug_dir = Path("/home/bg/itpt_debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(imgs):
            if isinstance(img, np.ndarray):
                out = img
                if out.dtype != np.uint8:
                    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
                path = debug_dir / f"{prefix}_{i}.png"
                cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR) if out.ndim == 3 and out.shape[2] == 3 else out)
                print(f"Debug image saved: {path}")

    def detect_nodes(self, cleaned_trees):
        print("Detecting nodes...")
        device = next(self.nodesDetection_model.parameters()).device
        nodes_by_image, corners_by_image = detect_nodes_batch(
            imgs_rgb=cleaned_trees,
            model=self.nodesDetection_model,
            device=device,
            img_size=1500,
            hm_size=1000,
            threshold=0.3,
            nms_size=3,
        )

        print(f"Number of detected nodes: {[len(n) for n in nodes_by_image]}")
        print(f"Number of detected corners: {[len(c) for c in corners_by_image]}")
        return nodes_by_image, corners_by_image

    def detect_texts(self, imgs_rgb):
        print("Detecting texts...")
        texts_by_image = detect_texts(imgs_rgb, self.textsDetector_model)
        print(f"Number of texts per image: {[len(t) for t in texts_by_image]}")
        return texts_by_image

    def build_newick(self, nodes, corners, texts):
        print("Building Newick...")
        if corners is None:
            points = nodes
        else:
            points = [Point(x, y, "node") for (x, y, _) in nodes] + \
                    [Point(x, y, "corner") for (x, y, _) in corners]

        newick = build_newick(points, texts=texts)
        print("Newick built: ", newick.to_string())
        return newick
