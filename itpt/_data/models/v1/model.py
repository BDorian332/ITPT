import torch
import cv2
from pathlib import Path
from itpt.core import Model
from itpt.core import build_newick
from .preprocessing.cropping import extract_tree_from_image, CroppingModel
from .preprocessing.denoising import denoise_image, load_and_preprocess_image, DenoisingModel, img_to_tensor, img_to_gray
from .postprocessing.ocr import detect_texts, get_textsDetector_model

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
        self.nodesDetection_model = None
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

        self.cropping_model.eval()
        self.denoising_model.eval()
        self.textsDetector_model = get_textsDetector_model()

        print("Models loaded")
        self._loaded = True

    def convert(self, path_or_array):
        self.ensure_loaded()

        img_rgb_resized, img_tensor, (H, W) = self.load_and_preprocess(path_or_array)

        cropped_trees = self.extract_tree([img_rgb_resized])
        cleaned_trees = self.clean_tree(cropped_trees)
        nodes_by_image = self.detect_nodes(cleaned_trees)
        texts_by_image = self.detect_texts([img_rgb_resized])

        newick = self.build_newick(nodes_by_image[0], texts_by_image[0])

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
        print("Trees obtained, shape:", trees.shape)
        return trees

    def clean_tree(self, cropped_trees):
        print("Cleaning trees...")
        cleaned_trees = denoise_image(cropped_trees, self.denoising_model, (512, 512))
        print("Cleaned trees obtained, shape:", cleaned_trees.shape)
        return cleaned_trees

    def detect_nodes(self, cleaned_trees):
        print("Detecting nodes...")
        #...
        print(f"Detected nodes shape: {nodes_by_image}")
        return nodes_by_image

    def detect_texts(self, imgs_rgb):
        print("Detecting texts...")
        texts_by_image = detect_texts(imgs_rgb, self.textsDetector_model)
        print("Found texts shape: ", texts_by_image.shape)
        return texts_by_image

    def build_newick(self, nodes, corners, texts):
        print("Building Newick...")
        newick = build_newick(nodes, corners, texts=texts)
        print("Newick built: ", newick.to_string())
        return newick
