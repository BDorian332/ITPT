import urllib.request
from pathlib import Path
from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self):
        self._loaded = False
        self._metadata = {
            "name": "Unknown",
            "description": "No description.",
            "input_spec": "Unknown",
            "version": 0
        }

    def get_metadata(self):
        return self._metadata

    def ensure_weights(self, weights_path, url, model_name):
        print(f"Checking {weights_path.name}")
        if weights_path.exists():
            print("Using local file")
            return weights_path

        home_dir = Path.home()
        fallback_dir = home_dir / ".cache" / "itpt" / "weights" / model_name
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / weights_path.name

        if fallback_path.exists():
            print("Using cached file")
            return fallback_path

        print(f"Weights not found. Downloading {weights_path.name} for model {model_name} into {fallback_path}...")
        try:
            urllib.request.urlretrieve(url, fallback_path)
            print(f"Downloaded {weights_path.name} successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to download {weights_path.name} from {url}: {e}")

        return fallback_path

    @abstractmethod
    def load(self):
        pass

    def ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("Model must be loaded before inference.")

    @abstractmethod
    def convert(self, img_path):
        pass
