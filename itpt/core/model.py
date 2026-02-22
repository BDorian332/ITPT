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
            "version": 0,
            "weights_urls": {}
        }

    def get_metadata(self):
        return self._metadata

    def get_model_cache_path(self):
        return Path.home() / ".cache" / "itpt" / self.get_metadata()["name"]

    def download_weights(self, url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            return dest

        print(f"Downloading weights from {url} to {dest}")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"Downloaded {weights_path.name} successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to download {dest.name} from {url}: {e}")

        return dest

    def ensure_weights(self, weights_path, url):
        print(f"Checking {weights_path.name}")
        if weights_path.exists():
            print("Using local file")
            return weights_path

        fallback_dir = self.get_model_cache_path() / "weights"
        fallback_path = fallback_dir / weights_path.name

        if fallback_path.exists():
            print("Using cached file")
            return fallback_path

        print(f"Weights not found.")
        self.download_weights(url, fallback_path)

        return fallback_path

    @abstractmethod
    def load(self):
        pass

    def ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("Model must be loaded before inference.")

    @abstractmethod
    def convert(self, path_or_array):
        pass
