from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self):
        self._loaded: bool = False
        self._metadata: Dict[str, Any] = {
            "name": "Unknown",
            "description": "No description.",
            "input_spec": "any",
            "version": 0
        }

    def get_metadata(self):
        return self._metadata

    @abstractmethod
    def get_input_spec(self):
        pass

    @abstractmethod
    def load(self, weights_path = None):
        pass

    def ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("Model must be loaded before inference.")

    @abstractmethod
    def convert(self, image):
        pass
