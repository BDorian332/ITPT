from importlib import resources
from itpt.core import Model

_REGISTRY = {}
BLACKLISTED_MODELS = ["v0"]

def _scan_models():
    global _REGISTRY
    _REGISTRY.clear()

    models_package = "itpt._data.models"

    for model_key in resources.contents(models_package):
        if resources.is_resource(models_package, model_key) or model_key in BLACKLISTED_MODELS:
            continue

        model_package = f"{models_package}.{model_key}"

        try:
            module = importlib.import_module(f"{model_package}.model")
        except Exception as e:
            print(f"Failed to load model '{model_key}': {e}")
            continue

        model_classes = [
            obj for obj in module.__dict__.values()
            if isinstance(obj, type) and issubclass(obj, Model) and obj is not Model
        ]

        if not model_classes:
            continue

        ModelClass = model_classes[0]

        try:
            instance = ModelClass()
        except Exception as e:
            print(f"Failed to instantiate model '{model_key}': {e}")
            continue

        model_name = instance._metadata.get("name")
        if not model_name:
            print(f"Model in '{model_name}' has no metadata name")
            continue

        _REGISTRY[model_name] = model_classes[0]

_scan_models()

def get_list():
    return list(_REGISTRY.keys())

def get_model(name):
    ModelClass = _REGISTRY.get(name)
    if ModelClass is None:
        raise ValueError(f"Model '{name}' not found. Available models: {list()}")
    return ModelClass()
