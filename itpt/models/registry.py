import importlib.util
from importlib import resources
from itpt.core import Model

_REGISTRY = {}

def _scan_models():
    global _REGISTRY
    _REGISTRY.clear()

    models_package = "itpt._data.models"

    for model_name in resources.contents(models_package):
        if resources.is_resource(models_package, model_name):
            continue

        model_package = f"{models_package}.{model_name}"

        try:
            with resources.path(model_package, "model.py") as model_file:
                model_file_path = str(model_file)
        except FileNotFoundError:
            continue

        module_name = "_dynamic_" + model_name
        spec = importlib.util.spec_from_file_location(module_name, model_file_path)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Failed to load model '{model_name}': {e}")
            continue

        model_classes = [
            obj for obj in module.__dict__.values()
            if isinstance(obj, type) and issubclass(obj, Model) and obj is not Model
        ]

        if not model_classes:
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
