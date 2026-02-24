from typing import List
from gui_vtk.main import Step

STEPS = [
    Step("Extract Tree", default_enabled=False),
    Step("Clean Tree", default_enabled=False),
    Step("Detect Texts", default_enabled=True)
]

SETTINGS = {
    "margin": {"name": "Newick Margin", "type": float, "default": 5.0},
    "max_distance": {"name": "Newick Max Distance", "type": float, "default": 20.0},
}

def load_model(model, weights_overrides):
    model.load(
        cropping_model_weights_path_or_url=weights_overrides.get("Cropping Model"),
        denoising_model_weights_path_or_url=weights_overrides.get("Denoising Model"),
        nodesdetection_model_weights_path_or_url=weights_overrides.get("Nodes Detection Model")
    )

def build_newick(model, nodes_by_image, texts_by_image=None, settings=None):
    settings = settings or {}
    margin = settings.get("margin", SETTINGS["margin"]["default"])
    max_distance = settings.get("max_distance", SETTINGS["max_distance"]["default"])

    return model.build_newick(nodes_by_image, texts_by_image=texts_by_image, margin=margin, max_distance=max_distance)

def run_steps(model, img_rgb, steps=None, settings=None):
    img_rgb_resized, img_tensor, (H, W) = model.load_and_preprocess_image(img_rgb)
    output = [img_rgb_resized]

    extract_step = next((s for s in steps if s.name == "Extract Tree"), None)
    if extract_step and extract_step.enabled:
        output = model.extract_tree(output)

    clean_step = next((s for s in steps if s.name == "Clean Tree"), None)
    if clean_step and clean_step.enabled:
        output = model.clean_tree(output)

    nodes_by_image = model.detect_nodes(output)

    detect_texts_step = next((s for s in steps if s.name == "Detect Texts"), None)
    texts_by_image = None
    if detect_texts_step and detect_texts_step.enabled:
        texts_by_image = model.detect_texts([img_rgb_resized])

    newick_by_image = build_newick(model, nodes_by_image, texts_by_image, settings)
    return newick_by_image[0], nodes_by_image[0] if nodes_by_image else [], texts_by_image[0] if texts_by_image else []
