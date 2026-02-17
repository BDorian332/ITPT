from typing import List
from gui_v1.main import Step
from itpt.core.newick import Point, build_newick

STEPS = [
    Step("Extract Tree", default_enabled=True),
    Step("Clean Tree", default_enabled=True),
    Step("Detect Texts", default_enabled=True)
]

def run_steps(model, img_rgb, steps=None):
    output = [img_rgb]

    extract_step = next((s for s in steps if s.name == "Extract Tree"), None)
    if extract_step and extract_step.enabled:
        output = model.extract_tree(output)

    clean_step = next((s for s in steps if s.name == "Clean Tree"), None)
    if clean_step and clean_step.enabled:
        output = model.clean_tree(output)

    nodes_by_image, corners_by_image = model.detect_nodes(output)
    nodes = nodes_by_image[0]
    corners = corners_by_image[0]

    detect_texts_step = next((s for s in steps if s.name == "Detect Texts"), None)
    if detect_texts_step and detect_texts_step.enabled:
        texts_by_image = model.detect_texts([img_rgb])
        texts = texts_by_image[0]
    else:
        texts = []

    points = [Point(x, y, "node") for (x, y, _) in nodes] + \
             [Point(x, y, "corner") for (x, y, _) in corners]

    newick = build_newick(points, texts=texts)

    return newick, points, texts
