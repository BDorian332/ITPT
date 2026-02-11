# ITPT

A Python library for converting phylogenetic tree images to Newick format.

## Project Structure

```
root/
    itpt/
        __init__.py
        core/
            __init__.py
            branches.py
            model.py
            newick.py

        models/
            __init__.py
            registry.py
        _data/
            models/
                ...
    gui_v0/
	    __init__.py
        main.py
    gui_v1/
        __init__.py
        ...
    dev/
	_datasets/
		...
        _notebooks/
		...
        generators/
            __init__.py
            generator_from_notebook.py
    sandbox/
        main.py
    tools/
	    __init__.py
        build.py
        clean.py
        generate_models.py
        run.py
    pyproject.toml
```

## Requirements

- Python (>=3.13,<3.15)
- Poetry

## Usage

### Installing ITPT from a `.whl` file

You can install ITPT directly from a pre-built wheel file (`.whl`). Follow these steps:

1. **Download the latest release**

Go to the [ITPT releases page](https://github.com/BDorian332/ITPT/releases) and download the `.whl` file from the latest release.

2. **Run the installation command**

```bash
pip install <path to the .whl just downloaded>
```

3. **Verify the installation**

```bash
pip show itpt
```

### For Users

#### Basic Usage

```python
from itpt.models import get_list, get_model

model_names = get_list()
print(model_names)

model = get_model("<model_name>")
model.load()

newick = model.convert("<input_path>")
print(newick.to_string())
```

#### Available Models

**V1** - Cropping + Cleaning + Heatmap

**Methods**

1. **Loading the Model**

- Method: `load(cropping_model_weights_path=None, denoising_model_weights_path=None, nodesdetection_model_weights_path=None)`
- Description: Loads the model weights for cropping, denoising and nodes detection neural networks, and initializes the OCR text detection model.
- Parameters:
  - `cropping_model_weights_path` (str, optional): path to the pre-trained cropping model weights. Defaults to `weights/cropping_model.pth` relative to the model directory.
  - `denoising_model_weights_path` (str, optional): path to the pre-trained denoising model weights. Defaults to `weights/denoising_model.pth` relative to the model directory.
  - `nodesdetection_model_weights_path` (str, optional): path to the pre-trained nodes detection model weights. Defaults to `weights/nodesdetection_model_weights.pth` relative to the model directory.
- Notes: Sets the internal flag `_loaded = True` after successful loading.

2. **Conversion of Tree Images**

- Method: `convert(image_path)`
- Description: Converts an input image containing a phylogenetic tree into a Newick format string via a sequential pipeline.
- Parameters:
  - `image_path` (str): path to the input image
- Steps:
  1. Calls `load_and_preprocess(image_path)` to read the image, resize it to 1500x1500px, and convert it to a tensor.
  2. Calls `extract_tree([img_rgb])` to extract the tree region using the `CroppingModel`.
  3. Calls `clean_tree(cropped_trees)` to denoise the result with the `DenoisingModel`.
  4. Calls `detect_nodes(cleaned_trees)` to identify topological nodes using `NodesDetectionModel`.
  5. Calls `detect_texts([img_rgb])` to run OCR and find texts on the original image.
  6. Calls `build_newick(nodes_by_image[0][0], nodes_by_image[0][1], texts)` to generate the final object.
- Returns: `Newick` object representing the tree.

3. **Supporting Methods**

- `load_and_preprocess(image_path)`:
  - Loads the image and prepares a tensor.
  - Returns `(img_rgb, img_tensor, (H, W))`.
- `extract_tree(imgs_rgb)`:
  - Uses `CroppingModel` to locate and crop trees from a list of images.
  - Returns `trees` (list of arrays of cropped images).
- `clean_tree(cropped_trees)`:
  - Uses `DenoisingModel` to clean the extracted trees.
  - Returns `cleaned_trees` (list of arrays of cleaned images).
- `detect_nodes(cleaned_trees)`:
  - Uses `NodesDetectionModel` to analyzes cleaned images to detect junctions and tips.
  - Returns `nodes_by_image` (list of pairs of lists of `Point` representing images nodes (the first list of the pair contains internal nodes and leaves, the second list contains corners)).
- `detect_texts(imgs_rgb)`:
  - Uses OCR (`texts_detector_model`) to extract text from the images.
  - Returns `texts_by_image` (list of lists of texts of original images. Each text is represented by its string and its bounding box).
- `build_newick(nodes, corners, texts)`:
  - Constructs a Newick object from detected nodes, corners, and texts.
  - Returns `newick` object.

### For Developers

#### Project Setup

```bash
poetry install
poetry install --with gui # installs needed dependencies for the GUI

{
eval $(poetry env activate) # to activate the virtual environment
jupyter notebook # to run Jupyter Notebook
}
OR
{
poetry run jupyter notebook # to directly run Jupyter notebook with the Poetry virtual environment
}
```

#### Adding a New Model

1. Create a notebook in: notebooks/models/
2. Tag the cells you want exported with:

```json
{
    "export": true
}
```

#### Generate Models from Notebooks

```bash
poetry run itpt-generate-models
```

This produces directories and files under:

```
itpt/_data/models/<model_name>/
    model.py
    ? (if a pre-trained model can be generated manually with the notebook)
```

#### Clean temporary files

```bash
poetry run itpt-clean --models
```

```bash
poetry run itpt-clean --build
```

```bash
poetry run itpt-clean --run
```

```bash
poetry run itpt-clean --all
```

#### Build the Python Library

```bash
poetry run itpt-build --lib
```

#### Build the Standalone GUI

```bash
poetry run itpt-build --gui
```

#### Run GUI on-the-fly

```bash
poetry run itpt-run --gui
```

#### Run Sandbox on-the-fly

```bash
poetry run itpt-run --sandbox
```

## Contributors

- Barrère Dorian : [barreredorian332@gmail.com](mailto:barreredorian332@gmail.com)
- Accini Arthur : [acciniarth@cy-tech.fr](mailto:acciniarth@cy-tech.fr)
- Chamaillard Lucas : [lchamaillard.lucas@gmail.com](mailto:lchamaillard.lucas@gmail.com)
- Reyne Matteo : [reynematte@cy-tech.fr](mailto:reynematte@cy-tech.fr)
