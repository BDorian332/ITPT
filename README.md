# ITPT

A Python library for converting phylogenetic tree images to Newick format.

## Project Structure

```
root/
    itpt/
        __init__.py
        core/
            __init__.py
            model.py
        models/
            __init__.py
            registry.py
        _data/
            models/
                ...
    gui/
        main.py
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
        build.py
        clean.py
        generate_models.py
        run.py
    pyproject.toml
```

## Requirements

- Python (>=3.13)
- Poetry

## Usage

### For Users

```python
from itpt.models import get_list, get_model

model_names = get_list()
print(model_names)

model = get_model("model_a")
model.load()

tree = model.convert("input.png")
print(tree)
```

### For Developers

#### Project Setup

```bash
poetry install
poetry install --with gui # installs needed dependencies for the GUI

{
poetry env activate # prints the activate command of the virtual environment to the console
jupyter notebook # to run the Jupyter notebook
}
OR
{
poetry run jupyter notebook # to directly run the Jupyter notebook with the Poetry virtual environment
}
```

#### Generate Models from Notebooks

```bash
poetry run itpt-generate-models
```

This produces directories and files under:

```
itpt/_data/models/<model_name>/
    code.py
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

#### Adding a New Model

1. Create a notebook in: notebooks/models/
2. Tag the cells you want exported with:

```json
{
    "export": true
}
```

## Contributors

- Barrère Dorian : [barreredorian332@gmail.com](mailto:barreredorian332@gmail.com)
- Accini Arthur : [acciniarth@cy-tech.fr](mailto:acciniarth@cy-tech.fr)
- Chamaillard Lucas : [lchamaillard.lucas@gmail.com](mailto:lchamaillard.lucas@gmail.com)
- Reyne Matteo : [reynematte@cy-tech.fr](mailto:reynematte@cy-tech.fr)
