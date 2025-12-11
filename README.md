# ITPT

A Python library for converting images of phylogenetic trees into the Newick format.

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
        notebooks/
        generator/
            __init__.py
            generator.py
    tools/
        build.py
        clean.py
        generate_models.py
        run.py
    project.toml
```

## Usage

```python
from itpt.models import list, get

print(list())
model = get("model_a")
tree = model.convert("input.png")
print(tree)
```

## Contributors

- Barrère Dorian : [barreredorian332@gmail.com](mailto:barreredorian332@gmail.com)
- Accini Arthur : [acciniarth@cy-tech.fr](mailto:acciniarth@cy-tech.fr)
- Chamaillard Lucas : [lchamaillard.lucas@gmail.com](mailto:lchamaillard.lucas@gmail.com)
- Reyne Matteo : [reynematte@cy-tech.fr](mailto:reynematte@cy-tech.fr)
