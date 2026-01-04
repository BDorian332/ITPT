import nbformat
import json
from nbconvert import PythonExporter
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parents[1] / "_notebooks"
DATA_MODELS_DIR = Path(__file__).parents[2] / "itpt" / "_data" / "models"

EXPORT_TAG = "export"

def generate_model_from_notebook(nb_path: Path):
    nb = nbformat.read(nb_path, as_version=4)

    exported_cells = [
        cell for cell in nb.cells
        if cell.cell_type == "code" and EXPORT_TAG in cell.get("metadata", {}).get("tags", [])
    ]

    if not exported_cells:
        print(f"No cell to export in {nb_path.name}, skipping.")
        return

    exporter = PythonExporter()
    code, _ = exporter.from_notebook_node(nbformat.NotebookNode(cells=exported_cells))

    namespace = {}
    try:
        exec(code, namespace)
    except Exception as e:
        print(f"Error executing notebook {nb_path.name}: {e}")
        return

    model_classes = [
        obj for obj in namespace.values()
        if isinstance(obj, type) and issubclass(obj, Model) and obj is not Model
    ]

    if not model_classes:
        print(f"No Model subclass found in {nb_path.name}, skipping.")
        return

    ModelClass = model_classes[0]

    try:
        temp_instance = ModelClass()
        model_name = temp_instance.get_metadata().get("name", nb_path.stem)
    except Exception as e:
        print(f"Failed to instantiate model in {nb_path.name}: {e}")
        return

    model_dir = DATA_MODELS_DIR / model_name.replace(" ", "_")

    if model_dir.exists():
        shutil.rmtree(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    code_file = model_dir / "code.py"
    with open(code_file, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"Model '{model_name}' generated at {model_dir}")

def generate_all_from_notebook():
    if not NOTEBOOKS_DIR.exists():
        print(f"Notebooks directory {NOTEBOOKS_DIR} does not exist!")
        return

    notebooks = list(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found.")
        return

    for nb_path in notebooks:
        generate_model_from_notebook(nb_path)
