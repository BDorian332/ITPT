import shutil
from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).parents[1]

def clean_generated_models():
    models_dir = PROJECT_ROOT / "itpt" / "_data" / "models"
    if models_dir.exists():
        shutil.rmtree(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

def clean_build_artifacts():
    # Python build artifacts
    for d in ["build", "dist"]:
        path = PROJECT_ROOT / d
        if path.exists():
            shutil.rmtree(path)
    for path in PROJECT_ROOT.glob("*.egg-info"):
        if path.is_dir():
            shutil.rmtree(path)

    # PyInstaller outputs
    gui_build = PROJECT_ROOT / "gui" / "build"
    gui_dist = PROJECT_ROOT / "gui" / "dist"
    for path in [gui_build, gui_dist]:
        if path.exists():
            shutil.rmtree(path)

def main():
    parser = argparse.ArgumentParser(description="Clean ITPT generated files and build artifacts")
    parser.add_argument("--models", action="store_true", help="Clean generated models in _data")
    parser.add_argument("--build", action="store_true", help="Clean build/dist/egg-info artifacts")
    parser.add_argument("--all", action="store_true", help="Clean everything")
    args = parser.parse_args()

    if args.all or args.models:
        print("=== Cleaning generated models ===")
        clean_generated_models()

    if args.all or args.build:
        print("=== Cleaning build artifacts ===")
        clean_build_artifacts()

    if not (args.models or args.build or args.all):
        print("Nothing to clean. Use --models, --build, or --all.")

if __name__ == "__main__":
    main()
