import os
import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
GUI_MAIN = PROJECT_ROOT / "gui" / "main.py"
MODELS_SRC_DIR = PROJECT_ROOT / "itpt" / "_data" / "models"
MODELS_TO_INCLUDE = ["v1"]

def build_lib():
    print("=== Building Python library ===")
    try:
        subprocess.run(["poetry", "build"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to build library: {e}")
        sys.exit(1)

def build_gui():
    print("=== Building standalone GUI ===")
    if not GUI_MAIN.exists():
        print(f"GUI main.py not found at {GUI_MAIN}")
        return

    if os.name == "nt": # Windows
        sep = ";"
    else: # Linux / macOS
        sep = ":"

    data_args = []
    for model_name in MODELS_TO_INCLUDE:
        model_path = MODELS_SRC_DIR / model_name
        if model_path.exists():
            dest_path = f"itpt/_data/models/{model_name}"
            data_args.extend(["--add-data", f"{model_path}{sep}{dest_path}"])
            print(f"Including model: {model_name}")
        else:
            print(f"Warning: Model folder {model_name} not found in {MODELS_SRC_DIR}")

    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--clean",
        "--noconfirm",
        "--collect-all", "torch",
        "--collect-all", "cv2",
        "--collect-all", "doctr",
        *data_args,
        "--name", "gui",
        str(GUI_MAIN)
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to build GUI: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="ITPT Build Tool")
    parser.add_argument("--lib", action="store_true", help="Build Python library only")
    parser.add_argument("--gui", action="store_true", help="Build standalone GUI")
    args = parser.parse_args()

    if args.lib:
        build_lib()

    if args.gui:
        build_gui()

    if not (args.lib or args.gui):
        print("Nothing to do. Use --lib, or --gui.")

if __name__ == "__main__":
    main()
