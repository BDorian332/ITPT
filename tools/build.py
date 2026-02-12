import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
MODELS_SRC_DIR = PROJECT_ROOT / "itpt" / "_data" / "models"
MODELS_TO_INCLUDE = ["v1"]

def build_lib():
    print("=== Building Python library ===")
    try:
        subprocess.run(["poetry", "build"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to build library: {e}")
        sys.exit(1)

def build_gui(version="v1", debug=False):
    gui_dir = PROJECT_ROOT / f"gui_{version}"
    gui_main = gui_dir / "main.py"

    print(f"=== Building standalone GUI ({version}) ===")
    if not gui_main.exists():
        print(f"GUI main.py not found at {gui_main}")
        return

    sep = ";" if os.name == "nt" else ":"

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
        "--clean",
        "--noconfirm",
    ]

    if system == "Darwin":
        cmd += ["--onedir"]
    else:
        cmd += ["--onefile"]

    if debug:
        cmd += ["--console"]
    else:
        cmd += ["--windowed"]

    cmd += [
        "--collect-all", "torch",
        "--collect-all", "cv2",
        "--collect-all", "doctr",
        *data_args,
        "--name", f"gui-{version}",
        str(gui_main)
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to build GUI ({version}): {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="ITPT Build Tool")
    parser.add_argument("--lib", action="store_true", help="Build Python library only")
    parser.add_argument("--gui", nargs="?", const="v1", choices=["v1", "v2"], help="Build standalone GUI (v1 or v2). Defaults to v1.")
    args = parser.parse_args()

    if args.lib:
        build_lib()

    if args.gui:
        build_gui(args.gui)

    if not (args.lib or args.gui):
        print("Nothing to do. Use --lib, or --gui [v1|v2].")

if __name__ == "__main__":
    main()
