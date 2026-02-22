import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
MODELS_TO_INCLUDE = ["v1"]

def build_lib():
    print("=== Building Python library ===")
    try:
        subprocess.run(["poetry", "build"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to build library: {e}")
        sys.exit(1)

def build_gui(version, debug):
    gui_dir = PROJECT_ROOT / f"gui_{version}"
    gui_main = gui_dir / "main.py"

    print(f"=== Building standalone GUI ({version}) ===")
    if not gui_main.exists():
        print(f"GUI main.py not found at {gui_main}")
        return

    sep = ";" if os.name == "nt" else ":"

    data_args = []

    models_src_dir = PROJECT_ROOT / "itpt" / "_data" / "models"
    for model_name in MODELS_TO_INCLUDE:
        model_path = models_src_dir / model_name
        if model_path.exists():
            for root, dirs, files in os.walk(model_path):
                if "__pycache__" in dirs:
                    dirs.remove("__pycache__")
                if "weights" in dirs:
                    dirs.remove("weights")

                for file in files:
                    source_file = Path(root) / file
                    rel_path = source_file.relative_to(PROJECT_ROOT)
                    dest_dir = rel_path.parent

                    data_args.extend(["--add-data", f"{source_file}{sep}{dest_dir}"])

            print(f"Including model: {model_name}")
        else:
            print(f"Warning: Model folder {model_name} not found")

    cmd = [
        "pyinstaller",
        "--clean",
        "--noconfirm",
    ]

    if platform.system() == "Darwin":
        cmd += ["--onedir"]
    else:
        cmd += ["--onefile"]

    if debug:
        cmd += ["--console"]
    else:
        cmd += ["--windowed"]

    cmd += [
        "--collect-submodules", "itpt",
        "--exclude-module", "itpt._data",
        "--collect-submodules", f"gui_{version}",
        "--collect-submodules", "torch",
        "--collect-submodules", "cv2",
        "--collect-submodules", "doctr",
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
    parser.add_argument("--debug", action="store_true", help="Run bundled application in console/debug mode (do not hide terminal)")
    args = parser.parse_args()

    if args.lib:
        build_lib()

    if args.gui:
        build_gui(args.gui, args.debug)

    if not (args.lib or args.gui):
        print("Nothing to do. Use --lib, or --gui [v1|v2]. You can also use --debug.")

if __name__ == "__main__":
    main()
