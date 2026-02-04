import os
import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
GUI_MAIN = PROJECT_ROOT / "gui" / "main.py"
DATA_MODELS_DIR = PROJECT_ROOT / "itpt" / "_data" / "models"

def build_lib():
    print("=== Building Python library ===")
    try:
        subprocess.run(["poetry", "build"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to build library: {e}")
        sys.exit(1)

def build_gui():
    print("=== Building GUI standalone ===")
    if not GUI_MAIN.exists():
        print(f"GUI main.py not found at {GUI_MAIN}")
        return

    if os.name == "nt": # Windows
        sep = ";"
    else: # Linux / macOS
        sep = ":"

    add_data_option = f"{DATA_MODELS_DIR}{sep}_data/models"

    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--noconfirm",
        "--add-data", add_data_option,
        "-n", "gui",
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
