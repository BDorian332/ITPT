import sys
from pathlib import Path
import subprocess
import argparse

PROJECT_ROOT = Path(__file__).parents[1]
GUI_MAIN = PROJECT_ROOT / "gui" / "main.py"

def run_gui():
    if not GUI_MAIN.exists():
        print(f"GUI main.py not found at {GUI_MAIN}")
        return

    print("=== Running GUI ===")
    try:
        subprocess.run([sys.executable, str(GUI_MAIN)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run GUI: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run ITPT application")
    parser.add_argument("--gui", action="store_true", help="Run the GUI application")
    args = parser.parse_args()

    if args.gui:
        run_gui()

    if not (args.gui):
        print("=== Nothing to do. Use --gui ===")

if __name__ == "__main__":
    main()
