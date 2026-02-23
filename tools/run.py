import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
SANDBOX_MAIN = PROJECT_ROOT / "sandbox" / "main.py"

def run_gui(version):
    gui_module = f"gui_{version}.main"
    print(f"=== Running GUI {version} ===")
    try:
        subprocess.run([sys.executable, "-m", gui_module], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run GUI {version}: {e}")
        sys.exit(1)

def run_sandbox():
    if not SANDBOX_MAIN.exists():
        print(f"Sandbox main.py not found at {SANDBOX_MAIN}")
        return

    print("=== Running Sandbox ===")
    try:
        subprocess.run([sys.executable, str(SANDBOX_MAIN)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run Sandbox: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run ITPT application")
    parser.add_argument("--gui", nargs="?", const="vtk", choices=["vtk", "vctk"], help="Run the GUI application (vtk or vctk). Defaults to vtk.")
    parser.add_argument("--sandbox", action="store_true", help="Run the sandbox example")
    args = parser.parse_args()

    if args.gui:
        run_gui(args.gui)

    if args.sandbox:
        run_sandbox()

    if not (args.gui or args.sandbox):
        print("Nothing to do.")

if __name__ == "__main__":
    main()
