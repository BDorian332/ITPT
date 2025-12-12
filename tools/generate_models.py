import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "dev" / "generator"))

from generator import *

def main():
    parser = argparse.ArgumentParser(description="Generate ITPT models from notebooks")
    args = parser.parse_args()

    print("=== Generating models from notebooks ===")
    generate_all()

if __name__ == "__main__":
    main()
