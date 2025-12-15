import sys
import argparse
from pathlib import Path
from dev.generators import *

def main():
    parser = argparse.ArgumentParser(description="Generate ITPT models from notebooks")
    args = parser.parse_args()

    print("=== Generating models from notebooks ===")
    generate_all_from_notebook()

if __name__ == "__main__":
    main()
