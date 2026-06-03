"""
Option Pricing Runner

Runs all derivatives pricing scripts.
"""

import subprocess
import sys
from pathlib import Path


SCRIPT_FILES = [
    "american_option_pricer.py",
    "asian_option_monte_carlo_pricer.py",
    "asian_option_box_muller_pricer.py"
]


def main() -> None:
    project_root = Path(__file__).parent

    for script_file in SCRIPT_FILES:
        print()
        print("=" * 90)
        print(f"Running {script_file}")
        print("=" * 90)

        subprocess.run(
            [sys.executable, str(project_root / script_file)],
            check=False
        )


if __name__ == "__main__":
    main()
