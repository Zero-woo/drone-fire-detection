"""Run from the repository root: python firelab.py --help."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from fire_detection.cli import main
    raise SystemExit(main())
