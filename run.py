"""Repository-local entry point: works without editable .pth loading on macOS."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cgtn.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
