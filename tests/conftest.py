"""Make the runnable example modules importable, as they are when run from ``examples/``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
