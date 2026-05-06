from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cutez import Config, autotune, compile as cutez_compile


def test_public_autotune_exports_exist():
    assert Config is not None
    assert callable(autotune)
    assert callable(cutez_compile)
