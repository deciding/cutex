from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

STUB_CONTENT = {
    ROOT / "cutex" / "__init__.pyi": """from cutlass.cute import *

import cutlass.cute.nvgpu as nvgpu
from . import runtime as runtime
from . import testing as testing

from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

__all__: list[str]
""",
    ROOT / "cutex" / "pipeline.pyi": """from cutlass.pipeline import *

__all__: list[str]
""",
    ROOT / "cutex" / "runtime.pyi": """from cutlass.cute.runtime import *
from cutlass.cute.runtime import from_dlpack, make_fake_stream

__all__: list[str]
""",
    ROOT / "cutex" / "testing.pyi": """from cutlass.cute.testing import *

__all__: list[str]
""",
    ROOT / "cutex" / "utils" / "__init__.pyi": """from cutlass.utils import *

__all__: list[str]
""",
    ROOT
    / "cutex"
    / "utils"
    / "sm100.pyi": """from cutlass.utils.blackwell_helpers import *

__all__: list[str]
""",
}


def main() -> None:
    for path, content in STUB_CONTENT.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


if __name__ == "__main__":
    main()
