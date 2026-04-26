# Cutex PyPI Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish this repository as a minimal `cutex` PyPI package that depends on `flash_attn_4==4.0.0b4` and re-exports the agreed `cutlass` APIs under `cutex`.

**Architecture:** Add a small forwarding Python package with only `cutex/__init__.py`, `cutex/pipeline.py`, `cutex/utils/__init__.py`, and `cutex/utils/sm100.py`. Use `pyproject.toml` for packaging metadata and add a minimal pytest-based import test suite that proves the public namespace works without modifying upstream `cutlass`.

**Tech Stack:** Python, setuptools, pytest, importlib-based module exports, PyPI packaging metadata

---

## File Structure

Files to create or modify for this work:

- Create: `pyproject.toml`
- Create: `cutex/__init__.py`
- Create: `cutex/pipeline.py`
- Create: `cutex/utils/__init__.py`
- Create: `cutex/utils/sm100.py`
- Create: `tests/test_package_imports.py`

Responsibilities:

- `pyproject.toml`: package metadata, dependency declaration, package discovery, pytest test dependency configuration
- `cutex/__init__.py`: top-level forwarding namespace for `cutlass.cute`, `cutlass.cute.nvgpu`, `cutlass.cute.runtime`, and `cutlass.cute.testing`
- `cutex/pipeline.py`: forwarding surface for `cutlass.pipeline`
- `cutex/utils/__init__.py`: forwarding surface for `cutlass.utils`
- `cutex/utils/sm100.py`: forwarding surface for `cutlass.utils.blackwell_helpers`
- `tests/test_package_imports.py`: import-first regression tests for the agreed public API

### Task 1: Establish Packaging Metadata

**Files:**
- Create: `pyproject.toml`
- Test: `tests/test_package_imports.py`

- [ ] **Step 1: Write the failing test**

```python
from importlib.metadata import PackageNotFoundError, metadata


def test_project_metadata_is_not_installed_yet():
    try:
        metadata("cutex")
    except PackageNotFoundError:
        assert True
        return

    raise AssertionError("cutex metadata unexpectedly exists before packaging is added")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_package_imports.py::test_project_metadata_is_not_installed_yet -v`
Expected: FAIL because `pytest` or the `tests/test_package_imports.py` file does not exist yet, confirming the test harness and package metadata still need to be created.

- [ ] **Step 3: Write minimal implementation**

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "cutex"
version = "0.1.0"
description = "Thin cutlass cute re-export package"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "flash_attn_4==4.0.0b4",
]

[project.optional-dependencies]
test = ["pytest>=8", "build>=1"]

[tool.setuptools.packages.find]
include = ["cutex", "cutex.*"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m build --sdist --wheel`
Expected: PASS with build artifacts created under `dist/`, proving `pyproject.toml` is valid and the project is packageable.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build: add cutex packaging metadata"
```

### Task 2: Add Import Regression Tests First

**Files:**
- Create: `tests/test_package_imports.py`
- Test: `tests/test_package_imports.py`

- [ ] **Step 1: Write the failing test**

```python
import importlib


def test_cutex_top_level_surface():
    cutex = importlib.import_module("cutex")
    assert hasattr(cutex, "cpasync")
    assert hasattr(cutex, "tcgen05")
    assert hasattr(cutex, "from_dlpack")
    assert hasattr(cutex, "nvgpu")
    assert hasattr(cutex, "runtime")
    assert hasattr(cutex, "testing")


def test_cutex_pipeline_surface():
    pipeline = importlib.import_module("cutex.pipeline")
    assert hasattr(pipeline, "pipeline_init_arrive")
    assert hasattr(pipeline, "pipeline_init_wait")


def test_cutex_utils_surface():
    utils = importlib.import_module("cutex.utils")
    sm100 = importlib.import_module("cutex.utils.sm100")
    assert utils is not None
    assert sm100 is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_package_imports.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cutex'` because the forwarding package has not been created yet.

- [ ] **Step 3: Write minimal implementation**

```python
# tests/test_package_imports.py
import importlib


def test_cutex_top_level_surface():
    cutex = importlib.import_module("cutex")
    assert hasattr(cutex, "cpasync")
    assert hasattr(cutex, "tcgen05")
    assert hasattr(cutex, "from_dlpack")
    assert hasattr(cutex, "nvgpu")
    assert hasattr(cutex, "runtime")
    assert hasattr(cutex, "testing")


def test_cutex_pipeline_surface():
    pipeline = importlib.import_module("cutex.pipeline")
    assert hasattr(pipeline, "pipeline_init_arrive")
    assert hasattr(pipeline, "pipeline_init_wait")


def test_cutex_utils_surface():
    utils = importlib.import_module("cutex.utils")
    sm100 = importlib.import_module("cutex.utils.sm100")
    assert utils is not None
    assert sm100 is not None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_package_imports.py -v`
Expected: still FAIL with `ModuleNotFoundError: No module named 'cutex'`. This is the correct red state before implementing the package modules.

- [ ] **Step 5: Commit**

```bash
git add tests/test_package_imports.py
git commit -m "test: add cutex import surface regression tests"
```

### Task 3: Implement `cutex` Top-Level Forwarding Surface

**Files:**
- Create: `cutex/__init__.py`
- Test: `tests/test_package_imports.py`

- [ ] **Step 1: Write the failing test**

```python
import importlib


def test_cutex_re_exports_representative_cute_symbols():
    upstream_cute = importlib.import_module("cutlass.cute")
    cutex = importlib.import_module("cutex")
    assert set(upstream_cute.__all__).issubset(set(cutex.__all__))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_package_imports.py::test_cutex_re_exports_representative_cute_symbols -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cutex'` because `cutex/__init__.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from cutlass.cute import *
from cutlass.cute import __all__ as cute_all

import cutlass.cute.nvgpu as nvgpu
import cutlass.cute.runtime as runtime
import cutlass.cute.testing as testing

from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

__all__ = list(cute_all) + [
    "nvgpu",
    "runtime",
    "testing",
    "cpasync",
    "tcgen05",
    "from_dlpack",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_package_imports.py::test_cutex_re_exports_representative_cute_symbols tests/test_package_imports.py::test_cutex_top_level_surface -v`
Expected: PASS, showing `cutex` behaves like the top-level `cute` namespace and exposes the agreed extra attributes.

- [ ] **Step 5: Commit**

```bash
git add cutex/__init__.py tests/test_package_imports.py
git commit -m "feat: add cutex top-level cute re-exports"
```

### Task 4: Implement `cutex.pipeline`

**Files:**
- Create: `cutex/pipeline.py`
- Test: `tests/test_package_imports.py`

- [ ] **Step 1: Write the failing test**

```python
import importlib


def test_cutex_pipeline_re_exports_functions():
    pipeline = importlib.import_module("cutex.pipeline")
    assert hasattr(pipeline, "pipeline_init_arrive")
    assert hasattr(pipeline, "pipeline_init_wait")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_package_imports.py::test_cutex_pipeline_re_exports_functions -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cutex.pipeline'` because `cutex/pipeline.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from cutlass.pipeline import *
from cutlass.pipeline import __all__ as pipeline_all

__all__ = list(pipeline_all)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_package_imports.py::test_cutex_pipeline_re_exports_functions tests/test_package_imports.py::test_cutex_pipeline_surface -v`
Expected: PASS, confirming `cutex.pipeline` forwards the required functions.

- [ ] **Step 5: Commit**

```bash
git add cutex/pipeline.py tests/test_package_imports.py
git commit -m "feat: add cutex pipeline forwarding module"
```

### Task 5: Implement `cutex.utils` and `cutex.utils.sm100`

**Files:**
- Create: `cutex/utils/__init__.py`
- Create: `cutex/utils/sm100.py`
- Test: `tests/test_package_imports.py`

- [ ] **Step 1: Write the failing test**

```python
import importlib


def test_cutex_utils_modules_import():
    utils = importlib.import_module("cutex.utils")
    sm100 = importlib.import_module("cutex.utils.sm100")
    assert hasattr(utils, "__dict__")
    assert hasattr(sm100, "__dict__")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_package_imports.py::test_cutex_utils_modules_import -v`
Expected: FAIL with `ModuleNotFoundError` for `cutex.utils` because the utils forwarding modules do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# cutex/utils/__init__.py
from cutlass.utils import *
from cutlass.utils import __all__ as utils_all

__all__ = list(utils_all)
```

```python
# cutex/utils/sm100.py
from cutlass.utils.blackwell_helpers import *
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_package_imports.py::test_cutex_utils_modules_import tests/test_package_imports.py::test_cutex_utils_surface -v`
Expected: PASS, confirming both forwarding modules import successfully.

- [ ] **Step 5: Commit**

```bash
git add cutex/utils/__init__.py cutex/utils/sm100.py tests/test_package_imports.py
git commit -m "feat: add cutex utils forwarding modules"
```

### Task 6: Verify End-to-End Packaging Surface

**Files:**
- Modify: `tests/test_package_imports.py`
- Test: `tests/test_package_imports.py`

- [ ] **Step 1: Write the failing test**

```python
import importlib


def test_all_promised_imports_work_together():
    cutex = importlib.import_module("cutex")
    pipeline = importlib.import_module("cutex.pipeline")
    utils = importlib.import_module("cutex.utils")
    sm100 = importlib.import_module("cutex.utils.sm100")

    assert cutex.cpasync is not None
    assert cutex.tcgen05 is not None
    assert cutex.from_dlpack is not None
    assert pipeline.pipeline_init_arrive is not None
    assert pipeline.pipeline_init_wait is not None
    assert utils is not None
    assert sm100 is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_package_imports.py::test_all_promised_imports_work_together -v`
Expected: If any forwarding module is incomplete, FAIL with the missing import or attribute. If it passes immediately, tighten the assertions to verify at least one representative `cutlass.cute` symbol and the three extra top-level exports.

- [ ] **Step 3: Write minimal implementation**

```python
# tests/test_package_imports.py
import importlib


def test_all_promised_imports_work_together():
    cutex = importlib.import_module("cutex")
    pipeline = importlib.import_module("cutex.pipeline")
    utils = importlib.import_module("cutex.utils")
    sm100 = importlib.import_module("cutex.utils.sm100")

    assert hasattr(cutex, "cpasync")
    assert hasattr(cutex, "tcgen05")
    assert hasattr(cutex, "from_dlpack")
    assert hasattr(cutex, "nvgpu")
    assert hasattr(cutex, "runtime")
    assert hasattr(cutex, "testing")
    assert hasattr(pipeline, "pipeline_init_arrive")
    assert hasattr(pipeline, "pipeline_init_wait")
    assert utils is not None
    assert sm100 is not None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_package_imports.py -v`
Expected: PASS for the full import suite.

Run: `python -m build --sdist --wheel`
Expected: PASS with source and wheel artifacts created in `dist/`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_package_imports.py pyproject.toml cutex/__init__.py cutex/pipeline.py cutex/utils/__init__.py cutex/utils/sm100.py
git commit -m "test: verify cutex packaging surface end to end"
```
