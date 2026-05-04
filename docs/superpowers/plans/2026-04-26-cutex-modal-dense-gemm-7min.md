# Cutex Modal Dense GEMM 7min Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the worktree copy of `modal/01_dense_gemm.py` install local `cutex` and run `dense_gemm_7min.py` through Modal using the `cutex` namespace.

**Architecture:** Extend the existing Modal image build so it mounts the worktree repository and installs local `cutex`, then finish the `dense_gemm_7min` runner branch inside `01_dense_gemm.py`. Update `dense_gemm_7min.py` only where its imports must move from `cutlass.*` to the current `cutex` package surface, then verify end to end with `modal run 01_dense_gemm.py` using the required tee-based logging pattern.

**Tech Stack:** Python, Modal, local pip package installation, CuTe DSL, cutex, torch, nvidia-cutlass-dsl

---

## File Structure

Files to create or modify for this work:

- Modify: `/mnt/ssd1/zining/cutex/.worktrees/cutex-pypi/modal/01_dense_gemm.py`
- Modify: `/mnt/ssd1/zining/cutex/.worktrees/cutex-pypi/modal/blackwell/dense_gemm_7min.py`

Responsibilities:

- `modal/01_dense_gemm.py`: Modal image setup, local repository mount, local `cutex` installation, benchmark selection, `dense_gemm_7min` execution path
- `modal/blackwell/dense_gemm_7min.py`: `cutex`-based imports for the kernel entry path if the current direct `cutlass.*` imports prevent running under the package we just built

### Task 1: Mount And Install Local `cutex` In The Modal Image

**Files:**
- Modify: `/mnt/ssd1/zining/cutex/.worktrees/cutex-pypi/modal/01_dense_gemm.py`

- [ ] **Step 1: Write the failing test**

Add a temporary image-build verification command inside `modal/01_dense_gemm.py` after the local repo mount is introduced so the image proves `cutex` is not yet installable from the intended mounted path. Use this exact command in the image chain before implementation is complete:

```python
.run_commands(
    "python -c \"import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec('cutex') else 1)\""
)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
modal run 01_dense_gemm.py > >(tee out.log) 2> >(tee error.log >&2)
```

Expected: FAIL during image build or startup because `cutex` is not yet installed into the Modal image from the worktree package path.

- [ ] **Step 3: Write minimal implementation**

Update the image setup in `modal/01_dense_gemm.py` to mount the worktree repository and install `cutex` from that mounted path. Keep the existing `blackwell` mount and add only what is needed for local package installation.

Use this structure in the image definition:

```python
cutlass_image = (
    cutlass_image.pip_install("torch", "pytest")
    .pip_install("nvidia-cutlass-dsl>=4.4.1")
    .pip_install("triton==3.5.1")
    .pip_install("teraxlang==3.5.1.dev4")
    .add_local_dir(
        root_dir.parent,
        remote_path="/workspace/cutex_repo",
        ignore=lambda path: ".worktrees" in str(path),
    )
    .run_commands("python -m pip install /workspace/cutex_repo")
    .add_local_dir(
        root_dir / "blackwell",
        remote_path="/workspace/cuteDSL/blackwell",
        ignore="cutlass",
    )
)
```

If the ignore lambda form is not supported by the current Modal version in this repo, replace it with the minimal supported ignore form that excludes `.worktrees` content while still mounting the root package files required for `pip install /workspace/cutex_repo`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
modal run 01_dense_gemm.py > >(tee out.log) 2> >(tee error.log >&2)
```

Expected: Image build passes the local package install step. The overall run may still fail later on `dense_gemm_7min` import or execution, which is acceptable at this stage.

- [ ] **Step 5: Commit**

```bash
git add /mnt/ssd1/zining/cutex/.worktrees/cutex-pypi/modal/01_dense_gemm.py
git commit -m "build: install local cutex in modal image"
```

### Task 2: Validate And Finish The Existing `dense_gemm_7min` Runner Branch

**Files:**
- Modify: `/mnt/ssd1/zining/cutex/.worktrees/cutex-pypi/modal/01_dense_gemm.py`

- [ ] **Step 1: Write the failing test**

Confirm the existing `dense_gemm_7min` branch in `run_dense_gemm()` calls the imported runner with the same argument structure used by `dense_gemm_7`. Use this target shape in the code path:

```python
if "dense_gemm_7min" in RUN_TESTS:
    print("\n=== 15. dense_gemm_7min.py Benchmark ===")
    from cuteDSL.blackwell.dense_gemm_7min import (
        run_dense_gemm as run_dense_gemm_7min,
    )

    us = run_dense_gemm_7min(
        (M, N, K),
        tolerance=0.1,
        warmup_iterations=warmup,
        iterations=repeats,
        skip_ref_check=False,
        init_mode=INIT_MODE,
        normal_mean=NORMAL_MEAN,
        normal_std=NORMAL_STD,
    )
    time_ms = us / 1000
    tflops7min = flops / time_ms / 1e9
    print(f"dense_gemm_7min: {time_ms:.4f} ms, {tflops7min:.2f} TFLOPS")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
modal run 01_dense_gemm.py > >(tee out.log) 2> >(tee error.log >&2)
```

Expected: If the existing branch is incomplete or the import path is wrong, FAIL with an import error, syntax error, or runtime exception in the `dense_gemm_7min` branch.

- [ ] **Step 3: Write minimal implementation**

Complete or correct the existing `dense_gemm_7min` branch in `modal/01_dense_gemm.py` so it fully matches the finished `dense_gemm_7` branch pattern, including the final timing and TFLOPS print.

Use this exact completion block if the file is currently truncated after `normal_mean`:

```python
    us = run_dense_gemm_7min(
        (M, N, K),
        tolerance=0.1,
        warmup_iterations=warmup,
        iterations=repeats,
        skip_ref_check=False,
        init_mode=INIT_MODE,
        normal_mean=NORMAL_MEAN,
        normal_std=NORMAL_STD,
    )
    time_ms = us / 1000
    tflops7min = flops / time_ms / 1e9
    print(f"dense_gemm_7min: {time_ms:.4f} ms, {tflops7min:.2f} TFLOPS")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
modal run 01_dense_gemm.py > >(tee out.log) 2> >(tee error.log >&2)
```

Expected: The run reaches the `dense_gemm_7min` module import and executes far enough to reveal whether import migration is still needed. If it now fails inside `dense_gemm_7min.py` because of the old `cutlass.*` imports, that is the correct handoff to Task 3.

- [ ] **Step 5: Commit**

```bash
git add /mnt/ssd1/zining/cutex/.worktrees/cutex-pypi/modal/01_dense_gemm.py
git commit -m "feat: finish modal runner path for dense_gemm_7min"
```

### Task 3: Migrate `dense_gemm_7min.py` Imports To `cutex`

**Files:**
- Modify: `/mnt/ssd1/zining/cutex/.worktrees/cutex-pypi/modal/blackwell/dense_gemm_7min.py`

- [ ] **Step 1: Write the failing test**

Use the current top-of-file import block as the red-state target if it still exists:

```python
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline

from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
import cutlass.cute.testing as testing
```

The expected migrated import shape is:

```python
import cutlass
import cutex as cute
import cutex.utils as utils
import cutex.pipeline as pipeline

from cutex.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutex import cpasync, tcgen05, from_dlpack, testing
import cutex.utils.sm100 as sm100_utils
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
modal run 01_dense_gemm.py > >(tee out.log) 2> >(tee error.log >&2)
```

Expected: FAIL in `dense_gemm_7min.py` if the module still depends on the old `cutlass.cute`/`cutlass.pipeline`/`cutlass.utils.blackwell_helpers` import path that this verification is meant to replace.

- [ ] **Step 3: Write minimal implementation**

Replace only the import block in `dense_gemm_7min.py` with the `cutex`-based version below, leaving the kernel logic unchanged.

```python
import cutlass
import cutex as cute
import cutex.utils as utils
import cutex.pipeline as pipeline

from cutex.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutex import cpasync, tcgen05, from_dlpack, testing
import cutex.utils.sm100 as sm100_utils

import cuda.bindings.driver as cuda
```

If the module needs `runtime` or `nvgpu` module objects directly rather than just `from_dlpack`, `cpasync`, and `tcgen05`, import them from `cutex` rather than reopening `cutlass.cute.*` imports.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
modal run 01_dense_gemm.py > >(tee out.log) 2> >(tee error.log >&2)
```

Expected: `dense_gemm_7min.py` imports and runs through the `cutex` namespace without import errors. If there is a later runtime failure unrelated to imports, inspect `error.log` and continue only if the failure clearly belongs to a different bug.

- [ ] **Step 5: Commit**

```bash
git add /mnt/ssd1/zining/cutex/.worktrees/cutex-pypi/modal/blackwell/dense_gemm_7min.py
git commit -m "refactor: run dense_gemm_7min through cutex"
```

### Task 4: Full Modal Verification And Log Review

**Files:**
- Modify: `/mnt/ssd1/zining/cutex/.worktrees/cutex-pypi/modal/01_dense_gemm.py` only if verification reveals a minimal remaining runner issue
- Modify: `/mnt/ssd1/zining/cutex/.worktrees/cutex-pypi/modal/blackwell/dense_gemm_7min.py` only if verification reveals a minimal remaining import-path issue

- [ ] **Step 1: Write the failing test**

Treat the end-to-end Modal command as the final verification target:

```bash
modal run 01_dense_gemm.py > >(tee out.log) 2> >(tee error.log >&2)
```

- [ ] **Step 2: Run test to verify it fails**

Run exactly:

```bash
modal run 01_dense_gemm.py > >(tee out.log) 2> >(tee error.log >&2)
```

Expected: If any remaining issue exists, it will appear in `error.log` or the run output before final success is achieved.

- [ ] **Step 3: Write minimal implementation**

Apply only the smallest remaining fix supported by the logs. Allowed fixes at this stage:
- adjust the local repository mount path for `pip install /workspace/cutex_repo`
- adjust the `RUN_TESTS` selection or the `dense_gemm_7min` branch wiring in `01_dense_gemm.py`
- adjust `dense_gemm_7min.py` imports if one more `cutlass.*` import slipped through

Do not broaden scope beyond the runner and the target kernel file.

- [ ] **Step 4: Run test to verify it passes**

Run exactly:

```bash
modal run 01_dense_gemm.py > >(tee out.log) 2> >(tee error.log >&2)
```

Expected:
- Modal image build succeeds
- local `cutex` installation succeeds
- `dense_gemm_7min.py` imports from the mounted worktree source tree
- the selected benchmark run completes without import or runtime failure

After the run, inspect:

```bash
python - <<'PY'
from pathlib import Path

for name in ["out.log", "error.log"]:
    path = Path(name)
    print(f"=== {name} exists: {path.exists()} size={path.stat().st_size if path.exists() else 0} ===")
PY
```

Expected: both log files exist and can be inspected for the final verification evidence.

- [ ] **Step 5: Commit**

```bash
git add /mnt/ssd1/zining/cutex/.worktrees/cutex-pypi/modal/01_dense_gemm.py /mnt/ssd1/zining/cutex/.worktrees/cutex-pypi/modal/blackwell/dense_gemm_7min.py
git commit -m "test: verify dense_gemm_7min with cutex on modal"
```
