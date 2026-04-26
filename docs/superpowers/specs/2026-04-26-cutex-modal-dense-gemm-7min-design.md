# Cutex Modal Dense GEMM 7min Design

## Goal

Update the worktree copy of `modal/01_dense_gemm.py` so `modal run 01_dense_gemm.py` installs the local `cutex` package and runs `dense_gemm_7min.py` through the Modal benchmark flow using `cutex` instead of the original CuTe import path.

## Scope

This design is limited to verifying the new `dense_gemm_7min.py` flow inside the existing Modal runner.

Included in scope:
- Target the worktree at `/mnt/ssd1/zining/cutex/.worktrees/cutex-pypi`
- Update the worktree copy of `modal/01_dense_gemm.py`
- Ensure the Modal image installs the local `cutex` package from the worktree repository
- Ensure the Modal image mounts the `blackwell` sources that include `dense_gemm_7min.py`
- Add a new run path so `01_dense_gemm.py` can execute `dense_gemm_7min.py`
- Verify the flow using `modal run 01_dense_gemm.py`

Out of scope for this change:
- Merging the worktree changes back to the main workspace
- Reworking the older dense GEMM runners
- Changing the package surface of `cutex`
- Doing a broader migration of all Blackwell kernels to `cutex`

## Execution Target

The implementation should modify and run the worktree copy of the Modal script, not the main workspace copy.

Primary target files:
- `/mnt/ssd1/zining/cutex/.worktrees/cutex-pypi/modal/01_dense_gemm.py`
- `/mnt/ssd1/zining/cutex/.worktrees/cutex-pypi/modal/blackwell/dense_gemm_7min.py` only if needed to swap imports to `cutex`

## Design

### Modal Image Setup

The existing Modal image in `01_dense_gemm.py` currently installs `nvidia-cutlass-dsl` and mounts the local `blackwell` directory.

This change should extend that setup so the image also installs the local `cutex` package from the worktree repository.

Recommended shape:
- Mount the relevant local repository path into the container
- Run `pip install` against that mounted local package path inside the Modal image build

This keeps verification tied to the exact local package contents rather than a previously published PyPI package.

### Blackwell Source Mount

`dense_gemm_7min.py` currently exists only in the worktree branch. The mounted `blackwell` directory in the Modal script must therefore come from the worktree copy so the new kernel file is available in the container.

### Benchmark Selection

`01_dense_gemm.py` uses a `RUN_TESTS` list to select which kernels to execute.

Add a new entry for `dense_gemm_7min` and a corresponding execution branch in `run_dense_gemm()`.

The execution style should follow the existing direct-import pattern already used for the other `dense_gemm_*` scripts whenever possible.

### Import Strategy

The intended verification is that `dense_gemm_7min.py` runs using `cutex` rather than the original CuTe import path.

If `dense_gemm_7min.py` still imports `cutlass.cute`, `cutlass.pipeline`, or `cutlass.utils.blackwell_helpers` directly, update only the imports needed for this script to use the `cutex` package surface.

The migration should stay minimal and focused on this kernel file.

## Verification

Verification should use Modal exactly through the runner script:

```bash
modal run 01_dense_gemm.py > >(tee out.log) 2> >(tee error.log >&2)
```

Success criteria for this first pass:
- The Modal image build succeeds
- Local `cutex` installs successfully in the Modal image
- `dense_gemm_7min.py` is importable from the mounted `blackwell` tree
- The `dense_gemm_7min` run path executes without import or runtime failure

If the Modal run fails, inspect `error.log` first and use the smallest follow-up fix needed.

## Implementation Notes

- Prefer matching the current structure of `01_dense_gemm.py`
- Avoid restructuring unrelated benchmark branches
- Keep the change limited to the runner and the target kernel unless a direct dependency forces one more file change
- Reuse the existing `RUN_TESTS` selection model instead of inventing a new CLI or workflow

## Future Work

If this verification succeeds, later work can migrate more Blackwell kernels and the broader Modal benchmark matrix to `cutex` in the same style.
