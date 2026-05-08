# Dense GEMM 7min Host Function Autotune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate `cutez.autotune` and `cutez.compile` into `modal/blackwell/dense_gemm_7min.py` so the existing `modal run modal/01_dense_gemm.py` path autotunes `mma_tiler_mn` and `cluster_shape_mn` through the host-side `host_function` flow.

**Architecture:** Make the smallest `cutez` runtime change needed to support a decorated plain host function as the autotune target, then update `dense_gemm_7min.py` to expose `mma_tiler_mn` and `cluster_shape_mn` in `host_function`, decorate it with a local config list, and call `cutez.compile(host_function, ...)` instead of raw `cute.compile(...)`. Leave `modal/01_dense_gemm.py` unchanged unless verification proves the current local package setup is insufficient.

**Tech Stack:** Python 3.13, pytest, Modal, CUTLASS CuTe DSL, local `cutez` package, `modal/blackwell/dense_gemm_7min.py`

---

## File Structure

- Modify: `cutez/compiler.py`
  Responsibility: allow `cutez.compile(...)` to autotune decorated plain functions in addition to callable objects, with the smallest behavior change needed for this integration.
- Modify: `cutez/_autotune_keys.py`
  Responsibility: if needed, support plain-function key extraction or argument-name binding for `key=["m", "n", "k"]` on host-side compile functions.
- Modify: `tests/test_cutez_autotune.py`
  Responsibility: add focused regression coverage for plain-function autotune compilation and key resolution.
- Modify: `modal/blackwell/dense_gemm_7min.py`
  Responsibility: expose `mma_tiler_mn` and `cluster_shape_mn` in `host_function`, add `@cutez.autotune(...)`, replace `cute.compile(...)` with `cutez.compile(...)`, and keep the rest of the benchmark flow stable.
- Modify: `modal/01_dense_gemm.py` only if verification proves the current local package mounting path does not work.

## Task 1: Support Plain Host Functions In `cutez.compile(...)`

**Files:**
- Modify: `cutez/compiler.py`
- Modify: `cutez/_autotune_keys.py`
- Modify: `tests/test_cutez_autotune.py`

- [ ] **Step 1: Write the failing plain-host-function autotune test**

Append this test to `tests/test_cutez_autotune.py`:

```python
def test_compile_autotunes_decorated_plain_host_function(cutez_module, monkeypatch):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")
    compile_calls = []
    bench_calls = []

    def fake_read_autotune_spec(kernel):
        return autotune_module.get_autotune_spec(kernel)

    def fake_compile(fn, *args, **kwargs):
        compile_calls.append((kwargs["mma_tiler_mn"], kwargs["cluster_shape_mn"], args))
        return f"compiled:{kwargs['mma_tiler_mn']}:{kwargs['cluster_shape_mn']}"

    def fake_benchmark(compiled, *args, **kwargs):
        bench_calls.append(compiled)
        return {
            "compiled:(128, 256):(1, 1)": 2.0,
            "compiled:(256, 256):(2, 1)": 1.0,
        }[compiled]

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )
    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(compiler_module, "benchmark", fake_benchmark, raising=False)

    @cutez_module.autotune(
        configs=[
            cutez_module.Config(kwargs={"mma_tiler_mn": (128, 256), "cluster_shape_mn": (1, 1)}),
            cutez_module.Config(kwargs={"mma_tiler_mn": (256, 256), "cluster_shape_mn": (2, 1)}),
        ],
        key=["m", "n", "k"],
    )
    def host_function(a, b, c, max_active_clusters, stream, m, n, k, mma_tiler_mn, cluster_shape_mn):
        return (a, b, c, max_active_clusters, stream, m, n, k, mma_tiler_mn, cluster_shape_mn)

    result = cutez_module.compile(
        host_function,
        "a",
        "b",
        "c",
        4,
        "stream0",
        m=8192,
        n=8192,
        k=4096,
    )

    assert result == "compiled:(256, 256):(2, 1)"
    assert [call[0] for call in compile_calls] == [(128, 256), (256, 256)]
    assert [call[1] for call in compile_calls] == [(1, 1), (2, 1)]
    assert bench_calls == [
        "compiled:(128, 256):(1, 1)",
        "compiled:(256, 256):(2, 1)",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cutez_autotune.py::test_compile_autotunes_decorated_plain_host_function -v`
Expected: FAIL because `cutez.compile(...)` currently does not run the autotune loop for decorated plain functions.

- [ ] **Step 3: Implement minimal plain-function autotune support**

Update `cutez/compiler.py` so the compile path can autotune decorated plain functions by using the decorated function itself as the compile target when `inspect.isfunction(kernel)` is true.

Expected implementation shape:

```python
import inspect


def _is_plain_function(kernel):
    return inspect.isfunction(kernel)


def _base_candidate_kwargs(kernel, runtime_key_values):
    if _is_plain_function(kernel):
        return {
            name: value
            for name, value in runtime_key_values.items()
            if name not in {"stream", "max_active_clusters"}
        }
    return dict(kernel.autotune_init_kwargs())


def _candidate_target(kernel, candidate_kwargs):
    if _is_plain_function(kernel):
        return kernel
    return type(kernel)(**candidate_kwargs)
```

Then update the compile loop so plain functions:
- do not require `autotune_init_kwargs()`
- still build candidate kwargs from resolved key values plus config kwargs
- call `cute.compile(host_function, *args, **candidate_kwargs, **kwargs)` for each config

Update `cutez/_autotune_keys.py` if needed so `m`, `n`, and `k` passed as keyword arguments are available in resolved key values for plain host functions.

- [ ] **Step 4: Run the new plain-function test to verify it passes**

Run: `pytest tests/test_cutez_autotune.py::test_compile_autotunes_decorated_plain_host_function -v`
Expected: PASS

- [ ] **Step 5: Run the focused autotune suite to catch regressions**

Run: `pytest tests/test_cutez_autotune.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add cutez/compiler.py cutez/_autotune_keys.py tests/test_cutez_autotune.py
git commit -m "support host function autotune"
```

## Task 2: Autotune `dense_gemm_7min.py` Through `host_function`

**Files:**
- Modify: `modal/blackwell/dense_gemm_7min.py`

- [ ] **Step 1: Write the failing local static assertions**

Add these assertions temporarily in a small local verification block near the compile-site refactor work while developing:

```python
assert callable(host_function)
assert mma_tiler_mnk[:2] == mma_tiler_mn
assert len(cluster_shape_mn) == 2
```

These are not permanent tests; they are guardrails while threading the new parameters through the host-function path.

- [ ] **Step 2: Expose the tuned values in `host_function`**

Change the `host_function` signature from:

```python
@cute.jit
def host_function(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
):
```

to:

```python
@cutez.autotune(
    configs=[
        cutez.Config(kwargs={"mma_tiler_mn": (128, 256), "cluster_shape_mn": (1, 1)}),
        cutez.Config(kwargs={"mma_tiler_mn": (256, 256), "cluster_shape_mn": (2, 1)}),
    ],
    key=["m", "n", "k"],
)
@cute.jit
def host_function(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
    m: cutlass.Constexpr,
    n: cutlass.Constexpr,
    k: cutlass.Constexpr,
    mma_tiler_mn: cutlass.Constexpr,
    cluster_shape_mn: cutlass.Constexpr,
):
```

Keep `use_2cta_instrs` and `use_tma_store` fixed in outer scope.

- [ ] **Step 3: Thread the two tuned values through the existing kernel construction path**

Inside `host_function`, replace the fixed outer-scope use sites with the new explicit parameters.

Specifically update:

```python
tiled_mma = sm100_utils.make_trivial_tiled_mma(
    io_dtype,
    tcgen05.OperandMajorMode.K,
    tcgen05.OperandMajorMode.K,
    acc_dtype,
    tcgen05.CtaGroup.TWO,
    mma_tiler_mn,
)
```

and keep:

```python
mma_tiler_mnk = (*mma_tiler_mn, mma_tiler_k)
cluster_shape_mnl = (*cluster_shape_mn, 1)
```

or the equivalent local reconstruction already used by the file.

Do not refactor unrelated host-function logic.

- [ ] **Step 4: Replace the direct compile helper with `cutez.compile(...)`**

Remove or collapse the existing cached helper:

```python
@lru_cache(maxsize=1)
def compile_mm(...):
    ...
```

Then update the compile call site to:

```python
compiled_gemm = cutez.compile(
    host_function,
    a_tensor,
    b_tensor,
    c_tensor,
    max_active_clusters,
    current_stream,
    m=m,
    n=n,
    k=k,
)
```

This ensures the autotune key fields are available explicitly and the compile path uses the shared autotune runtime.

- [ ] **Step 5: Keep the correctness and outer benchmark flow unchanged**

Preserve these sections except for the new `compiled_gemm` source:

```python
if not skip_ref_check:
    compiled_gemm(a_tensor, b_tensor, c_tensor, current_stream)
    compare(...)

exec_time = testing.benchmark(
    compiled_gemm,
    workspace_generator=generate_tensors,
    workspace_count=workspace_count,
    stream=current_stream,
    warmup_iterations=warmup_iterations,
    iterations=iterations,
)
```

Do not change the correctness-check semantics or the benchmark-reporting flow.

- [ ] **Step 6: Run a local static syntax/import check**

Run: `python -m compileall modal/blackwell/dense_gemm_7min.py cutez`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add modal/blackwell/dense_gemm_7min.py
git commit -m "autotune dense_gemm_7min host function"
```

## Task 3: Verify Modal End-To-End Path

**Files:**
- Modify: `modal/01_dense_gemm.py` only if required by failed verification
- Modify: `modal/blackwell/dense_gemm_7min.py` if verification reveals an integration bug

- [ ] **Step 1: Run the end-to-end Modal command**

Run: `modal run modal/01_dense_gemm.py`
Expected: the image builds or reuses successfully, `dense_gemm_7min` runs, `cutez.compile(...)` is exercised, and the run reaches correctness/benchmark output without crashing.

- [ ] **Step 2: If the Modal run fails due to local `cutez` packaging only, make the smallest runner fix**

If the current `.add_local_dir(root_dir.parent / "cutez", remote_path="/workspace/cutez", copy=True)` path is insufficient, change only what is needed to make imports work.

Allowed small fix shape:

```python
.add_local_dir(root_dir.parent, remote_path="/workspace/repo", copy=True)
```

paired with:

```python
import sys
sys.path.insert(0, "/workspace/repo")
```

Do not make this change unless the Modal run proves it is necessary.

- [ ] **Step 3: Re-run the same Modal command if a runner fix was required**

Run: `modal run modal/01_dense_gemm.py`
Expected: PASS on the rerun

- [ ] **Step 4: Re-run the focused local autotune suite after any verification-driven code fix**

Run: `pytest tests/test_cutez_autotune.py -v`
Expected: PASS

- [ ] **Step 5: Commit only if code changed during verification**

```bash
git add modal/01_dense_gemm.py modal/blackwell/dense_gemm_7min.py tests/test_cutez_autotune.py
git commit -m "fix modal dense_gemm autotune integration"
```

If no code changed during verification, skip this commit step.

## Final Verification

- [ ] Run: `pytest tests/test_cutez_autotune.py -v`
Expected: PASS

- [ ] Run: `python -m compileall cutez modal/blackwell/dense_gemm_7min.py`
Expected: PASS

- [ ] Run: `modal run modal/01_dense_gemm.py`
Expected: PASS for the `dense_gemm_7min` path

- [ ] Run: `git status --short`
Expected: clean working tree or only unrelated pre-existing files
