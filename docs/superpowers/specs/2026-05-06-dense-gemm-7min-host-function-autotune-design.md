# Dense GEMM 7min Host Function Autotune Design

## Goal

Integrate the new `cutez.autotune` and `cutez.compile` flow into `modal/blackwell/dense_gemm_7min.py` so that `modal run modal/01_dense_gemm.py` exercises autotuned compilation for the `dense_gemm_7min` benchmark path.

The integration should be minimal and should preserve the existing script structure as much as possible.

## Scope

Included in scope:
- Update `modal/blackwell/dense_gemm_7min.py` to use `@cutez.autotune(...)`
- Replace the current direct `cute.compile(...)` path with `cutez.compile(...)`
- Expose only `mma_tiler_mn` and `cluster_shape_mn` as autotuned config values
- Keep `modal/01_dense_gemm.py` unchanged unless verification shows that the existing local `cutez` packaging path does not work
- Verify the integration by running `modal run modal/01_dense_gemm.py`

Out of scope:
- Refactoring `dense_gemm_7min.py` into a class-based kernel wrapper
- Tuning `use_2cta_instrs` or `use_tma_store`
- Adding new Modal flags, benchmark modes, or CLI surface
- General cleanup of unrelated dense GEMM scripts

## Context

`modal/01_dense_gemm.py` already routes the `dense_gemm_7min` benchmark through the existing Modal image setup and already includes the local `cutez` package directory in the container image.

`modal/blackwell/dense_gemm_7min.py` currently uses a host-side `host_function` compiled through a small fixed helper:

```python
@lru_cache(maxsize=1)
def compile_mm(a, b, c, max_active_clusters):
    return cute.compile(host_function, a, b, c, max_active_clusters, stream)
```

That helper only caches a single fixed compile result and does not express a tuning space. The file already has meaningful compile-time kernel choices in `mma_tiler_mn` and `cluster_shape_mn`, which makes it a good first integration target for the shared `cutez` autotune flow.

The user explicitly wants to keep the file host-function-centered rather than introducing a new class wrapper, so this design uses `@cutez.autotune(...)` directly on `host_function`.

## Public Behavior

After this change:

- `dense_gemm_7min.py` should compile its kernel through `cutez.compile(host_function, ...)`
- `host_function` should carry autotune metadata via `@cutez.autotune(...)`
- `cutez` should benchmark candidate configs and cache the best result by problem key
- `modal run modal/01_dense_gemm.py` should still run the same benchmark path, but now the `dense_gemm_7min` case should use the shared autotune runtime

No new user-facing runner options should be added.

## Design

### Keep `host_function` As The Autotune Target

`dense_gemm_7min.py` should stay structurally close to its current form.

Do not introduce a new kernel class or adapter object. Instead:

1. decorate `host_function` with `@cutez.autotune(...)`
2. make the tunable values explicit in the `host_function` signature
3. replace `cute.compile(host_function, ...)` with `cutez.compile(host_function, ...)`

This keeps the integration aligned with how the file is already structured.

### Tuned Parameters

Only these two parameters should be autotuned:

- `mma_tiler_mn`
- `cluster_shape_mn`

These values should move from outer-scope fixed inputs into explicit `host_function` parameters so the autotuner can vary them per config.

The following should remain fixed exactly as they are today:

- `use_2cta_instrs`
- `use_tma_store`
- dtypes
- correctness tolerance
- benchmark iteration counts
- tensor initialization mode

### `host_function` Signature Change

The minimal required integration change is to make the two tuned parameters explicit in the host-side compile function:

```python
@cutez.autotune(...)
@cute.jit
def host_function(
    a,
    b,
    c,
    max_active_clusters,
    stream,
    mma_tiler_mn,
    cluster_shape_mn,
):
    return kernel_entrypoint(
        a,
        b,
        c,
        max_active_clusters,
        stream,
        mma_tiler_mn,
        cluster_shape_mn,
    )
```

If lower-level helper or kernel-construction code currently reads those values from closure scope, `host_function` should thread them through directly. No broader refactor is needed.

The important point is that the autotuner can only vary values that appear in the decorated function's compile interface.

### Autotune Configs

`host_function` should be decorated with a static list of `cutez.Config(...)` values that vary only the two selected knobs.

Example shape:

```python
@cutez.autotune(
    configs=[
        cutez.Config(kwargs={"mma_tiler_mn": (256, 256), "cluster_shape_mn": (2, 1)}),
        cutez.Config(kwargs={"mma_tiler_mn": (128, 256), "cluster_shape_mn": (1, 1)}),
    ],
    key=["m", "n", "k"],
)
```

The file should continue to define its search space locally. The first pass does not need any dynamic config generation.

### Autotune Key

The best config should be cached by problem shape, not by config choices.

Recommended first-pass key:

- `m`
- `n`
- `k`

Reasoning:

- the benchmark path appears fixed to a single dtype combination in the current Modal flow
- the key should identify when to retune, not what is being tuned
- `mma_tiler_mn` and `cluster_shape_mn` are candidate choices and therefore should not appear in `key=[...]`

If implementation reveals that dtype or layout materially participates in compilation for this script path, that can be added later, but the initial integration should stay minimal.

### Compile Call Site

Replace the current cached helper flow:

```python
compiled_gemm = compile_mm(a_tensor, b_tensor, c_tensor, max_active_clusters)
```

with a direct shared-runtime compile path based on the decorated host function:

```python
compiled_gemm = cutez.compile(
    host_function,
    a_tensor,
    b_tensor,
    c_tensor,
    max_active_clusters,
    current_stream,
)
```

The current one-entry `lru_cache` helper should be removed or collapsed if it no longer serves a purpose after `cutez.compile(...)` takes over caching.

### Benchmarking And Correctness Flow

Keep the rest of `dense_gemm_7min.py` behavior as close to current as possible.

Preserve:

- tensor creation logic
- correctness checking against the reference path
- outer `testing.benchmark(...)` timing/reporting flow

This means there are two distinct timing layers:

1. internal autotune benchmarking used by `cutez.compile(...)` to choose a config
2. the existing benchmark/reporting flow already used by the script after the best compiled kernel is selected

This is acceptable for the first integration because it minimizes disruption to the existing benchmark harness.

## Files To Change

Primary file:
- `modal/blackwell/dense_gemm_7min.py`

Do not change unless required by verification:
- `modal/01_dense_gemm.py`

The current Modal runner should remain unchanged unless the local `cutez` package import path fails in practice.

## Error Handling

The integration should rely on the existing `cutez` failure behavior.

If all autotune configs fail, the script should surface the `cutez` autotune error rather than silently falling back to the old fixed compile path.

This is important so that incorrect autotune integration fails loudly during Modal verification.

## Testing

Verification should happen in two layers.

### Local Static Verification

Before running Modal:

- ensure `dense_gemm_7min.py` imports `cutez`
- ensure the file still compiles as Python
- add or update any light local checks that are practical without requiring GPU execution

### Modal End-to-End Verification

Primary acceptance command:

```bash
modal run modal/01_dense_gemm.py
```

Success criteria:

- the Modal image starts successfully
- `dense_gemm_7min.py` imports and executes
- `cutez.compile(...)` is exercised for the `dense_gemm_7min` test path
- the run reaches correctness and benchmark output without crashing

## Implementation Notes

- Prefer the smallest correct change in `dense_gemm_7min.py`
- Do not convert the file to a class-based kernel API for this integration
- Keep the autotune config list close to the existing tuning parameter definitions so the benchmark remains easy to inspect
- If `cutez.compile(...)` currently expects metadata from a callable object in a way that blocks decorated host functions, fix that in `cutez` only as much as needed to support the direct `host_function` usage described here

## Future Work

Possible follow-ups after this integration:

- expand the tuning space to include `use_2cta_instrs` or `use_tma_store`
- migrate other dense GEMM scripts to the same shared autotune path
- refine compile-cache signatures for better reuse across equivalent tensor-like inputs
- reduce duplicate timing work between autotune selection and outer benchmark reporting if it becomes a meaningful overhead
