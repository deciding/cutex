# Cutez Autotune Design

## Goal

Add a reusable `cutez` autotuning flow for CuTe DSL kernels that provides a Triton-like user experience for `configs` and `key`, while adapting to CuTe's host-side `cute.compile(...)` model.

The new API should let kernel authors declare autotune metadata once and then obtain the best compiled kernel through `cutez.compile(kernel, ...)`.

## Scope

This design is limited to the first implementation of `cutez.autotune` and `cutez.compile`.

Included in scope:
- Add a `@cutez.autotune(...)` decorator for kernel `__call__` methods
- Add a CuTe-aware `cutez.compile(kernel, *args, **kwargs)` entrypoint
- Support tuning across multiple kernel-construction configs
- Compile and benchmark every candidate config on cache miss
- Cache the best config by a user-defined `key=[...]`
- Cache compiled candidates separately from autotune results
- Support explicit kernel reconstruction and explicit key-value resolution hooks

Out of scope for this change:
- Porting existing kernels to the new API
- Adding Triton-only config knobs such as `num_warps`, `num_ctas`, `num_stages`, or `maxnreg`
- Search-space pruning, perf models, or genetic search
- Persistent on-disk autotune result caching unless it falls out naturally from the implementation
- A combined convenience API that both tunes and launches the kernel in one call

## Context

Triton autotune decorates a device kernel and selects among launch/compiler configs. CuTe DSL differs in two important ways:

- The practical autotune unit is a host-side `cute.compile(...)` flow, not the device kernel body alone
- The relevant tuning parameters are kernel-construction parameters such as `mma_tiler_mn`, `cluster_shape_mn`, `use_2cta_instrs`, and `use_tma_store`

The guidance in `ref/guidance.md` already demonstrates the correct CuTe pattern:

1. build one kernel object per config
2. compile each candidate with `cute.compile(...)`
3. benchmark each compiled candidate
4. cache the winning result by an input-dependent key

This design generalizes that pattern into shared `cutez` infrastructure.

## Public API

### Decorator Placement

`@cutez.autotune(...)` should decorate the kernel class `__call__` method, which is also decorated with `@cute.jit`.

Example shape:

```python
class PersistentDenseGemmKernel:
    def __init__(
        self,
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        use_tma_store,
    ):
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.use_tma_store = use_tma_store

    def autotune_init_kwargs(self):
        return {
            "acc_dtype": self.acc_dtype,
            "use_2cta_instrs": self.use_2cta_instrs,
            "mma_tiler_mn": self.mma_tiler_mn,
            "cluster_shape_mn": self.cluster_shape_mn,
            "use_tma_store": self.use_tma_store,
        }

    def autotune_key_values(self, a, b, c, stream):
        m, k = a.shape
        n = b.shape[0]
        return {
            "m": m,
            "n": n,
            "k": k,
            "acc_dtype": self.acc_dtype,
            "a_dtype": getattr(a, "element_type", None),
            "b_dtype": getattr(b, "element_type", None),
            "c_dtype": getattr(c, "element_type", None),
        }

    @cutez.autotune(
        configs=[
            cutez.Config(kwargs={
                "mma_tiler_mn": (256, 256),
                "cluster_shape_mn": (2, 1),
                "use_2cta_instrs": True,
                "use_tma_store": True,
            }),
            cutez.Config(kwargs={
                "mma_tiler_mn": (128, 256),
                "cluster_shape_mn": (1, 1),
                "use_2cta_instrs": False,
                "use_tma_store": True,
            }),
        ],
        key=["m", "n", "k", "a_dtype", "b_dtype", "c_dtype", "acc_dtype"],
        warmup=5,
        rep=100,
        cache_results=True,
    )
    @cute.jit
    def __call__(self, a, b, c, stream):
        ...
```

### Runtime Entry Point

Kernel users call `cutez.compile(kernel, *args, **kwargs)` instead of raw `cute.compile(kernel, *args, **kwargs)` when they want autotune support.

Example shape:

```python
kernel = PersistentDenseGemmKernel(...)
compiled = cutez.compile(kernel, a, b, c, stream)
compiled(a, b, c, stream)
```

If the kernel does not carry autotune metadata, `cutez.compile(...)` should delegate directly to `cute.compile(...)`.

## Design

### `cutez.Config`

`cutez.Config` is a CuTe-specific configuration object. It represents kernel-construction overrides, not Triton compiler or launch knobs.

Expected first-pass shape:

```python
class Config:
    def __init__(self, kwargs: dict, name: str | None = None, pre_hook=None):
        self.kwargs = kwargs
        self.name = name
        self.pre_hook = pre_hook
```

Included in `kwargs`:
- `mma_tiler_mn`
- `cluster_shape_mn`
- `use_2cta_instrs`
- `use_tma_store`
- other kernel constructor overrides that future CuTe kernels may expose

Explicitly excluded from v1:
- `num_warps`
- `num_ctas`
- `num_stages`
- `maxnreg`

These do not map cleanly to the CuTe kernel-construction model described in the current guidance and examples.

### `@cutez.autotune(...)`

The decorator is declarative. It should attach autotune metadata to the decorated `__call__` entrypoint but should not itself perform compilation or benchmarking.

Recommended fields:
- `configs`: list of `cutez.Config`
- `key`: list of names that define when to retune
- `warmup`: benchmark warmup count
- `rep`: benchmark repetition count
- `cache_results`: whether autotune results are cached in memory and optionally later on disk
- `do_bench`: optional custom benchmark function

The decorator should store metadata in a well-defined attribute on the wrapped callable so `cutez.compile(...)` can discover it reliably.

### `cutez.compile(...)`

`cutez.compile(kernel, *args, **kwargs)` is the autotune engine.

When autotune metadata exists, it should:

1. inspect `kernel.__call__` and load autotune metadata
2. resolve named values for the requested `key=[...]`
3. build the autotune cache key
4. return the previously selected compiled kernel on cache hit
5. otherwise rebuild one kernel instance per config
6. compile each candidate with `cute.compile(candidate_kernel, *args, **kwargs)`
7. benchmark each compiled candidate
8. select the best candidate
9. cache the winning config and best compiled kernel
10. return the best compiled kernel

When autotune metadata does not exist, it should:

1. delegate directly to `cute.compile(kernel, *args, **kwargs)`

### Kernel Reconstruction Contract

Because config values affect kernel initialization, `cutez.compile(...)` must create one kernel instance per config.

The design therefore requires autotuned kernels to expose an explicit reconstruction hook:

```python
def autotune_init_kwargs(self) -> dict:
    ...
```

Candidate reconstruction should follow this pattern:

```python
base_kwargs = kernel.autotune_init_kwargs()
candidate_kwargs = {**base_kwargs, **config.kwargs}
candidate_kernel = type(kernel)(**candidate_kwargs)
```

This explicit contract is preferred over fragile implicit attribute introspection.

If an autotuned kernel does not implement `autotune_init_kwargs()`, `cutez.compile(...)` should raise a clear error.

### Key Resolution for `key=[...]`

The public API should use Triton-like `key=[...]`, but `cutez.compile(...)` must define how those names are resolved.

Resolution order:

1. derive built-in default values from compile args where possible
2. if the kernel defines `autotune_key_values(*args, **kwargs)`, merge its returned mapping
3. kernel-provided values override built-in defaults

Expected first-pass explicit hook:

```python
def autotune_key_values(self, *args, **kwargs) -> dict:
    ...
```

This hook returns the named values from which the final autotune key is assembled.

Example:

```python
key_dict = resolve_key_values(kernel, *args, **kwargs)
tuning_key = tuple(key_dict[name] for name in meta.key)
```

If any requested key name cannot be resolved, `cutez.compile(...)` should raise a clear error naming the missing field.

### Compile Signature vs Autotune Key

The design should keep compile caching and autotune-result caching separate.

Autotune key:
- defined by user-facing `key=[...]`
- identifies when the best config decision can be reused

Compile signature:
- derived internally from compile-sensitive information
- identifies when a compiled artifact can be reused

The compile signature should include:
- kernel class identity
- candidate reconstruction kwargs
- tensor dtype and layout information needed by `cute.compile(...)`
- shape information when compile output depends on shape

This separation prevents unnecessary recompilation and unnecessary rebenchmarking.

## Caching

### Compile Cache

Maps:

`(kernel_class, candidate_init_signature, compile_signature)` -> compiled kernel

Purpose:
- avoid recompiling the same candidate kernel variant

### Autotune Cache

Maps:

`(kernel_class, tuning_key)` -> winning config identity

Purpose:
- avoid rebenchmarking on future calls with matching keys

### Optional Fast-Path Cache

Maps:

`(kernel_class, tuning_key)` -> best compiled kernel

Purpose:
- return the compiled winner directly without an extra config lookup

This can be added in the first implementation if it keeps the code simpler, but the conceptual model should remain split between compile caching and autotune-result caching.

## Benchmarking

The default benchmark should follow the guidance in `ref/guidance.md`:

- run warmups
- run multiple timed iterations
- synchronize correctly
- return a stable timing metric

Recommended default metric for v1:
- median execution time

The benchmark target is the compiled candidate returned by `cute.compile(...)`, invoked with the original runtime launch arguments.

The decorator should also accept an optional `do_bench` override for kernels that need custom setup.

## Error Handling

Candidate-level failures should not abort autotuning unless every config fails.

For each config, treat the following as a failed candidate:
- kernel reconstruction failure
- `cute.compile(...)` failure
- benchmark launch failure

If all candidates fail, raise a dedicated autotune error containing per-config failure summaries.

Also raise clear errors for:
- missing `autotune_init_kwargs()` on autotuned kernels
- missing required names in `key=[...]`
- empty config lists when autotune is explicitly requested

## Testing

The first implementation should verify:

- undecorated kernels still pass through `cutez.compile(...)` to raw `cute.compile(...)`
- autotune metadata is discoverable from decorated `__call__`
- one candidate kernel instance is rebuilt per config
- all configs are compiled and benchmarked on cache miss
- best config is cached by the resolved tuning key
- subsequent calls with the same tuning key skip benchmarking
- compile cache avoids recompiling the same candidate variant
- missing reconstruction hook and missing key names raise clear errors

Tests can use a fake benchmark function and a lightweight compile stub where direct GPU execution would make the test too heavy.

## Implementation Notes

- Keep v1 focused on exhaustive search across declared configs
- Prefer a small number of explicit hooks over broad magic introspection
- Keep `cutez.compile(...)` as a thin compatibility wrapper when autotune metadata is absent
- Avoid imitating Triton fields that do not match CuTe semantics
- Preserve the existing CuTe usage pattern where users still work with kernel objects and compiled callables

## Future Work

Possible follow-up improvements after v1:

- persistent disk cache for autotune results
- config pruning and performance-model filtering
- convenience wrappers for tune-and-launch flows
- helper utilities for common GEMM key derivation
- migration of `modal/blackwell/dense_gemm_7min.py` and related examples to the shared autotune API
