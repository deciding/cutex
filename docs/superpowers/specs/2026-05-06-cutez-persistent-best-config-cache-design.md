# Cutez Persistent Best Config Cache Design

## Goal

Add an optional persistent best-config cache for `cutez` autotuning so the winning config for a resolved tuning key can survive process boundaries and be reused on later runs.

The persistence layer should be much smaller than Triton's cache system and should only save the selected best config, not compiled artifacts or full timing histories.

## Scope

Included in scope:
- Add optional persistent caching for `tuning_key -> best_config`
- Make persistence opt-in through `@cutez.autotune(...)`
- Have `cutez.compile(...)` automatically detect and use the configured persistence path from the autotune spec
- Store cache entries in a human-readable JSON file at an explicit user-provided path
- Seed the existing in-memory best-config cache from disk when a matching entry exists

Out of scope:
- Persisting compiled callables or compile-cache entries
- Persisting full candidate timing tables
- Adding implicit default cache locations
- Adding a global cache manager abstraction
- Environment-variable-driven cache locations

## Context

Current `cutez` behavior keeps autotune state only in memory:

- `_BEST_CONFIG_CACHE` stores the chosen best config for the current Python process
- `_COMPILE_CACHE` stores compiled candidate artifacts only for the current Python process

This means the current best-config result is lost on a fresh Python process or a fresh Modal container run.

The Triton reference in `ref/autotuner.py` does more:

- it keeps an in-memory best-config cache
- it can optionally persist autotune results to disk through `check_disk_cache(...)`

However, Triton persists timing data and uses a broader cache-manager system. This design intentionally keeps `cutez` much smaller and only persists the chosen best config.

## Public API

### `@cutez.autotune(...)`

Extend the autotune decorator with an optional `cache_path` field.

Example:

```python
@cutez.autotune(
    configs=[
        cutez.Config(kwargs={"mma_tiler_mn": (256, 256), "cluster_shape_mn": (2, 1), "ab_stages": 7}),
    ],
    key=["m", "n", "k"],
    cache_results=True,
    cache_path="/tmp/cutez_dense_gemm_autotune.json",
)
def host_function(a, b, c, stream, mma_tiler_mn, cluster_shape_mn, ab_stages):
    return launch_dense_gemm(a, b, c, stream, mma_tiler_mn, cluster_shape_mn, ab_stages)
```

If `cache_path` is omitted or `None`:
- behavior remains memory-only

If `cache_path` is provided:
- `cutez.compile(...)` should automatically consult and update that file through the autotune spec

The caller should not need to pass the path again to `cutez.compile(...)`.

## Design

### What Gets Persisted

Persist only:
- stable kernel identifier
- resolved tuning key
- best config

Do not persist:
- compiled artifacts
- compile cache
- candidate timings
- failure histories

This means the disk cache is just a persisted version of `_BEST_CONFIG_CACHE`, not a persisted version of the whole autotune runtime state.

### Stable Kernel Identity

The in-memory cache currently uses Python identity through `_kernel_identity(kernel)`, which is not serializable across runs.

For disk persistence, derive a stable string identifier:

- plain function: `module.qualname`
- callable object: `module.ClassName.__call__`

Examples:

- `modal.blackwell.dense_gemm_7min.host_function`
- `my_package.kernel.PersistentDenseGemmKernel.__call__`

This identifier should be used only for disk-cache lookup and serialization. The in-memory cache can keep using the current Python identity-based keying.

### Disk Cache Key

Use a human-readable serialized key structure with:

- `kernel`: stable kernel identifier string
- `key`: serialized list form of the resolved tuning key tuple

Example serialized entry:

```json
{
  "kernel": "modal.blackwell.dense_gemm_7min.host_function",
  "key": [8192, 8192, 4096],
  "config": {
    "kwargs": {
      "mma_tiler_mn": [256, 256],
      "cluster_shape_mn": [2, 1],
      "ab_stages": 7
    },
    "name": null
  }
}
```

The persisted file should be easy to inspect and edit manually when debugging.

### File Format

Store a JSON object with an `entries` list.

Recommended first-pass shape:

```json
{
  "entries": [
    {
      "kernel": "modal.blackwell.dense_gemm_7min.host_function",
      "key": [8192, 8192, 4096],
      "config": {
        "kwargs": {
          "mma_tiler_mn": [256, 256],
          "cluster_shape_mn": [2, 1],
          "ab_stages": 7
        },
        "name": null
      }
    }
  ]
}
```

This is intentionally simpler than Triton's cache-manager JSON format.

### Runtime Flow In `cutez.compile(...)`

When `cache_path` is configured in the autotune spec:

1. resolve the in-memory tuning key as today
2. check `_BEST_CONFIG_CACHE`
3. if in-memory miss:
   - load the JSON file if it exists
   - derive the stable serialized disk key
   - search for a matching entry
   - if found, reconstruct `Config` from the stored data
   - seed `_BEST_CONFIG_CACHE`
4. continue normal compile flow
5. when benchmarking selects a winning config:
   - update `_BEST_CONFIG_CACHE`
   - update or append the matching JSON entry on disk

This means disk persistence is a backing store for the existing in-memory best-config cache, not a separate caching layer.

### Config Reconstruction From Disk

The persisted file should only reconstruct fields that are safely serializable and already part of the public `Config` surface:

- `kwargs`
- `name`

Do not attempt to serialize or reconstruct:
- `pre_hook`
- arbitrary callable objects

For the first pass, reconstructed persisted configs should always come back as:

```python
Config(kwargs=<loaded kwargs>, name=<loaded name>, pre_hook=None)
```

### When To Skip Persistence

Persistence should be skipped automatically when it is not safe or not meaningful.

Skip reading/writing the disk cache when:
- `cache_path` is not provided
- any configured `Config` has `pre_hook`
- the winning config cannot be serialized cleanly
- the kernel stable identifier cannot be derived
- the JSON file cannot be parsed
- file I/O fails

In these cases, `cutez` should fall back to the current memory-only behavior.

When `verbose=True`, `cutez.compile(...)` should print a concise message when persistence is skipped or when a disk cache entry is loaded.

### File I/O Behavior

The persistence path is explicit, so `cutez` may create parent directories as needed.

Recommended first-pass behavior:
- if parent directories do not exist, create them
- if the file does not exist, treat it as empty cache
- if the file is malformed JSON, ignore it and continue memory-only for that call

The first version should favor robustness over strict failure.

## Error Handling

Persistence failures should not fail kernel compilation or autotuning.

Examples:
- malformed JSON
- permission denied
- serialization failure for a config field

These should degrade to memory-only behavior.

The autotune runtime should still fail normally only when actual candidate compilation/benchmark selection fails.

## Testing

The first implementation should verify:

- memory-only behavior remains unchanged when `cache_path` is absent
- a winning best config is written to the JSON file when `cache_path` is present
- a second fresh process-style load can reconstruct and reuse the best config from disk
- `cutez.compile(..., verbose=True)` reports disk cache hits distinctly from in-memory cache hits
- disk persistence is skipped when any config has `pre_hook`
- malformed JSON degrades safely to memory-only behavior
- persisted `Config` reconstruction preserves `kwargs` and `name`

Tests can simulate a fresh process by clearing `_BEST_CONFIG_CACHE` between calls while keeping the JSON file on disk.

## Implementation Notes

- Keep the first version best-config-only
- Do not change `_COMPILE_CACHE` persistence behavior
- Do not add environment-controlled paths or hashed global cache directories
- Keep the JSON file human-readable
- Keep the persistence path in the autotune spec, not in the compile call

## Future Work

Possible follow-ups after this change:

- persist full candidate timing history similar to Triton
- add cache-version metadata and invalidation rules tied to backend/runtime versions
- add a helper API to inspect or clear persisted cache files
- add optional global cache-root conventions if explicit-path-only proves too limiting
