# Cutez Autotune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `cutez.autotune` and `cutez.compile` so CuTe kernel classes can declare autotune metadata on `__call__`, and `cutez.compile(kernel, ...)` can reconstruct candidates, compile and benchmark all configs, cache the winner by key, and return the best compiled kernel.

**Architecture:** Keep the current `cutez` package surface lightweight by moving the new autotune runtime into small focused modules under `cutez/`, then re-export the public API from `cutez/__init__.py`. Use exhaustive search in v1 with two separate caches: a compile cache for candidate artifacts and an autotune cache for best-config decisions.

**Tech Stack:** Python 3.13, setuptools packaging, pytest, monkeypatch-based unit tests, CuTe DSL via `cutlass.cute` and `cute.compile`

---

## File Structure

- Modify: `cutez/__init__.py`
  Responsibility: keep the existing public package exports and re-export the new autotune API.
- Create: `cutez/autotune.py`
  Responsibility: define `Config`, autotune metadata container, decorator, and public error types.
- Create: `cutez/compiler.py`
  Responsibility: implement `cutez.compile(...)`, cache lookup, candidate reconstruction, and delegation to raw `cute.compile(...)` when no autotune metadata exists.
- Create: `cutez/benchmark.py`
  Responsibility: hold the default benchmark helper used by the autotuner.
- Create: `cutez/_autotune_keys.py`
  Responsibility: resolve `key=[...]` names from compile args and optional kernel hooks.
- Create: `tests/test_cutez_autotune.py`
  Responsibility: unit tests for decorator metadata, compile path selection, cache behavior, and error handling using fakes.

## Task 1: Scaffold Runtime Modules And Basic Exports

**Files:**
- Create: `cutez/autotune.py`
- Create: `cutez/compiler.py`
- Create: `cutez/benchmark.py`
- Create: `cutez/_autotune_keys.py`
- Modify: `cutez/__init__.py`
- Test: `tests/test_cutez_autotune.py`

- [ ] **Step 1: Write the failing export test**

```python
from cutez import Config, autotune, compile as cutez_compile


def test_public_autotune_exports_exist():
    assert Config is not None
    assert autotune is not None
    assert callable(cutez_compile)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cutez_autotune.py::test_public_autotune_exports_exist -v`
Expected: FAIL with `ModuleNotFoundError` for `tests/test_cutez_autotune.py` or `ImportError` because `Config`, `autotune`, and `compile` are not exported from `cutez`.

- [ ] **Step 3: Create the initial test file**

```python
from cutez import Config, autotune, compile as cutez_compile


def test_public_autotune_exports_exist():
    assert Config is not None
    assert autotune is not None
    assert callable(cutez_compile)
```

- [ ] **Step 4: Add minimal runtime scaffolding**

Create `cutez/autotune.py`:

```python
from dataclasses import dataclass, field
from typing import Any, Callable


AUTOTUNE_ATTR = "__cutez_autotune__"


class CutezAutotuneError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    kwargs: dict[str, Any]
    name: str | None = None
    pre_hook: Callable[..., Any] | None = None


@dataclass(frozen=True)
class AutotuneMeta:
    configs: tuple[Config, ...]
    key: tuple[str, ...]
    warmup: int = 5
    rep: int = 100
    cache_results: bool = True
    do_bench: Callable[..., Any] | None = None


def autotune(*, configs, key, warmup=5, rep=100, cache_results=True, do_bench=None):
    meta = AutotuneMeta(
        configs=tuple(configs),
        key=tuple(key),
        warmup=warmup,
        rep=rep,
        cache_results=cache_results,
        do_bench=do_bench,
    )

    def decorator(fn):
        setattr(fn, AUTOTUNE_ATTR, meta)
        return fn

    return decorator
```

Create `cutez/compiler.py`:

```python
import cutlass.cute as cute


def compile(kernel, *args, **kwargs):
    return cute.compile(kernel, *args, **kwargs)
```

Create `cutez/benchmark.py`:

```python
def default_benchmark(fn, *, warmup, rep):
    raise NotImplementedError("default benchmark not implemented yet")
```

Create `cutez/_autotune_keys.py`:

```python
def resolve_key_values(kernel, *args, **kwargs):
    return {}
```

Update `cutez/__init__.py` to add these imports near the top-level imports:

```python
from .autotune import Config, CutezAutotuneError, autotune
from .compiler import compile
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_cutez_autotune.py::test_public_autotune_exports_exist -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add cutez/__init__.py cutez/autotune.py cutez/compiler.py cutez/benchmark.py cutez/_autotune_keys.py tests/test_cutez_autotune.py
git commit -m "add cutez autotune scaffolding"
```

## Task 2: Store And Read Autotune Metadata

**Files:**
- Modify: `cutez/autotune.py`
- Modify: `cutez/compiler.py`
- Test: `tests/test_cutez_autotune.py`

- [ ] **Step 1: Write the failing metadata tests**

Append these tests to `tests/test_cutez_autotune.py`:

```python
from cutez.autotune import AUTOTUNE_ATTR, AutotuneMeta


def test_autotune_decorator_attaches_metadata_to_call():
    class Kernel:
        @autotune(configs=[Config(kwargs={"tile": 128})], key=["m"])
        def __call__(self, a):
            return a

    meta = getattr(Kernel.__call__, AUTOTUNE_ATTR)
    assert isinstance(meta, AutotuneMeta)
    assert meta.key == ("m",)
    assert meta.configs[0].kwargs == {"tile": 128}


def test_compile_delegates_when_no_autotune(monkeypatch):
    calls = []

    class FakeCute:
        @staticmethod
        def compile(kernel, *args, **kwargs):
            calls.append((kernel, args, kwargs))
            return "compiled"

    class Kernel:
        def __call__(self, a):
            return a

    monkeypatch.setattr("cutez.compiler.cute", FakeCute)
    result = cutez_compile(Kernel(), "arg")
    assert result == "compiled"
    assert len(calls) == 1
```

- [ ] **Step 2: Run tests to verify the first new failure**

Run: `pytest tests/test_cutez_autotune.py::test_compile_delegates_when_no_autotune -v`
Expected: FAIL if the compiler module path or monkeypatch target does not yet match the implementation.

- [ ] **Step 3: Add a metadata accessor helper and use it in compiler**

Update `cutez/autotune.py` with:

```python
def get_autotune_meta(callable_obj):
    return getattr(callable_obj, AUTOTUNE_ATTR, None)
```

Update `cutez/compiler.py` to:

```python
import cutlass.cute as cute

from .autotune import get_autotune_meta


def compile(kernel, *args, **kwargs):
    meta = get_autotune_meta(kernel.__call__)
    if meta is None:
        return cute.compile(kernel, *args, **kwargs)
    return cute.compile(kernel, *args, **kwargs)
```

- [ ] **Step 4: Run the metadata tests to verify they pass**

Run: `pytest tests/test_cutez_autotune.py::test_autotune_decorator_attaches_metadata_to_call tests/test_cutez_autotune.py::test_compile_delegates_when_no_autotune -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cutez/autotune.py cutez/compiler.py tests/test_cutez_autotune.py
git commit -m "store cutez autotune metadata"
```

## Task 3: Add Explicit Reconstruction And Key Hooks

**Files:**
- Modify: `cutez/compiler.py`
- Modify: `cutez/_autotune_keys.py`
- Modify: `cutez/autotune.py`
- Test: `tests/test_cutez_autotune.py`

- [ ] **Step 1: Write the failing reconstruction and key tests**

Append these tests to `tests/test_cutez_autotune.py`:

```python
import pytest


def test_autotuned_kernel_requires_reconstruction_hook(monkeypatch):
    class Kernel:
        @autotune(configs=[Config(kwargs={"tile": 128})], key=["m"])
        def __call__(self, a):
            return a

    class FakeCute:
        @staticmethod
        def compile(kernel, *args, **kwargs):
            return kernel

    monkeypatch.setattr("cutez.compiler.cute", FakeCute)

    with pytest.raises(CutezAutotuneError, match="autotune_init_kwargs"):
        cutez_compile(Kernel(), "arg")


def test_autotune_key_values_hook_overrides_defaults():
    class Kernel:
        def autotune_init_kwargs(self):
            return {"tile": 64}

        def autotune_key_values(self, a, b):
            return {"m": 99, "n": 77}

        @autotune(configs=[Config(kwargs={"tile": 128})], key=["m", "n"])
        def __call__(self, a, b):
            return a, b

    resolved = resolve_key_values(Kernel(), FakeTensor((4, 8)), FakeTensor((8, 16)))
    assert resolved["m"] == 99
    assert resolved["n"] == 77
```

- [ ] **Step 2: Run the missing-hook test to verify it fails**

Run: `pytest tests/test_cutez_autotune.py::test_autotuned_kernel_requires_reconstruction_hook -v`
Expected: FAIL because `cutez.compile(...)` still delegates instead of checking for `autotune_init_kwargs()`.

- [ ] **Step 3: Add shared test fakes for tensor metadata**

Add this helper near the top of `tests/test_cutez_autotune.py`:

```python
class FakeTensor:
    def __init__(self, shape, element_type="f16", stride=None):
        self.shape = shape
        self.element_type = element_type
        self.stride = stride if stride is not None else tuple(reversed(range(len(shape))))
```

- [ ] **Step 4: Implement key resolution and reconstruction validation**

Update `cutez/_autotune_keys.py` to:

```python
def _default_key_values(args):
    values = {}
    if len(args) >= 1 and hasattr(args[0], "shape") and len(args[0].shape) >= 2:
        values["m"] = args[0].shape[0]
        values["k"] = args[0].shape[1]
        values["a_dtype"] = getattr(args[0], "element_type", None)
    if len(args) >= 2 and hasattr(args[1], "shape") and len(args[1].shape) >= 1:
        values["n"] = args[1].shape[0]
        values["b_dtype"] = getattr(args[1], "element_type", None)
    if len(args) >= 3:
        values["c_dtype"] = getattr(args[2], "element_type", None)
    return values


def resolve_key_values(kernel, *args, **kwargs):
    values = _default_key_values(args)
    if hasattr(kernel, "autotune_key_values"):
        values.update(kernel.autotune_key_values(*args, **kwargs))
    return values
```

Update `cutez/compiler.py` to add:

```python
from ._autotune_keys import resolve_key_values
from .autotune import CutezAutotuneError, get_autotune_meta


def _require_init_kwargs(kernel):
    if not hasattr(kernel, "autotune_init_kwargs"):
        raise CutezAutotuneError("Autotuned kernels must implement autotune_init_kwargs()")
    return kernel.autotune_init_kwargs()


def _resolve_tuning_key(kernel, meta, args, kwargs):
    key_values = resolve_key_values(kernel, *args, **kwargs)
    missing = [name for name in meta.key if name not in key_values]
    if missing:
        raise CutezAutotuneError(f"Missing autotune key values: {missing}")
    return tuple(key_values[name] for name in meta.key)


def compile(kernel, *args, **kwargs):
    meta = get_autotune_meta(kernel.__call__)
    if meta is None:
        return cute.compile(kernel, *args, **kwargs)
    _require_init_kwargs(kernel)
    _resolve_tuning_key(kernel, meta, args, kwargs)
    return cute.compile(kernel, *args, **kwargs)
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `pytest tests/test_cutez_autotune.py::test_autotuned_kernel_requires_reconstruction_hook tests/test_cutez_autotune.py::test_autotune_key_values_hook_overrides_defaults -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add cutez/compiler.py cutez/_autotune_keys.py tests/test_cutez_autotune.py
git commit -m "add autotune key and rebuild hooks"
```

## Task 4: Compile One Candidate Per Config And Select The Fastest

**Files:**
- Modify: `cutez/compiler.py`
- Modify: `cutez/benchmark.py`
- Test: `tests/test_cutez_autotune.py`

- [ ] **Step 1: Write the failing candidate enumeration test**

Append this test to `tests/test_cutez_autotune.py`:

```python
def test_compile_builds_and_benchmarks_all_configs(monkeypatch):
    compiled_tiles = []
    bench_inputs = []

    class Kernel:
        def __init__(self, tile=64):
            self.tile = tile

        def autotune_init_kwargs(self):
            return {"tile": self.tile}

        def autotune_key_values(self, a):
            return {"m": a.shape[0]}

        @autotune(
            configs=[Config(kwargs={"tile": 64}), Config(kwargs={"tile": 128})],
            key=["m"],
        )
        def __call__(self, a):
            return a

    class FakeCompiled:
        def __init__(self, tile):
            self.tile = tile

        def __call__(self, *args, **kwargs):
            return self.tile

    class FakeCute:
        @staticmethod
        def compile(kernel, *args, **kwargs):
            compiled_tiles.append(kernel.tile)
            return FakeCompiled(kernel.tile)

    def fake_bench(compiled, args, kwargs, *, warmup, rep):
        bench_inputs.append(compiled.tile)
        return {64: 5.0, 128: 1.0}[compiled.tile]

    monkeypatch.setattr("cutez.compiler.cute", FakeCute)
    monkeypatch.setattr("cutez.compiler.default_benchmark", fake_bench)

    compiled = cutez_compile(Kernel(), FakeTensor((32, 64)))
    assert compiled.tile == 128
    assert compiled_tiles == [64, 128]
    assert bench_inputs == [64, 128]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cutez_autotune.py::test_compile_builds_and_benchmarks_all_configs -v`
Expected: FAIL because only one kernel is compiled and no benchmark loop exists yet.

- [ ] **Step 3: Implement benchmark wrapper and candidate loop**

Update `cutez/benchmark.py` to:

```python
import time


def default_benchmark(compiled, args, kwargs, *, warmup, rep):
    for _ in range(warmup):
        compiled(*args, **kwargs)
    start = time.perf_counter()
    for _ in range(rep):
        compiled(*args, **kwargs)
    end = time.perf_counter()
    return (end - start) / max(rep, 1)
```

Update `cutez/compiler.py` to add:

```python
from .benchmark import default_benchmark


def _rebuild_kernel(kernel, config):
    base_kwargs = kernel.autotune_init_kwargs()
    candidate_kwargs = {**base_kwargs, **config.kwargs}
    return type(kernel)(**candidate_kwargs)


def _bench_candidate(meta, compiled, args, kwargs):
    bench_fn = meta.do_bench or default_benchmark
    return bench_fn(compiled, args, kwargs, warmup=meta.warmup, rep=meta.rep)


def compile(kernel, *args, **kwargs):
    meta = get_autotune_meta(kernel.__call__)
    if meta is None:
        return cute.compile(kernel, *args, **kwargs)

    _require_init_kwargs(kernel)
    _resolve_tuning_key(kernel, meta, args, kwargs)

    best_time = None
    best_compiled = None
    for config in meta.configs:
        candidate_kernel = _rebuild_kernel(kernel, config)
        compiled_candidate = cute.compile(candidate_kernel, *args, **kwargs)
        current_time = _bench_candidate(meta, compiled_candidate, args, kwargs)
        if best_time is None or current_time < best_time:
            best_time = current_time
            best_compiled = compiled_candidate

    if best_compiled is None:
        raise CutezAutotuneError("No valid autotune candidates were compiled")
    return best_compiled
```

- [ ] **Step 4: Run the candidate test to verify it passes**

Run: `pytest tests/test_cutez_autotune.py::test_compile_builds_and_benchmarks_all_configs -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cutez/compiler.py cutez/benchmark.py tests/test_cutez_autotune.py
git commit -m "benchmark all cutez autotune configs"
```

## Task 5: Add Compile Cache And Best-Config Cache

**Files:**
- Modify: `cutez/compiler.py`
- Modify: `cutez/autotune.py`
- Test: `tests/test_cutez_autotune.py`

- [ ] **Step 1: Write the failing cache tests**

Append these tests to `tests/test_cutez_autotune.py`:

```python
def test_compile_reuses_cached_best_candidate(monkeypatch):
    compile_calls = []
    bench_calls = []

    class Kernel:
        def __init__(self, tile=64):
            self.tile = tile

        def autotune_init_kwargs(self):
            return {"tile": self.tile}

        def autotune_key_values(self, a):
            return {"m": a.shape[0]}

        @autotune(configs=[Config(kwargs={"tile": 64}), Config(kwargs={"tile": 128})], key=["m"])
        def __call__(self, a):
            return a

    class FakeCompiled:
        def __init__(self, tile):
            self.tile = tile

    class FakeCute:
        @staticmethod
        def compile(kernel, *args, **kwargs):
            compile_calls.append(kernel.tile)
            return FakeCompiled(kernel.tile)

    def fake_bench(compiled, args, kwargs, *, warmup, rep):
        bench_calls.append(compiled.tile)
        return {64: 4.0, 128: 1.0}[compiled.tile]

    monkeypatch.setattr("cutez.compiler.cute", FakeCute)
    monkeypatch.setattr("cutez.compiler.default_benchmark", fake_bench)

    first = cutez_compile(Kernel(), FakeTensor((32, 64)))
    second = cutez_compile(Kernel(), FakeTensor((32, 64)))

    assert first.tile == 128
    assert second.tile == 128
    assert compile_calls == [64, 128]
    assert bench_calls == [64, 128]


def test_compile_cache_reuses_compiled_variant_across_keys(monkeypatch):
    compile_calls = []

    class Kernel:
        def __init__(self, tile=64, dtype="f16"):
            self.tile = tile
            self.dtype = dtype

        def autotune_init_kwargs(self):
            return {"tile": self.tile, "dtype": self.dtype}

        def autotune_key_values(self, a):
            return {"m": a.shape[0]}

        @autotune(configs=[Config(kwargs={"tile": 64})], key=["m"])
        def __call__(self, a):
            return a

    class FakeCompiled:
        def __init__(self, tile):
            self.tile = tile

        def __call__(self, *args, **kwargs):
            return self.tile

    class FakeCute:
        @staticmethod
        def compile(kernel, *args, **kwargs):
            compile_calls.append((kernel.tile, args[0].element_type))
            return FakeCompiled(kernel.tile)

    monkeypatch.setattr("cutez.compiler.cute", FakeCute)
    monkeypatch.setattr("cutez.compiler.default_benchmark", lambda compiled, args, kwargs, *, warmup, rep: 1.0)

    cutez_compile(Kernel(), FakeTensor((32, 64), element_type="f16"))
    cutez_compile(Kernel(), FakeTensor((32, 64), element_type="f16"))

    assert compile_calls == [(64, "f16")]
```

- [ ] **Step 2: Run one cache test to verify it fails**

Run: `pytest tests/test_cutez_autotune.py::test_compile_reuses_cached_best_candidate -v`
Expected: FAIL because the compile and benchmark loops run again on the second call.

- [ ] **Step 3: Implement in-memory caches**

Update `cutez/compiler.py` to add module globals and helpers:

```python
_BEST_CONFIG_CACHE = {}
_BEST_COMPILED_CACHE = {}
_COMPILED_KERNEL_CACHE = {}


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _compile_signature(kernel, args, kwargs):
    arg_signature = []
    for arg in args:
        if hasattr(arg, "shape"):
            arg_signature.append(
                (
                    tuple(arg.shape),
                    getattr(arg, "element_type", None),
                    getattr(arg, "stride", None),
                )
            )
        else:
            arg_signature.append(arg)
    return (type(kernel), tuple(arg_signature), _freeze(kwargs))


def _candidate_cache_key(candidate_kernel, args, kwargs):
    return (
        type(candidate_kernel),
        _freeze(candidate_kernel.autotune_init_kwargs()),
        _compile_signature(candidate_kernel, args, kwargs),
    )
```

Then update the main `compile(...)` flow to:

```python
    tuning_key = _resolve_tuning_key(kernel, meta, args, kwargs)
    best_cache_key = (type(kernel), tuning_key)
    if best_cache_key in _BEST_COMPILED_CACHE:
        return _BEST_COMPILED_CACHE[best_cache_key]

    best_time = None
    best_compiled = None
    best_config = None
    for config in meta.configs:
        candidate_kernel = _rebuild_kernel(kernel, config)
        candidate_key = _candidate_cache_key(candidate_kernel, args, kwargs)
        if candidate_key not in _COMPILED_KERNEL_CACHE:
            _COMPILED_KERNEL_CACHE[candidate_key] = cute.compile(candidate_kernel, *args, **kwargs)
        compiled_candidate = _COMPILED_KERNEL_CACHE[candidate_key]
        current_time = _bench_candidate(meta, compiled_candidate, args, kwargs)
        if best_time is None or current_time < best_time:
            best_time = current_time
            best_compiled = compiled_candidate
            best_config = config

    if best_compiled is None:
        raise CutezAutotuneError("No valid autotune candidates were compiled")

    _BEST_CONFIG_CACHE[best_cache_key] = best_config
    _BEST_COMPILED_CACHE[best_cache_key] = best_compiled
    return best_compiled
```

- [ ] **Step 4: Add a cache reset helper for tests**

Update `cutez/compiler.py` with:

```python
def _reset_caches():
    _BEST_CONFIG_CACHE.clear()
    _BEST_COMPILED_CACHE.clear()
    _COMPILED_KERNEL_CACHE.clear()
```
```

Add this fixture to `tests/test_cutez_autotune.py`:

```python
import pytest


@pytest.fixture(autouse=True)
def reset_cutez_caches():
    from cutez.compiler import _reset_caches

    _reset_caches()
    yield
    _reset_caches()
```

- [ ] **Step 5: Run cache tests to verify they pass**

Run: `pytest tests/test_cutez_autotune.py::test_compile_reuses_cached_best_candidate tests/test_cutez_autotune.py::test_compile_cache_reuses_compiled_variant_across_keys -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add cutez/compiler.py tests/test_cutez_autotune.py
git commit -m "cache cutez autotune results"
```

## Task 6: Handle Candidate Failures And Empty Configs Cleanly

**Files:**
- Modify: `cutez/autotune.py`
- Modify: `cutez/compiler.py`
- Test: `tests/test_cutez_autotune.py`

- [ ] **Step 1: Write the failing failure-mode tests**

Append these tests to `tests/test_cutez_autotune.py`:

```python
import pytest


def test_autotune_requires_non_empty_configs():
    with pytest.raises(CutezAutotuneError, match="at least one config"):
        @autotune(configs=[], key=["m"])
        def fn(a):
            return a


def test_compile_skips_failed_candidates_and_uses_valid_one(monkeypatch):
    class Kernel:
        def __init__(self, tile=64):
            self.tile = tile

        def autotune_init_kwargs(self):
            return {"tile": self.tile}

        def autotune_key_values(self, a):
            return {"m": a.shape[0]}

        @autotune(configs=[Config(kwargs={"tile": 64}), Config(kwargs={"tile": 128})], key=["m"])
        def __call__(self, a):
            return a

    class FakeCompiled:
        def __init__(self, tile):
            self.tile = tile

        def __call__(self, *args, **kwargs):
            return self.tile

    class FakeCute:
        @staticmethod
        def compile(kernel, *args, **kwargs):
            if kernel.tile == 64:
                raise RuntimeError("bad config")
            return FakeCompiled(kernel.tile)

    monkeypatch.setattr("cutez.compiler.cute", FakeCute)
    monkeypatch.setattr("cutez.compiler.default_benchmark", lambda compiled, args, kwargs, *, warmup, rep: 1.0)

    compiled = cutez_compile(Kernel(), FakeTensor((32, 64)))
    assert compiled.tile == 128


def test_compile_raises_when_all_candidates_fail(monkeypatch):
    class Kernel:
        def __init__(self, tile=64):
            self.tile = tile

        def autotune_init_kwargs(self):
            return {"tile": self.tile}

        def autotune_key_values(self, a):
            return {"m": a.shape[0]}

        @autotune(configs=[Config(kwargs={"tile": 64})], key=["m"])
        def __call__(self, a):
            return a

    class FakeCute:
        @staticmethod
        def compile(kernel, *args, **kwargs):
            raise RuntimeError("always fails")

    monkeypatch.setattr("cutez.compiler.cute", FakeCute)

    with pytest.raises(CutezAutotuneError, match="No valid autotune candidates"):
        cutez_compile(Kernel(), FakeTensor((32, 64)))
```

- [ ] **Step 2: Run one failure-mode test to verify it fails**

Run: `pytest tests/test_cutez_autotune.py::test_autotune_requires_non_empty_configs -v`
Expected: FAIL because the decorator currently accepts empty config lists.

- [ ] **Step 3: Implement validation and candidate error collection**

Update `cutez/autotune.py` inside `autotune(...)`:

```python
    if not configs:
        raise CutezAutotuneError("cutez.autotune requires at least one config")
```

Update the candidate loop in `cutez/compiler.py` to:

```python
    failures = []
    for config in meta.configs:
        try:
            candidate_kernel = _rebuild_kernel(kernel, config)
            candidate_key = _candidate_cache_key(candidate_kernel, args, kwargs)
            if candidate_key not in _COMPILED_KERNEL_CACHE:
                _COMPILED_KERNEL_CACHE[candidate_key] = cute.compile(candidate_kernel, *args, **kwargs)
            compiled_candidate = _COMPILED_KERNEL_CACHE[candidate_key]
            current_time = _bench_candidate(meta, compiled_candidate, args, kwargs)
        except Exception as exc:
            failures.append((config.name or str(config.kwargs), str(exc)))
            continue

        if best_time is None or current_time < best_time:
            best_time = current_time
            best_compiled = compiled_candidate
            best_config = config

    if best_compiled is None:
        raise CutezAutotuneError(f"No valid autotune candidates were compiled: {failures}")
```

- [ ] **Step 4: Run the failure-mode tests to verify they pass**

Run: `pytest tests/test_cutez_autotune.py::test_autotune_requires_non_empty_configs tests/test_cutez_autotune.py::test_compile_skips_failed_candidates_and_uses_valid_one tests/test_cutez_autotune.py::test_compile_raises_when_all_candidates_fail -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cutez/autotune.py cutez/compiler.py tests/test_cutez_autotune.py
git commit -m "handle cutez autotune failures"
```

## Task 7: Final Verification And Public API Cleanup

**Files:**
- Modify: `cutez/__init__.py`
- Modify: `tests/test_cutez_autotune.py`

- [ ] **Step 1: Add a public API smoke test**

Append this test to `tests/test_cutez_autotune.py`:

```python
def test_public_api_smoke(monkeypatch):
    class Kernel:
        def __init__(self, tile=64):
            self.tile = tile

        def autotune_init_kwargs(self):
            return {"tile": self.tile}

        def autotune_key_values(self, a):
            return {"m": a.shape[0]}

        @autotune(configs=[Config(kwargs={"tile": 64}), Config(kwargs={"tile": 128})], key=["m"])
        def __call__(self, a):
            return a

    class FakeCompiled:
        def __init__(self, tile):
            self.tile = tile

        def __call__(self, *args, **kwargs):
            return self.tile

    class FakeCute:
        @staticmethod
        def compile(kernel, *args, **kwargs):
            return FakeCompiled(kernel.tile)

    monkeypatch.setattr("cutez.compiler.cute", FakeCute)
    monkeypatch.setattr("cutez.compiler.default_benchmark", lambda compiled, args, kwargs, *, warmup, rep: float(256 - compiled.tile))

    compiled = cutez_compile(Kernel(), FakeTensor((64, 128)))
    assert compiled.tile == 128
```

- [ ] **Step 2: Run the full focused test file**

Run: `pytest tests/test_cutez_autotune.py -v`
Expected: PASS for all autotune tests.

- [ ] **Step 3: Ensure public exports remain explicit**

Update the top of `cutez/__init__.py` to keep the new API imports grouped and easy to find:

```python
from .autotune import Config, CutezAutotuneError, autotune
from .compiler import compile
```

Do not move or refactor the existing math and CuTe helper exports unless required by import ordering.

- [ ] **Step 4: Re-run the full focused test file**

Run: `pytest tests/test_cutez_autotune.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cutez/__init__.py tests/test_cutez_autotune.py
git commit -m "finish cutez autotune api"
```

## Final Verification

- [ ] Run: `pytest tests/test_cutez_autotune.py -v`
Expected: PASS

- [ ] Run: `python -m compileall cutez`
Expected: PASS with `.py` files under `cutez` compiled successfully

- [ ] Run: `git status --short`
Expected: clean working tree or only expected unrelated files
