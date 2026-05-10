# Cutez Persistent Best Config Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional explicit-path JSON disk cache for `cutez` so `@cutez.autotune(...)` can persist and reload `tuning_key -> best_config` automatically across fresh Python processes.

**Architecture:** Extend `AutotuneSpec` with an optional `cache_path`, then add a small persistence layer in `cutez/compiler.py` that reads and writes a human-readable JSON best-config file and seeds the existing in-memory `_BEST_CONFIG_CACHE`. Keep compiled-artifact caching in memory only, and skip disk persistence automatically for unsafe cases such as configs with `pre_hook`.

**Tech Stack:** Python 3.13, standard-library JSON/pathlib, pytest, existing `cutez` autotune runtime

---

## File Structure

- Modify: `cutez/autotune.py`
  Responsibility: add `cache_path` to the public autotune metadata surface and keep metadata normalization explicit.
- Modify: `cutez/compiler.py`
  Responsibility: implement stable disk-key derivation, JSON read/write helpers, safe fallback-to-memory-only behavior, and integration with the existing in-memory best-config cache.
- Modify: `tests/test_cutez_autotune.py`
  Responsibility: regression tests for `cache_path` metadata, persisted best-config write/read, pre-hook skip behavior, malformed JSON fallback, and verbose disk-cache reporting.

## Task 1: Add `cache_path` To The Autotune API

**Files:**
- Modify: `cutez/autotune.py`
- Modify: `tests/test_cutez_autotune.py`

- [ ] **Step 1: Write the failing metadata test**

Append this test to `tests/test_cutez_autotune.py`:

```python
def test_autotune_decorator_stores_optional_cache_path(cutez_module):
    spec_holder = {}

    @cutez_module.autotune(
        configs=[cutez_module.Config(kwargs={"tile": 128})],
        key=["m"],
        cache_path="/tmp/cutez-cache.json",
    )
    def host_function(m, tile):
        return (m, tile)

    spec_holder["spec"] = importlib.import_module("cutez.autotune").get_autotune_spec(
        host_function
    )

    assert spec_holder["spec"].cache_path == "/tmp/cutez-cache.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cutez_autotune.py::test_autotune_decorator_stores_optional_cache_path -v`
Expected: FAIL because `AutotuneSpec` and `autotune(...)` do not yet accept `cache_path`.

- [ ] **Step 3: Add `cache_path` to the autotune metadata model**

Update `cutez/autotune.py` so `AutotuneSpec` includes:

```python
cache_path: str | None = None
```

and `autotune(...)` accepts:

```python
cache_path: str | None = None
```

then stores it in the normalized spec:

```python
spec = AutotuneSpec(
    configs=tuple(configs),
    key=tuple(key),
    warmup=warmup,
    rep=rep,
    cache_results=cache_results,
    do_bench=do_bench,
    cache_path=cache_path,
)
```

- [ ] **Step 4: Run the metadata test to verify it passes**

Run: `pytest tests/test_cutez_autotune.py::test_autotune_decorator_stores_optional_cache_path -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cutez/autotune.py tests/test_cutez_autotune.py
git commit -m "add cutez cache path metadata"
```

## Task 2: Persist And Reload Best Configs From JSON

**Files:**
- Modify: `cutez/compiler.py`
- Modify: `tests/test_cutez_autotune.py`

- [ ] **Step 1: Write the failing persistence tests**

Append these tests to `tests/test_cutez_autotune.py`:

```python
def test_compile_persists_best_config_to_json(cutez_module, monkeypatch, tmp_path):
    compiler_module = importlib.import_module("cutez.compiler")
    cache_path = tmp_path / "cutez-cache.json"

    def fake_compile(candidate_kernel, *args, **kwargs):
        return f"compiled:{kwargs['tile']}"

    def fake_benchmark(compiled_kernel, *args, **kwargs):
        return {"compiled:16": 2.0, "compiled:32": 1.0}[compiled_kernel]

    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(compiler_module, "benchmark", fake_benchmark, raising=False)

    @cutez_module.autotune(
        configs=[
            cutez_module.Config(kwargs={"tile": 16}, name="slow"),
            cutez_module.Config(kwargs={"tile": 32}, name="fast"),
        ],
        key=["m"],
        cache_path=str(cache_path),
    )
    def host_function(m, tile):
        return (m, tile)

    host_function.autotune_init_kwargs = lambda: {"m": 0, "tile": 0}
    host_function.autotune_key_values = lambda *args, **kwargs: {"m": 7}

    result = cutez_module.compile(host_function, "arg0", stream="stream0")

    assert result == "compiled:32"
    data = json.loads(cache_path.read_text())
    assert data["entries"][0]["kernel"].endswith("host_function")
    assert data["entries"][0]["key"] == [7]
    assert data["entries"][0]["config"]["kwargs"] == {"tile": 32}
    assert data["entries"][0]["config"]["name"] == "fast"


def test_compile_loads_best_config_from_json_before_benchmarking(
    cutez_module, monkeypatch, tmp_path
):
    compiler_module = importlib.import_module("cutez.compiler")
    cache_path = tmp_path / "cutez-cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "kernel": "tests.test_cutez_autotune.host_function",
                        "key": [7],
                        "config": {"kwargs": {"tile": 32}, "name": "fast"},
                    }
                ]
            }
        )
    )

    compile_calls = []
    benchmark_calls = []

    def fake_compile(candidate_kernel, *args, **kwargs):
        compile_calls.append(kwargs)
        return f"compiled:{kwargs['tile']}"

    def fake_benchmark(compiled_kernel, *args, **kwargs):
        benchmark_calls.append(compiled_kernel)
        return 999.0

    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(compiler_module, "benchmark", fake_benchmark, raising=False)

    @cutez_module.autotune(
        configs=[
            cutez_module.Config(kwargs={"tile": 16}, name="slow"),
            cutez_module.Config(kwargs={"tile": 32}, name="fast"),
        ],
        key=["m"],
        cache_path=str(cache_path),
    )
    def host_function(m, tile):
        return (m, tile)

    host_function.__module__ = "tests.test_cutez_autotune"
    host_function.autotune_init_kwargs = lambda: {"m": 0, "tile": 0}
    host_function.autotune_key_values = lambda *args, **kwargs: {"m": 7}

    compiler_module._BEST_CONFIG_CACHE.clear()
    result = cutez_module.compile(host_function, "arg0", stream="stream0")

    assert result == "compiled:32"
    assert compile_calls == [{"stream": "stream0", "m": 7, "tile": 32}]
    assert benchmark_calls == []
```

- [ ] **Step 2: Run the write-path persistence test to verify it fails**

Run: `pytest tests/test_cutez_autotune.py::test_compile_persists_best_config_to_json -v`
Expected: FAIL because `cache_path` is not yet used by `cutez.compile(...)`.

- [ ] **Step 3: Add minimal disk-cache helpers in `cutez/compiler.py`**

Add helpers with this responsibility split:

```python
def _kernel_disk_identity(kernel):
    ...


def _disk_cache_enabled(spec):
    return bool(spec.cache_path)


def _can_persist_configs(spec):
    return not any(config.pre_hook is not None for config in spec.configs)


def _load_best_config_from_disk(kernel, spec, tuning_key_tuple):
    ...


def _store_best_config_to_disk(kernel, spec, tuning_key_tuple, best_config):
    ...
```

Implement a human-readable JSON file shape:

```python
{
    "entries": [
        {
            "kernel": "module.qualname",
            "key": [7],
            "config": {
                "kwargs": {"tile": 32},
                "name": "fast",
            },
        }
    ]
}
```

Then integrate the helpers into `compile(...)` so the flow becomes:

1. check `_BEST_CONFIG_CACHE`
2. on miss, if disk cache is enabled and safe, load best config from JSON and seed `_BEST_CONFIG_CACHE`
3. after benchmarking finds a winner, store it to disk when disk cache is enabled and safe

Persist only `kwargs` and `name`; reconstructed configs should come back as `Config(kwargs=..., name=..., pre_hook=None)`.

- [ ] **Step 4: Run the write/read persistence tests to verify they pass**

Run: `pytest tests/test_cutez_autotune.py::test_compile_persists_best_config_to_json tests/test_cutez_autotune.py::test_compile_loads_best_config_from_json_before_benchmarking -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cutez/compiler.py tests/test_cutez_autotune.py
git commit -m "persist cutez best configs"
```

## Task 3: Safe Fallbacks And Verbose Disk-Cache Reporting

**Files:**
- Modify: `cutez/compiler.py`
- Modify: `tests/test_cutez_autotune.py`

- [ ] **Step 1: Write the failing safety tests**

Append these tests to `tests/test_cutez_autotune.py`:

```python
def test_compile_skips_disk_persistence_when_config_has_pre_hook(
    cutez_module, monkeypatch, tmp_path
):
    compiler_module = importlib.import_module("cutez.compiler")
    cache_path = tmp_path / "cutez-cache.json"

    monkeypatch.setattr(
        compiler_module.cute, "compile", lambda candidate_kernel, *args, **kwargs: "compiled"
    )
    monkeypatch.setattr(compiler_module, "benchmark", lambda *args, **kwargs: 1.0, raising=False)

    @cutez_module.autotune(
        configs=[
            cutez_module.Config(kwargs={"tile": 32}, pre_hook=lambda *a, **k: None)
        ],
        key=["m"],
        cache_path=str(cache_path),
    )
    def host_function(m, tile):
        return (m, tile)

    host_function.autotune_init_kwargs = lambda: {"m": 0, "tile": 0}
    host_function.autotune_key_values = lambda *args, **kwargs: {"m": 7}

    cutez_module.compile(host_function, "arg0", stream="stream0")

    assert not cache_path.exists()


def test_compile_ignores_malformed_disk_cache_and_rebuilds_in_memory(
    cutez_module, monkeypatch, tmp_path
):
    compiler_module = importlib.import_module("cutez.compiler")
    cache_path = tmp_path / "cutez-cache.json"
    cache_path.write_text("{not valid json")

    compile_calls = []

    def fake_compile(candidate_kernel, *args, **kwargs):
        compile_calls.append(kwargs)
        return f"compiled:{kwargs['tile']}"

    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(compiler_module, "benchmark", lambda compiled, *args, **kwargs: {"compiled:16": 2.0, "compiled:32": 1.0}[compiled], raising=False)

    @cutez_module.autotune(
        configs=[
            cutez_module.Config(kwargs={"tile": 16}),
            cutez_module.Config(kwargs={"tile": 32}),
        ],
        key=["m"],
        cache_path=str(cache_path),
    )
    def host_function(m, tile):
        return (m, tile)

    host_function.autotune_init_kwargs = lambda: {"m": 0, "tile": 0}
    host_function.autotune_key_values = lambda *args, **kwargs: {"m": 7}

    result = cutez_module.compile(host_function, "arg0", stream="stream0")

    assert result == "compiled:32"
    assert len(compile_calls) == 2


def test_compile_verbose_reports_disk_cache_hit(cutez_module, monkeypatch, tmp_path, capsys):
    compiler_module = importlib.import_module("cutez.compiler")
    cache_path = tmp_path / "cutez-cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "kernel": "tests.test_cutez_autotune.host_function",
                        "key": [7],
                        "config": {"kwargs": {"tile": 32}, "name": "fast"},
                    }
                ]
            }
        )
    )

    monkeypatch.setattr(
        compiler_module.cute, "compile", lambda candidate_kernel, *args, **kwargs: f"compiled:{kwargs['tile']}"
    )

    @cutez_module.autotune(
        configs=[
            cutez_module.Config(kwargs={"tile": 16}, name="slow"),
            cutez_module.Config(kwargs={"tile": 32}, name="fast"),
        ],
        key=["m"],
        cache_path=str(cache_path),
    )
    def host_function(m, tile):
        return (m, tile)

    host_function.__module__ = "tests.test_cutez_autotune"
    host_function.autotune_init_kwargs = lambda: {"m": 0, "tile": 0}
    host_function.autotune_key_values = lambda *args, **kwargs: {"m": 7}

    compiler_module._BEST_CONFIG_CACHE.clear()
    cutez_module.compile(host_function, "arg0", stream="stream0", verbose=True)
    captured = capsys.readouterr()

    assert "disk_cache_hit" in captured.out
    assert "best_config={'tile': 32}" in captured.out
```

- [ ] **Step 2: Run one safety test to verify it fails**

Run: `pytest tests/test_cutez_autotune.py::test_compile_verbose_reports_disk_cache_hit -v`
Expected: FAIL because no disk-cache verbose message exists yet.

- [ ] **Step 3: Add safe fallback and verbose reporting behavior**

Update `cutez/compiler.py` so:

- persistence is skipped when any config has `pre_hook`
- malformed JSON falls back to memory-only behavior without raising
- `verbose=True` prints a distinct disk-cache message when a persisted best config is loaded, for example:

```python
print(f"[cutez.autotune] disk_cache_hit key={tuning_key[1]} best_config={loaded_config.kwargs}")
```

Keep in-memory cache-hit logging distinct from disk-cache-hit logging.

- [ ] **Step 4: Run the safety tests to verify they pass**

Run: `pytest tests/test_cutez_autotune.py::test_compile_skips_disk_persistence_when_config_has_pre_hook tests/test_cutez_autotune.py::test_compile_ignores_malformed_disk_cache_and_rebuilds_in_memory tests/test_cutez_autotune.py::test_compile_verbose_reports_disk_cache_hit -v`
Expected: PASS

- [ ] **Step 5: Run the full focused autotune suite**

Run: `pytest tests/test_cutez_autotune.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add cutez/compiler.py tests/test_cutez_autotune.py
git commit -m "handle cutez persistent cache fallbacks"
```

## Final Verification

- [ ] Run: `pytest tests/test_cutez_autotune.py -v`
Expected: PASS

- [ ] Run: `python -m compileall cutez`
Expected: PASS

- [ ] Run: `git status --short`
Expected: clean working tree or only unrelated pre-existing files
