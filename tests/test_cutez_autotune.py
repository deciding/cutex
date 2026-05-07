from pathlib import Path
import importlib
import sys

import pytest


class Constexpr:
    pass


@pytest.fixture
def cutez_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))

    for name in ["cutez", "cutez.autotune", "cutez.compiler"]:
        sys.modules.pop(name, None)

    return importlib.import_module("cutez")


def test_autotune_decorator_stores_normalized_metadata(cutez_module):
    autotune_module = importlib.import_module("cutez.autotune")
    pre_hook = object()
    config = cutez_module.Config(kwargs={"tile": 128}, name="fast", pre_hook=pre_hook)

    class Kernel:
        @cutez_module.autotune(configs=[config], key=["m", "n"], warmup=3, rep=7)
        def __call__(self, *args, **kwargs):
            return args, kwargs

    spec = autotune_module.get_autotune_spec(Kernel())

    assert spec.configs == (config,)
    assert spec.key == ("m", "n")
    assert spec.warmup == 3
    assert spec.rep == 7
    assert spec.cache_results is True
    assert spec.do_bench is None
    assert spec.configs[0].kwargs == {"tile": 128}
    assert spec.configs[0].name == "fast"
    assert spec.configs[0].pre_hook is pre_hook


def test_autotune_decorator_rejects_empty_config_lists(cutez_module):
    with pytest.raises(ValueError, match="at least one config"):

        class Kernel:
            @cutez_module.autotune(configs=[], key=["m"])
            def __call__(self, *args, **kwargs):
                return args, kwargs


def test_compile_forwards_to_cutlass_cute_compile(cutez_module, monkeypatch):
    compiler_module = importlib.import_module("cutez.compiler")
    calls = []
    expected_result = object()

    def fake_compile(*args, **kwargs):
        calls.append((args, kwargs))
        return expected_result

    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)

    kernel = object()
    arg1 = object()
    arg2 = object()
    result = cutez_module.compile(kernel, arg1, arg2, stream="stream0")

    assert result is expected_result
    assert calls == [((kernel, arg1, arg2), {"stream": "stream0"})]


def test_public_autotune_api_has_explicit_stable_exports(cutez_module):
    exported_names = set(getattr(cutez_module, "__all__", dir(cutez_module)))

    assert {"Config", "autotune", "compile"}.issubset(exported_names)
    assert "get_smem_store_op" in exported_names
    assert cutez_module.Config is importlib.import_module("cutez.autotune").Config
    assert cutez_module.autotune is importlib.import_module("cutez.autotune").autotune
    assert cutez_module.compile is importlib.import_module("cutez.compiler").compile


def test_compile_reads_decorated_call_metadata_before_delegating(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")
    calls = []
    read_calls = []

    def fake_read_autotune_spec(kernel):
        spec = autotune_module.get_autotune_spec(kernel)
        read_calls.append((kernel, spec))
        return spec

    def fake_compile(*args, **kwargs):
        calls.append((args, kwargs))
        return "compiled"

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )
    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)

    class Kernel:
        def __init__(self, m, tile):
            self.m = m
            self.tile = tile

        def autotune_init_kwargs(self):
            return {"m": self.m, "tile": self.tile}

        def autotune_key_values(self, *args, **kwargs):
            return {"m": 1}

        @cutez_module.autotune(
            configs=[cutez_module.Config(kwargs={"tile": 64})], key=["m"]
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    kernel = Kernel(m=1, tile=0)

    result = cutez_module.compile(kernel, "arg0", stream="stream0")

    assert result == "compiled"
    assert len(read_calls) == 1
    assert read_calls[0][0] is kernel
    assert read_calls[0][1] is not None
    assert read_calls[0][1].key == ("m",)
    assert len(calls) == 1
    assert calls[0][0][0].tile == 64
    assert calls[0][0][1:] == ("arg0",)
    assert calls[0][1] == {"stream": "stream0"}


def test_compile_reads_decorated_function_metadata_before_delegating(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")
    calls = []
    read_calls = []

    def fake_read_autotune_spec(kernel):
        spec = autotune_module.get_autotune_spec(kernel)
        read_calls.append((kernel, spec))
        return spec

    def fake_compile(*args, **kwargs):
        calls.append((args, kwargs))
        return "compiled"

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )
    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)

    class Kernel:
        def __init__(self, n, tile):
            self.n = n
            self.tile = tile

        def autotune_init_kwargs(self):
            return {"n": self.n, "tile": self.tile}

        def autotune_key_values(self, *args, **kwargs):
            return {"n": 2}

        @cutez_module.autotune(
            configs=[cutez_module.Config(kwargs={"tile": 32})], key=["n"]
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    kernel = Kernel(n=2, tile=0)

    result = cutez_module.compile(kernel, "arg0", stream="stream0")

    assert result == "compiled"
    assert len(read_calls) == 1
    assert read_calls[0][0] is kernel
    assert read_calls[0][1] is not None
    assert read_calls[0][1].key == ("n",)
    assert len(calls) == 1
    assert calls[0][0][0].tile == 32
    assert calls[0][0][1:] == ("arg0",)
    assert calls[0][1] == {"stream": "stream0"}


def test_compile_requires_autotune_init_kwargs_for_decorated_kernels(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    monkeypatch.setattr(
        compiler_module.cute, "compile", lambda *args, **kwargs: "compiled"
    )

    class Kernel:
        def autotune_key_values(self, *args, **kwargs):
            return {"m": 1}

        @cutez_module.autotune(
            configs=[cutez_module.Config(kwargs={"tile": 16})], key=["m"]
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    with pytest.raises(AttributeError, match="autotune_init_kwargs"):
        cutez_module.compile(Kernel(), "arg0", stream="stream0")


def test_compile_prefers_explicit_autotune_key_values_over_defaults(
    cutez_module, monkeypatch
):
    from cutez._autotune_keys import resolve_autotune_key_values

    class Kernel:
        def autotune_init_kwargs(self):
            return {"m": 1, "n": 2}

        def autotune_key_values(self, *args, **kwargs):
            return {"m": 7, "extra": 9}

    values = resolve_autotune_key_values(Kernel(), "arg0", stream="stream0")

    assert values == {"m": 7, "n": 2, "extra": 9}


def test_compile_autotunes_decorated_plain_host_functions(cutez_module, monkeypatch):
    compiler_module = importlib.import_module("cutez.compiler")
    compile_calls = []
    benchmark_calls = []

    def fake_compile(candidate_kernel, *args, **kwargs):
        compile_calls.append((candidate_kernel, args, kwargs))
        return f"compiled:{kwargs['tile']}"

    def fake_benchmark(compiled_kernel, *args, **kwargs):
        benchmark_calls.append((compiled_kernel, args, kwargs))
        return {"compiled:16": 2.0, "compiled:32": 1.0}[compiled_kernel]

    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(compiler_module, "benchmark", fake_benchmark, raising=False)

    @cutez_module.autotune(
        configs=[
            cutez_module.Config(kwargs={"tile": 16}),
            cutez_module.Config(kwargs={"tile": 32}),
        ],
        key=["m"],
    )
    def host_function(m, tile):
        return (m, tile)

    host_function.autotune_init_kwargs = lambda: {"m": 0, "tile": 0}
    host_function.autotune_key_values = lambda *args, **kwargs: {"m": 7}

    result = cutez_module.compile(host_function, "arg0", stream="stream0")

    assert result == "compiled:32"
    assert [call[0] for call in compile_calls] == [host_function, host_function]
    assert [call[1] for call in compile_calls] == [("arg0",), ("arg0",)]
    assert [call[2] for call in compile_calls] == [
        {"stream": "stream0", "m": 7, "tile": 16},
        {"stream": "stream0", "m": 7, "tile": 32},
    ]
    assert [call[0] for call in benchmark_calls] == ["compiled:16", "compiled:32"]


def test_compile_reuses_cached_plain_function_candidate_on_repeated_calls(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    compile_calls = []
    benchmark_calls = []

    def fake_compile(candidate_kernel, *args, **kwargs):
        compile_calls.append((candidate_kernel, args, kwargs))
        return f"compiled:{kwargs['tile']}"

    def fake_benchmark(compiled_kernel, *args, **kwargs):
        benchmark_calls.append((compiled_kernel, args, kwargs))
        return {"compiled:16": 2.0, "compiled:32": 1.0}[compiled_kernel]

    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(compiler_module, "benchmark", fake_benchmark, raising=False)

    @cutez_module.autotune(
        configs=[
            cutez_module.Config(kwargs={"tile": 16}),
            cutez_module.Config(kwargs={"tile": 32}),
        ],
        key=["m"],
    )
    def host_function(m, tile):
        return (m, tile)

    host_function.autotune_init_kwargs = lambda: {"m": 0, "tile": 0}
    host_function.autotune_key_values = lambda *args, **kwargs: {"m": 7}

    result0 = cutez_module.compile(host_function, "arg0", stream="stream0")
    result1 = cutez_module.compile(host_function, "arg0", stream="stream0")

    assert result0 == "compiled:32"
    assert result1 == "compiled:32"
    assert [call[0] for call in compile_calls] == [host_function, host_function]
    assert [call[1] for call in compile_calls] == [("arg0",), ("arg0",)]
    assert [call[2] for call in compile_calls] == [
        {"stream": "stream0", "m": 7, "tile": 16},
        {"stream": "stream0", "m": 7, "tile": 32},
    ]
    assert [call[0] for call in benchmark_calls] == ["compiled:16", "compiled:32"]


def test_compile_passes_runtime_args_and_autotune_kwargs_for_plain_host_functions(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    compile_calls = []
    benchmark_calls = []

    def fake_compile(candidate_kernel, *args, **kwargs):
        compile_calls.append((candidate_kernel, args, kwargs))
        return f"compiled:{kwargs['tile']}"

    def fake_benchmark(compiled_kernel, *args, **kwargs):
        benchmark_calls.append((compiled_kernel, args, kwargs))
        return {"compiled:16": 2.0, "compiled:32": 1.0}[compiled_kernel]

    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(compiler_module, "benchmark", fake_benchmark, raising=False)

    @cutez_module.autotune(
        configs=[
            cutez_module.Config(kwargs={"tile": 16}),
            cutez_module.Config(kwargs={"tile": 32}),
        ],
        key=["m"],
    )
    def host_function(a, b, stream, m, tile):
        return (a, b, stream, m, tile)

    host_function.autotune_init_kwargs = lambda: {"m": 0, "tile": 0}
    host_function.autotune_key_values = lambda a, b, *args, **kwargs: {
        "m": len(a) + len(b)
    }

    result = cutez_module.compile(host_function, "aa", "bbb", stream="stream0")

    assert result == "compiled:32"
    assert [call[0] for call in compile_calls] == [host_function, host_function]
    assert [call[1] for call in compile_calls] == [("aa", "bbb"), ("aa", "bbb")]
    assert [call[2] for call in compile_calls] == [
        {"stream": "stream0", "m": 5, "tile": 16},
        {"stream": "stream0", "m": 5, "tile": 32},
    ]
    assert [call[0] for call in benchmark_calls] == ["compiled:16", "compiled:32"]


def test_compile_benchmarks_plain_host_functions_without_constexpr_positional_args(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    compile_calls = []
    benchmark_calls = []

    def compiled_runtime(a, b, stream):
        return (a, b, stream)

    def fake_compile(candidate_kernel, *args, **kwargs):
        compile_calls.append((candidate_kernel, args, kwargs))
        return compiled_runtime

    def fake_benchmark(compiled_kernel, *args, **kwargs):
        benchmark_calls.append((compiled_kernel, args, kwargs))
        return 1.0

    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(compiler_module, "benchmark", fake_benchmark, raising=False)

    @cutez_module.autotune(
        configs=[cutez_module.Config(kwargs={"tile": 32})],
        key=["m"],
    )
    def host_function(a, b, compile_only: Constexpr, stream, m, tile):
        return (a, b, compile_only, stream, m, tile)

    host_function.autotune_init_kwargs = lambda: {"m": 0, "tile": 0}
    host_function.autotune_key_values = lambda a, b, *args, **kwargs: {
        "m": len(a) + len(b)
    }

    result = cutez_module.compile(host_function, "aa", "bbb", True, stream="stream0")

    assert callable(result)
    assert [call[1] for call in compile_calls] == [("aa", "bbb", True)]
    assert [call[2] for call in compile_calls] == [
        {"stream": "stream0", "m": 5, "tile": 32}
    ]
    assert len(benchmark_calls) == 1
    assert benchmark_calls[0][1] == ("aa", "bbb")
    assert benchmark_calls[0][2] == {"stream": "stream0", "warmup": 0, "rep": 0}


def test_compile_benchmarks_plain_host_functions_without_compile_time_kwargs(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    compile_calls = []
    benchmark_calls = []

    def compiled_runtime(a, b, stream):
        return (a, b, stream)

    def fake_compile(candidate_kernel, *args, **kwargs):
        compile_calls.append((candidate_kernel, args, kwargs))
        return compiled_runtime

    def fake_benchmark(compiled_kernel, *args, **kwargs):
        benchmark_calls.append((compiled_kernel, args, kwargs))
        return 1.0

    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(compiler_module, "benchmark", fake_benchmark, raising=False)

    @cutez_module.autotune(
        configs=[cutez_module.Config(kwargs={"tile": 32})],
        key=["m"],
    )
    def host_function(a, b, stream, *, compile_only: Constexpr, m, tile):
        return (a, b, stream, compile_only, m, tile)

    host_function.autotune_init_kwargs = lambda: {"m": 0, "tile": 0}
    host_function.autotune_key_values = lambda a, b, *args, **kwargs: {
        "m": len(a) + len(b)
    }

    result = cutez_module.compile(
        host_function,
        "aa",
        "bbb",
        stream="stream0",
        compile_only=True,
    )

    assert callable(result)
    assert [call[1] for call in compile_calls] == [("aa", "bbb")]
    assert [call[2] for call in compile_calls] == [
        {"stream": "stream0", "compile_only": True, "m": 5, "tile": 32}
    ]
    assert len(benchmark_calls) == 1
    assert benchmark_calls[0][1] == ("aa", "bbb")
    assert benchmark_calls[0][2] == {"stream": "stream0", "warmup": 0, "rep": 0}


def test_compile_benchmark_preserves_later_runtime_kwargs_after_sparse_binding(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    compile_calls = []
    benchmark_calls = []

    def compiled_runtime(a, b, stream):
        return (a, b, stream)

    def fake_compile(candidate_kernel, *args, **kwargs):
        compile_calls.append((candidate_kernel, args, kwargs))
        return compiled_runtime

    def fake_benchmark(compiled_kernel, *args, **kwargs):
        benchmark_calls.append((compiled_kernel, args, kwargs))
        return 1.0

    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(compiler_module, "benchmark", fake_benchmark, raising=False)

    @cutez_module.autotune(
        configs=[cutez_module.Config(kwargs={"tile": 32})],
        key=["m"],
    )
    def host_function(a, b, optional_runtime=None, stream=None, *, m, tile):
        return (a, b, optional_runtime, stream, m, tile)

    host_function.autotune_init_kwargs = lambda: {"m": 0, "tile": 0}
    host_function.autotune_key_values = lambda a, b, *args, **kwargs: {
        "m": len(a) + len(b)
    }

    result = cutez_module.compile(host_function, "aa", "bbb", stream="stream0")

    assert callable(result)
    assert [call[1] for call in compile_calls] == [("aa", "bbb")]
    assert [call[2] for call in compile_calls] == [
        {"stream": "stream0", "m": 5, "tile": 32}
    ]
    assert len(benchmark_calls) == 1
    assert benchmark_calls[0][1] == ("aa", "bbb")
    assert benchmark_calls[0][2] == {"stream": "stream0", "warmup": 0, "rep": 0}


def test_compile_isolates_plain_function_caches_by_function_identity(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    compile_calls = []
    benchmark_calls = []

    def fake_compile(candidate_kernel, *args, **kwargs):
        compile_calls.append((candidate_kernel, args, kwargs))
        return f"compiled:{candidate_kernel.__name__}:{kwargs['tile']}"

    def fake_benchmark(compiled_kernel, *args, **kwargs):
        benchmark_calls.append((compiled_kernel, args, kwargs))
        return {
            "compiled:host_function_a:16": 2.0,
            "compiled:host_function_a:32": 1.0,
            "compiled:host_function_b:16": 2.0,
            "compiled:host_function_b:32": 1.0,
        }[compiled_kernel]

    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(compiler_module, "benchmark", fake_benchmark, raising=False)

    @cutez_module.autotune(
        configs=[
            cutez_module.Config(kwargs={"tile": 16}),
            cutez_module.Config(kwargs={"tile": 32}),
        ],
        key=["m"],
    )
    def host_function_a(m, tile):
        return ("a", m, tile)

    @cutez_module.autotune(
        configs=[
            cutez_module.Config(kwargs={"tile": 16}),
            cutez_module.Config(kwargs={"tile": 32}),
        ],
        key=["m"],
    )
    def host_function_b(m, tile):
        return ("b", m, tile)

    for host_function in (host_function_a, host_function_b):
        host_function.autotune_init_kwargs = lambda: {"m": 0, "tile": 0}
        host_function.autotune_key_values = lambda *args, **kwargs: {"m": 7}

    result_a = cutez_module.compile(host_function_a, "arg0", stream="stream0")
    result_b = cutez_module.compile(host_function_b, "arg0", stream="stream0")

    assert result_a == "compiled:host_function_a:32"
    assert result_b == "compiled:host_function_b:32"
    assert [call[0] for call in compile_calls] == [
        host_function_a,
        host_function_a,
        host_function_b,
        host_function_b,
    ]
    assert [call[2] for call in compile_calls] == [
        {"stream": "stream0", "m": 7, "tile": 16},
        {"stream": "stream0", "m": 7, "tile": 32},
        {"stream": "stream0", "m": 7, "tile": 16},
        {"stream": "stream0", "m": 7, "tile": 32},
    ]
    assert [call[0] for call in benchmark_calls] == [
        "compiled:host_function_a:16",
        "compiled:host_function_a:32",
        "compiled:host_function_b:16",
        "compiled:host_function_b:32",
    ]


def test_compile_compiles_and_benchmarks_every_config_and_returns_fastest_candidate(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")
    compile_calls = []
    benchmark_calls = []

    def fake_read_autotune_spec(kernel):
        return autotune_module.get_autotune_spec(kernel)

    def fake_compile(candidate_kernel, *args, **kwargs):
        compile_calls.append((candidate_kernel, args, kwargs))
        return f"compiled:{candidate_kernel.mma_tiler_mn}"

    def fake_benchmark(compiled_kernel, *args, **kwargs):
        benchmark_calls.append((compiled_kernel, args, kwargs))
        return {"compiled:(128, 256)": 2.0, "compiled:(256, 256)": 1.0}[compiled_kernel]

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )
    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(compiler_module, "benchmark", fake_benchmark, raising=False)

    class Kernel:
        def __init__(self, mma_tiler_mn, m):
            self.mma_tiler_mn = mma_tiler_mn
            self.m = m

        def autotune_init_kwargs(self):
            return {"mma_tiler_mn": self.mma_tiler_mn, "m": self.m}

        def autotune_key_values(self, *args, **kwargs):
            return {"m": 1}

        @cutez_module.autotune(
            configs=[
                cutez_module.Config(kwargs={"mma_tiler_mn": (128, 256)}),
                cutez_module.Config(kwargs={"mma_tiler_mn": (256, 256)}),
            ],
            key=["m"],
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    result = cutez_module.compile(Kernel((64, 64), 0), "arg0", stream="stream0")

    assert result == "compiled:(256, 256)"
    assert [call[0].mma_tiler_mn for call in compile_calls] == [(128, 256), (256, 256)]
    assert [call[0] for call in benchmark_calls] == [
        "compiled:(128, 256)",
        "compiled:(256, 256)",
    ]


def test_compile_uses_runtime_key_values_when_reconstructing_candidates(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")
    seen = []

    def fake_read_autotune_spec(kernel):
        return autotune_module.get_autotune_spec(kernel)

    def fake_compile(candidate_kernel, *args, **kwargs):
        seen.append((candidate_kernel.m, candidate_kernel.tile))
        return "compiled"

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )
    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(
        compiler_module, "benchmark", lambda fn, *args, **kwargs: 1.0, raising=False
    )

    class Kernel:
        def __init__(self, m, tile):
            self.m = m
            self.tile = tile

        def autotune_init_kwargs(self):
            return {"m": self.m, "tile": self.tile}

        def autotune_key_values(self, *args, **kwargs):
            return {"m": 99}

        @cutez_module.autotune(
            configs=[cutez_module.Config(kwargs={"tile": 8})], key=["m"]
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    cutez_module.compile(Kernel(7, 0), "arg0", stream="stream0")

    assert seen == [(99, 8)]


def test_compile_runs_config_pre_hook_and_custom_do_bench(cutez_module, monkeypatch):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")
    events = []

    def fake_read_autotune_spec(kernel):
        return autotune_module.get_autotune_spec(kernel)

    def fake_compile(candidate_kernel, *args, **kwargs):
        return candidate_kernel

    def fake_do_bench(fn, *args, warmup=0, rep=0, **kwargs):
        events.append((warmup, rep, args, kwargs))
        return 0.5

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )
    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(
        compiler_module, "benchmark", lambda *args, **kwargs: 9.0, raising=False
    )

    class Kernel:
        def __init__(self, m, tile):
            self.m = m
            self.tile = tile

        def autotune_init_kwargs(self):
            return {"m": self.m, "tile": self.tile}

        def autotune_key_values(self, *args, **kwargs):
            return {"m": 1}

        @cutez_module.autotune(
            configs=[
                cutez_module.Config(
                    kwargs={"tile": 16},
                    pre_hook=lambda candidate, *a, **k: events.append(
                        (candidate.tile, a, k)
                    ),
                )
            ],
            key=["m"],
            warmup=3,
            rep=5,
            do_bench=fake_do_bench,
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    result = cutez_module.compile(Kernel(1, 0), "arg0", stream="stream0")

    assert result.tile == 16
    assert events == [
        (16, ("arg0",), {"stream": "stream0"}),
        (3, 5, ("arg0",), {"stream": "stream0"}),
    ]


def test_compile_raises_when_candidate_constructor_rejects_config(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")

    def fake_read_autotune_spec(kernel):
        return autotune_module.get_autotune_spec(kernel)

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )

    class Kernel:
        def autotune_init_kwargs(self):
            return {"m": 1}

        def autotune_key_values(self, *args, **kwargs):
            return {"m": 1}

        @cutez_module.autotune(
            configs=[cutez_module.Config(kwargs={"tile": 16})], key=["m"]
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    with pytest.raises(
        autotune_module.AutotuneError, match="all autotune candidates failed"
    ) as exc_info:
        cutez_module.compile(Kernel(), "arg0", stream="stream0")

    assert "Kernel() takes no arguments" in str(exc_info.value)


def test_compile_skips_failed_candidates_when_another_config_succeeds(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")
    compile_calls = []
    benchmark_calls = []

    def fake_read_autotune_spec(kernel):
        return autotune_module.get_autotune_spec(kernel)

    def fake_compile(candidate_kernel, *args, **kwargs):
        compile_calls.append(candidate_kernel.tile)
        if candidate_kernel.tile == 16:
            raise RuntimeError("compile failed for tile 16")
        return f"compiled:{candidate_kernel.tile}"

    def fake_benchmark(compiled_kernel, *args, **kwargs):
        benchmark_calls.append(compiled_kernel)
        return 1.0

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )
    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(compiler_module, "benchmark", fake_benchmark, raising=False)

    class Kernel:
        def __init__(self, m, tile):
            self.m = m
            self.tile = tile

        def autotune_init_kwargs(self):
            return {"m": self.m, "tile": self.tile}

        def autotune_key_values(self, *args, **kwargs):
            return {"m": self.m}

        @cutez_module.autotune(
            configs=[
                cutez_module.Config(kwargs={"tile": 16}, name="broken"),
                cutez_module.Config(kwargs={"tile": 32}, name="working"),
            ],
            key=["m"],
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    result = cutez_module.compile(Kernel(7, 0), "arg0", stream="stream0")

    assert result == "compiled:32"
    assert compile_calls == [16, 32]
    assert benchmark_calls == ["compiled:32"]


def test_compile_raises_autotune_error_when_all_candidates_fail(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")

    def fake_read_autotune_spec(kernel):
        return autotune_module.get_autotune_spec(kernel)

    def fake_compile(candidate_kernel, *args, **kwargs):
        raise RuntimeError(f"compile failed for tile {candidate_kernel.tile}")

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )
    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)

    class Kernel:
        def __init__(self, m, tile):
            self.m = m
            self.tile = tile

        def autotune_init_kwargs(self):
            return {"m": self.m, "tile": self.tile}

        def autotune_key_values(self, *args, **kwargs):
            return {"m": self.m}

        @cutez_module.autotune(
            configs=[
                cutez_module.Config(kwargs={"tile": 16}, name="first"),
                cutez_module.Config(kwargs={"tile": 32}, name="second"),
            ],
            key=["m"],
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    with pytest.raises(
        autotune_module.AutotuneError, match="all autotune candidates failed"
    ) as exc_info:
        cutez_module.compile(Kernel(7, 0), "arg0", stream="stream0")

    message = str(exc_info.value)
    assert "first" in message
    assert "second" in message
    assert "tile 16" in message
    assert "tile 32" in message


def test_compile_does_not_swallow_pre_hook_failures(cutez_module, monkeypatch):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")

    def fake_read_autotune_spec(kernel):
        return autotune_module.get_autotune_spec(kernel)

    def fake_compile(candidate_kernel, *args, **kwargs):
        return f"compiled:{candidate_kernel.tile}"

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )
    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)

    class Kernel:
        def __init__(self, m, tile):
            self.m = m
            self.tile = tile

        def autotune_init_kwargs(self):
            return {"m": self.m, "tile": self.tile}

        def autotune_key_values(self, *args, **kwargs):
            return {"m": self.m}

        @cutez_module.autotune(
            configs=[
                cutez_module.Config(
                    kwargs={"tile": 16},
                    pre_hook=lambda candidate, *a, **k: (_ for _ in ()).throw(
                        RuntimeError("pre hook failed")
                    ),
                ),
                cutez_module.Config(kwargs={"tile": 32}, name="working"),
            ],
            key=["m"],
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    with pytest.raises(RuntimeError, match="pre hook failed") as exc_info:
        cutez_module.compile(Kernel(7, 0), "arg0", stream="stream0")

    assert exc_info.type is RuntimeError


def test_compile_skips_failed_benchmarks_when_another_config_succeeds(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")
    benchmark_calls = []

    def fake_read_autotune_spec(kernel):
        return autotune_module.get_autotune_spec(kernel)

    def fake_compile(candidate_kernel, *args, **kwargs):
        return f"compiled:{candidate_kernel.tile}"

    def failing_do_bench(compiled_kernel, *args, **kwargs):
        benchmark_calls.append(compiled_kernel)
        if compiled_kernel == "compiled:16":
            raise RuntimeError(f"benchmark failed for {compiled_kernel}")
        return 1.0

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )
    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)

    class Kernel:
        def __init__(self, m, tile):
            self.m = m
            self.tile = tile

        def autotune_init_kwargs(self):
            return {"m": self.m, "tile": self.tile}

        def autotune_key_values(self, *args, **kwargs):
            return {"m": self.m}

        @cutez_module.autotune(
            configs=[
                cutez_module.Config(kwargs={"tile": 16}, name="broken_bench"),
                cutez_module.Config(kwargs={"tile": 32}, name="working"),
            ],
            key=["m"],
            do_bench=failing_do_bench,
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    result = cutez_module.compile(Kernel(7, 0), "arg0", stream="stream0")

    assert result == "compiled:32"
    assert benchmark_calls == ["compiled:16", "compiled:32"]


def test_compile_raises_autotune_error_when_all_benchmarks_fail(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")

    def fake_read_autotune_spec(kernel):
        return autotune_module.get_autotune_spec(kernel)

    def fake_compile(candidate_kernel, *args, **kwargs):
        return f"compiled:{candidate_kernel.tile}"

    def failing_do_bench(compiled_kernel, *args, **kwargs):
        raise RuntimeError(f"benchmark failed for {compiled_kernel}")

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )
    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)

    class Kernel:
        def __init__(self, m, tile):
            self.m = m
            self.tile = tile

        def autotune_init_kwargs(self):
            return {"m": self.m, "tile": self.tile}

        def autotune_key_values(self, *args, **kwargs):
            return {"m": self.m}

        @cutez_module.autotune(
            configs=[
                cutez_module.Config(kwargs={"tile": 16}, name="broken_a"),
                cutez_module.Config(kwargs={"tile": 32}, name="broken_b"),
            ],
            key=["m"],
            do_bench=failing_do_bench,
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    with pytest.raises(
        autotune_module.AutotuneError, match="all autotune candidates failed"
    ) as exc_info:
        cutez_module.compile(Kernel(7, 0), "arg0", stream="stream0")

    message = str(exc_info.value)
    assert "broken_a" in message
    assert "broken_b" in message
    assert "benchmark failed for compiled:16" in message
    assert "benchmark failed for compiled:32" in message


def test_compile_raises_clear_error_for_missing_autotune_key(cutez_module, monkeypatch):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")

    def fake_read_autotune_spec(kernel):
        return autotune_module.get_autotune_spec(kernel)

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )

    class Kernel:
        def __init__(self, m, tile):
            self.m = m
            self.tile = tile

        def autotune_init_kwargs(self):
            return {"m": self.m, "tile": self.tile}

        def autotune_key_values(self, *args, **kwargs):
            return {"m": self.m}

        @cutez_module.autotune(
            configs=[cutez_module.Config(kwargs={"tile": 16})], key=["missing_field"]
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    with pytest.raises(
        autotune_module.AutotuneError,
        match="missing autotune key field 'missing_field'",
    ) as exc_info:
        cutez_module.compile(Kernel(7, 0), "arg0", stream="stream0")

    assert "available keys" in str(exc_info.value)


def test_compile_caches_best_config_by_tuning_key(cutez_module, monkeypatch):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")
    compile_calls = []
    benchmark_calls = []

    def fake_read_autotune_spec(kernel):
        return autotune_module.get_autotune_spec(kernel)

    def fake_compile(candidate_kernel, *args, **kwargs):
        compile_calls.append(candidate_kernel.tile)
        return f"compiled:{candidate_kernel.tile}"

    def fake_benchmark(compiled_kernel, *args, **kwargs):
        benchmark_calls.append(compiled_kernel)
        return {"compiled:16": 2.0, "compiled:32": 1.0}[compiled_kernel]

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )
    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(compiler_module, "benchmark", fake_benchmark, raising=False)

    class Kernel:
        def __init__(self, m, tile):
            self.m = m
            self.tile = tile

        def autotune_init_kwargs(self):
            return {"m": self.m, "tile": self.tile}

        def autotune_key_values(self, *args, **kwargs):
            return {"m": self.m}

        @cutez_module.autotune(
            configs=[
                cutez_module.Config(kwargs={"tile": 16}),
                cutez_module.Config(kwargs={"tile": 32}),
            ],
            key=["m"],
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    kernel = Kernel(7, 0)

    first = cutez_module.compile(kernel, "arg0", stream="stream0")
    second = cutez_module.compile(kernel, "arg0", stream="stream0")

    assert first == "compiled:32"
    assert second == "compiled:32"
    assert compile_calls == [16, 32]
    assert benchmark_calls == ["compiled:16", "compiled:32"]


def test_compile_reuses_compiled_candidates_across_tuning_keys(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")
    compile_calls = []
    benchmark_calls = []

    def fake_read_autotune_spec(kernel):
        return autotune_module.get_autotune_spec(kernel)

    def fake_compile(candidate_kernel, *args, **kwargs):
        compile_calls.append((candidate_kernel.m, candidate_kernel.tile))
        return f"compiled:{candidate_kernel.tile}"

    def fake_benchmark(compiled_kernel, *args, **kwargs):
        benchmark_calls.append(compiled_kernel)
        return {"compiled:16": 2.0, "compiled:32": 1.0}[compiled_kernel]

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )
    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(compiler_module, "benchmark", fake_benchmark, raising=False)

    class Kernel:
        def __init__(self, m, tile):
            self.m = m
            self.tile = tile
            self.problem_id = 7

        def autotune_init_kwargs(self):
            return {"m": self.m, "tile": self.tile}

        def autotune_key_values(self, arg0, **kwargs):
            return {"problem_id": self.problem_id}

        @cutez_module.autotune(
            configs=[
                cutez_module.Config(kwargs={"tile": 16}),
                cutez_module.Config(kwargs={"tile": 32}),
            ],
            key=["problem_id"],
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    kernel = Kernel(0, 0)

    first = cutez_module.compile(kernel, "shape", stream="stream0")
    kernel.problem_id = 9
    second = cutez_module.compile(kernel, "shape", stream="stream0")

    assert first == "compiled:32"
    assert second == "compiled:32"
    assert compile_calls == [(0, 16), (0, 32)]
    assert benchmark_calls == [
        "compiled:16",
        "compiled:32",
        "compiled:16",
        "compiled:32",
    ]


def test_compile_cache_distinguishes_positional_compile_inputs(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")
    compile_calls = []

    def fake_read_autotune_spec(kernel):
        return autotune_module.get_autotune_spec(kernel)

    def fake_compile(candidate_kernel, *args, **kwargs):
        compile_calls.append((candidate_kernel.tile, args[0]))
        return f"compiled:{candidate_kernel.tile}:{args[0]}"

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )
    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(
        compiler_module, "benchmark", lambda compiled, *a, **k: 0.0, raising=False
    )

    class Kernel:
        def __init__(self, tile):
            self.tile = tile

        def autotune_init_kwargs(self):
            return {"tile": self.tile}

        def autotune_key_values(self, problem_id, **kwargs):
            return {"problem_id": problem_id}

        @cutez_module.autotune(
            configs=[cutez_module.Config(kwargs={"tile": 16})],
            key=["problem_id"],
            cache_results=False,
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    kernel = Kernel(0)

    first = cutez_module.compile(kernel, "shape_a", stream="stream0")
    second = cutez_module.compile(kernel, "shape_b", stream="stream0")

    assert first == "compiled:16:shape_a"
    assert second == "compiled:16:shape_b"
    assert compile_calls == [(16, "shape_a"), (16, "shape_b")]


def test_compile_cache_distinguishes_configs_with_different_pre_hooks(
    cutez_module, monkeypatch
):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")
    compile_calls = []

    def fake_read_autotune_spec(kernel):
        return autotune_module.get_autotune_spec(kernel)

    def fake_compile(candidate_kernel, *args, **kwargs):
        compile_calls.append(candidate_kernel.marker)
        return f"compiled:{candidate_kernel.marker}"

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )
    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(
        compiler_module, "benchmark", lambda compiled, *a, **k: 0.0, raising=False
    )

    class Kernel:
        def __init__(self, tile, marker="base"):
            self.tile = tile
            self.marker = marker

        def autotune_init_kwargs(self):
            return {"tile": self.tile, "marker": self.marker}

        def autotune_key_values(self, problem_id, **kwargs):
            return {"problem_id": problem_id}

        @cutez_module.autotune(
            configs=[
                cutez_module.Config(
                    kwargs={"tile": 16},
                    pre_hook=lambda candidate, *a, **k: setattr(
                        candidate, "marker", "hook_a"
                    ),
                ),
                cutez_module.Config(
                    kwargs={"tile": 16},
                    pre_hook=lambda candidate, *a, **k: setattr(
                        candidate, "marker", "hook_b"
                    ),
                ),
            ],
            key=["problem_id"],
            cache_results=False,
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    kernel = Kernel(0)

    first = cutez_module.compile(kernel, "shape_a", stream="stream0")
    second = cutez_module.compile(kernel, "shape_b", stream="stream0")

    assert first == "compiled:hook_a"
    assert second == "compiled:hook_a"
    assert compile_calls == ["hook_a", "hook_b", "hook_a", "hook_b"]


def test_compile_cache_accepts_unhashable_positional_inputs(cutez_module, monkeypatch):
    compiler_module = importlib.import_module("cutez.compiler")
    autotune_module = importlib.import_module("cutez.autotune")
    compile_calls = []

    class ShapeArg:
        __hash__ = None

        def __init__(self, label):
            self.label = label

    def fake_read_autotune_spec(kernel):
        return autotune_module.get_autotune_spec(kernel)

    def fake_compile(candidate_kernel, *args, **kwargs):
        compile_calls.append((candidate_kernel.tile, args[0].label))
        return f"compiled:{candidate_kernel.tile}:{args[0].label}"

    monkeypatch.setattr(
        compiler_module, "read_autotune_spec", fake_read_autotune_spec, raising=False
    )
    monkeypatch.setattr(compiler_module.cute, "compile", fake_compile)
    monkeypatch.setattr(
        compiler_module, "benchmark", lambda compiled, *a, **k: 0.0, raising=False
    )

    class Kernel:
        def __init__(self, tile, shape_meta=None):
            self.tile = tile
            self.shape_meta = shape_meta or {"sizes": [1, 2]}

        def autotune_init_kwargs(self):
            return {"tile": self.tile, "shape_meta": self.shape_meta}

        def autotune_key_values(self, problem_id, **kwargs):
            return {"problem_id": problem_id.label}

        @cutez_module.autotune(
            configs=[cutez_module.Config(kwargs={"tile": 16})],
            key=["problem_id"],
            cache_results=False,
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    kernel = Kernel(0)

    first = cutez_module.compile(kernel, ShapeArg("shape_a"), stream="stream0")
    second = cutez_module.compile(kernel, ShapeArg("shape_b"), stream="stream0")

    assert first == "compiled:16:shape_a"
    assert second == "compiled:16:shape_b"
    assert compile_calls == [(16, "shape_a"), (16, "shape_b")]
