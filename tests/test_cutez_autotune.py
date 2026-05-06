from pathlib import Path
import importlib
import sys

import pytest


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
        @cutez_module.autotune(
            configs=[cutez_module.Config(kwargs={"tile": 64})], key=["m"]
        )
        def __call__(self, *args, **kwargs):
            return args, kwargs

    kernel = Kernel()

    result = cutez_module.compile(kernel, "arg0", stream="stream0")

    assert result == "compiled"
    assert len(read_calls) == 1
    assert read_calls[0][0] is kernel
    assert read_calls[0][1] is not None
    assert read_calls[0][1].key == ("m",)
    assert calls == [((kernel, "arg0"), {"stream": "stream0"})]
