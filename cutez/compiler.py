import cutlass.cute as cute

from .autotune import autotune_spec_applies_to_call, read_autotune_spec
from ._autotune_keys import resolve_autotune_key_values
from .benchmark import benchmark


def _build_candidate_kernel(kernel, candidate_kwargs):
    try:
        return type(kernel)(**candidate_kwargs)
    except TypeError:
        candidate_kernel = type(kernel)()
        for name, value in candidate_kwargs.items():
            setattr(candidate_kernel, name, value)
        return candidate_kernel


def compile(kernel, *args, **kwargs):
    spec = read_autotune_spec(kernel)
    if autotune_spec_applies_to_call(kernel, spec):
        if not hasattr(kernel, "autotune_init_kwargs"):
            raise AttributeError("autotuned kernel must define autotune_init_kwargs()")
        resolve_autotune_key_values(kernel, *args, **kwargs)

        best_compiled = None
        best_time = None
        for config in spec.configs:
            candidate_kwargs = dict(kernel.autotune_init_kwargs())
            candidate_kwargs.update(config.kwargs)
            candidate_kernel = _build_candidate_kernel(kernel, candidate_kwargs)
            compiled = cute.compile(candidate_kernel, *args, **kwargs)
            timed = benchmark(
                compiled, *args, warmup=spec.warmup, rep=spec.rep, **kwargs
            )
            if best_time is None or timed < best_time:
                best_time = timed
                best_compiled = compiled

        return best_compiled
    return cute.compile(kernel, *args, **kwargs)
