import cutlass.cute as cute

from .autotune import (
    AutotuneError,
    autotune_spec_applies_to_call,
    config_identity,
    freeze_for_cache,
    read_autotune_spec,
)
from ._autotune_keys import resolve_autotune_key_values
from .benchmark import benchmark


_COMPILE_CACHE = {}
_BEST_CONFIG_CACHE = {}


def _config_label(config):
    return config.name or repr(dict(config.kwargs))


def _format_candidate_failures(failures):
    lines = ["all autotune candidates failed"]
    for label, exc in failures:
        lines.append(f"- {label}: {type(exc).__name__}: {exc}")
    return "\n".join(lines)


def _tuning_key(kernel, spec, runtime_key_values):
    return (type(kernel), tuple(runtime_key_values[name] for name in spec.key))


def _candidate_kwargs(kernel, runtime_key_values, config):
    candidate_kwargs = dict(kernel.autotune_init_kwargs())
    for name, value in runtime_key_values.items():
        if name in candidate_kwargs:
            candidate_kwargs[name] = value
    candidate_kwargs.update(config.kwargs)
    return candidate_kwargs


def _compile_signature(args, kwargs):
    return (freeze_for_cache(args), freeze_for_cache(kwargs))


def _compile_cache_key(kernel, candidate_kwargs, config, args, kwargs):
    return (
        type(kernel),
        config_identity(config),
        freeze_for_cache(candidate_kwargs),
        _compile_signature(args, kwargs),
    )


def _get_cached_compiled_candidate(kernel, candidate_kwargs, config, args, kwargs):
    cache_key = _compile_cache_key(kernel, candidate_kwargs, config, args, kwargs)
    return _COMPILE_CACHE.get(cache_key), cache_key


def _compile_candidate(candidate_kernel, cache_key, args, kwargs):
    compiled = cute.compile(candidate_kernel, *args, **kwargs)
    _COMPILE_CACHE[cache_key] = compiled
    return compiled


def _reconstruct_candidate(kernel, candidate_kwargs):
    return type(kernel)(**candidate_kwargs)


def compile(kernel, *args, **kwargs):
    spec = read_autotune_spec(kernel)
    if autotune_spec_applies_to_call(kernel, spec):
        if not hasattr(kernel, "autotune_init_kwargs"):
            raise AttributeError("autotuned kernel must define autotune_init_kwargs()")
        runtime_key_values = resolve_autotune_key_values(kernel, *args, **kwargs)
        do_bench = spec.do_bench or benchmark
        tuning_key = _tuning_key(kernel, spec, runtime_key_values)

        cached_best = _BEST_CONFIG_CACHE.get(tuning_key) if spec.cache_results else None
        if cached_best is not None:
            best_config, best_candidate_kwargs = cached_best
            cached_compiled, cache_key = _get_cached_compiled_candidate(
                kernel, best_candidate_kwargs, best_config, args, kwargs
            )
            if cached_compiled is not None:
                return cached_compiled

            candidate_kernel = _reconstruct_candidate(kernel, best_candidate_kwargs)
            if best_config.pre_hook is not None:
                best_config.pre_hook(candidate_kernel, *args, **kwargs)
            return _compile_candidate(candidate_kernel, cache_key, args, kwargs)

        best_compiled = None
        best_time = None
        best_config = None
        best_candidate_kwargs = None
        failures = []
        for config in spec.configs:
            candidate_kwargs = _candidate_kwargs(kernel, runtime_key_values, config)
            cached_compiled, cache_key = _get_cached_compiled_candidate(
                kernel, candidate_kwargs, config, args, kwargs
            )
            if cached_compiled is not None:
                compiled = cached_compiled
            else:
                try:
                    candidate_kernel = _reconstruct_candidate(kernel, candidate_kwargs)
                except Exception as exc:
                    failures.append((_config_label(config), exc))
                    continue

                if config.pre_hook is not None:
                    config.pre_hook(candidate_kernel, *args, **kwargs)

                try:
                    compiled = _compile_candidate(
                        candidate_kernel, cache_key, args, kwargs
                    )
                except Exception as exc:
                    failures.append((_config_label(config), exc))
                    continue
            timed = do_bench(
                compiled, *args, warmup=spec.warmup, rep=spec.rep, **kwargs
            )
            if best_time is None or timed < best_time:
                best_time = timed
                best_compiled = compiled
                best_config = config
                best_candidate_kwargs = candidate_kwargs

        if best_config is None:
            raise AutotuneError(_format_candidate_failures(failures))

        if spec.cache_results and best_config is not None:
            _BEST_CONFIG_CACHE[tuning_key] = (best_config, best_candidate_kwargs)

        return best_compiled
    return cute.compile(kernel, *args, **kwargs)
