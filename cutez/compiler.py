import json

import cutlass.cute as cute

from .autotune import (
    AutotuneError,
    Config,
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


def _log_verbose(kwargs, message):
    if kwargs.get("verbose"):
        print(message)


def _format_candidate_failures(failures):
    lines = ["all autotune candidates failed"]
    for label, exc in failures:
        lines.append(f"- {label}: {type(exc).__name__}: {exc}")
    return "\n".join(lines)


def _tuning_key(kernel, spec, runtime_key_values):
    missing_name = next(
        (name for name in spec.key if name not in runtime_key_values), None
    )
    if missing_name is not None:
        available_keys = ", ".join(sorted(runtime_key_values)) or "none"
        raise AutotuneError(
            f"missing autotune key field '{missing_name}' in resolved key values; "
            f"available keys: {available_keys}"
        )
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


def _persistent_cache_enabled(spec):
    return (
        spec.cache_results
        and spec.cache_path is not None
        and all(config.pre_hook is None for config in spec.configs)
    )


def _stable_kernel_identifier(kernel):
    target = (
        kernel
        if hasattr(kernel, "__module__") and hasattr(kernel, "__qualname__")
        else None
    )
    if target is None:
        target = getattr(type(kernel), "__call__", None)
    if target is None:
        return None

    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if not module or not qualname:
        return None
    return f"{module}.{qualname}"


def _persisted_entry_matches(entry, kernel_id, tuning_values):
    return (
        isinstance(entry, dict)
        and entry.get("kernel") == kernel_id
        and entry.get("key") == list(tuning_values)
    )


def _load_persistent_entries(cache_path):
    try:
        payload = json.loads(cache_path.read_text())
    except FileNotFoundError:
        return [], False
    except (OSError, json.JSONDecodeError):
        return [], True

    entries = payload.get("entries")
    if not isinstance(entries, list):
        return [], False
    return entries, False


def _persisted_config_from_entry(entry):
    config = entry.get("config")
    if not isinstance(config, dict):
        return None

    kwargs = config.get("kwargs")
    if not isinstance(kwargs, dict):
        return None

    name = config.get("name")
    if name is not None and not isinstance(name, str):
        return None

    return Config(kwargs=kwargs, name=name, pre_hook=None)


def _config_is_in_spec(config, spec):
    return any(
        config.kwargs == current.kwargs and config.name == current.name
        for current in spec.configs
    )


def _is_json_serializable(value):
    try:
        json.dumps(value)
    except TypeError:
        return False
    return True


def _load_persisted_best_config(kernel, spec, runtime_key_values, tuning_key):
    if not _persistent_cache_enabled(spec):
        return None, False

    kernel_id = _stable_kernel_identifier(kernel)
    if kernel_id is None:
        return None, False

    tuning_values = tuple(runtime_key_values[name] for name in spec.key)
    entries, had_read_error = _load_persistent_entries(spec.cache_path)
    for entry in entries:
        if not _persisted_entry_matches(entry, kernel_id, tuning_values):
            continue
        config = _persisted_config_from_entry(entry)
        if config is None:
            return None, had_read_error
        if not _config_is_in_spec(config, spec):
            return None, had_read_error
        candidate_kwargs = _candidate_kwargs(kernel, runtime_key_values, config)
        cached_best = (config, candidate_kwargs)
        _BEST_CONFIG_CACHE[tuning_key] = cached_best
        return cached_best, had_read_error
    return None, had_read_error


def _persist_best_config(
    kernel, spec, runtime_key_values, best_config, *, skip_write=False
):
    if not _persistent_cache_enabled(spec):
        return
    if skip_write:
        return

    kernel_id = _stable_kernel_identifier(kernel)
    if kernel_id is None:
        return

    persisted_config = {"kwargs": dict(best_config.kwargs), "name": best_config.name}
    entry = {
        "kernel": kernel_id,
        "key": [runtime_key_values[name] for name in spec.key],
        "config": persisted_config,
    }
    if not _is_json_serializable(entry):
        return

    persisted_entries, had_read_error = _load_persistent_entries(spec.cache_path)
    if had_read_error:
        return

    entries = [
        existing
        for existing in persisted_entries
        if not _persisted_entry_matches(
            existing, kernel_id, tuple(runtime_key_values[name] for name in spec.key)
        )
    ]
    entries.append(entry)

    try:
        spec.cache_path.parent.mkdir(parents=True, exist_ok=True)
        spec.cache_path.write_text(json.dumps({"entries": entries}, indent=2))
    except (OSError, TypeError):
        return


def compile(kernel, *args, **kwargs):
    spec = read_autotune_spec(kernel)
    if autotune_spec_applies_to_call(kernel, spec):
        if not hasattr(kernel, "autotune_init_kwargs"):
            raise AttributeError("autotuned kernel must define autotune_init_kwargs()")
        runtime_key_values = resolve_autotune_key_values(kernel, *args, **kwargs)
        do_bench = spec.do_bench or benchmark
        tuning_key = _tuning_key(kernel, spec, runtime_key_values)

        cached_best = _BEST_CONFIG_CACHE.get(tuning_key) if spec.cache_results else None
        loaded_from_disk = False
        skip_persist_for_call = False
        if cached_best is None:
            cached_best, skip_persist_for_call = _load_persisted_best_config(
                kernel, spec, runtime_key_values, tuning_key
            )
            loaded_from_disk = cached_best is not None
        if cached_best is not None:
            best_config, best_candidate_kwargs = cached_best
            cached_compiled, cache_key = _get_cached_compiled_candidate(
                kernel, best_candidate_kwargs, best_config, args, kwargs
            )
            if cached_compiled is not None:
                if loaded_from_disk:
                    _log_verbose(kwargs, "disk-cache-hit")
                else:
                    _log_verbose(kwargs, "cache-hit")
                return cached_compiled

            candidate_kernel = _reconstruct_candidate(kernel, best_candidate_kwargs)
            if best_config.pre_hook is not None:
                best_config.pre_hook(candidate_kernel, *args, **kwargs)
            if loaded_from_disk:
                _log_verbose(kwargs, "disk-cache-hit")
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
            try:
                timed = do_bench(
                    compiled, *args, warmup=spec.warmup, rep=spec.rep, **kwargs
                )
            except Exception as exc:
                failures.append((_config_label(config), exc))
                continue
            if best_time is None or timed < best_time:
                best_time = timed
                best_compiled = compiled
                best_config = config
                best_candidate_kwargs = candidate_kwargs

        if best_config is None:
            raise AutotuneError(_format_candidate_failures(failures))

        if spec.cache_results and best_config is not None:
            _BEST_CONFIG_CACHE[tuning_key] = (best_config, best_candidate_kwargs)
            _persist_best_config(
                kernel,
                spec,
                runtime_key_values,
                best_config,
                skip_write=skip_persist_for_call,
            )

        return best_compiled
    return cute.compile(kernel, *args, **kwargs)
