from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class Config:
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    name: str | None = None
    pre_hook: Callable[..., Any] | None = None


@dataclass(frozen=True)
class AutotuneSpec:
    configs: tuple[Config, ...]
    key: tuple[str, ...]
    warmup: int = 0
    rep: int = 0
    cache_results: bool = True
    do_bench: Callable[..., Any] | None = None


def autotune(
    *,
    configs: list[Config],
    key: list[str],
    warmup: int = 0,
    rep: int = 0,
    cache_results: bool = True,
    do_bench: Callable[..., Any] | None = None,
):
    spec = AutotuneSpec(
        configs=tuple(configs),
        key=tuple(key),
        warmup=warmup,
        rep=rep,
        cache_results=cache_results,
        do_bench=do_bench,
    )

    def decorator(fn):
        setattr(fn, "__cutez_autotune__", spec)
        return fn

    return decorator


def read_autotune_spec(kernel) -> AutotuneSpec | None:
    spec = getattr(kernel, "__cutez_autotune__", None)
    if spec is not None:
        return spec

    call = getattr(kernel, "__call__", None)
    return getattr(call, "__cutez_autotune__", None)


def autotune_spec_applies_to_call(kernel, spec: AutotuneSpec | None) -> bool:
    if spec is None:
        return False
    return not inspect.isfunction(kernel)


def get_autotune_spec(kernel) -> AutotuneSpec | None:
    return read_autotune_spec(kernel)


def config_identity(config: Config) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(config.kwargs.items()))
