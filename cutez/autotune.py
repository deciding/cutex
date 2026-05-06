from dataclasses import dataclass, field
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
