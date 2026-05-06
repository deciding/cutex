import cutlass.cute as cute

from .autotune import autotune_spec_applies_to_call, read_autotune_spec
from ._autotune_keys import resolve_autotune_key_values


def compile(kernel, *args, **kwargs):
    spec = read_autotune_spec(kernel)
    if autotune_spec_applies_to_call(kernel, spec):
        if not hasattr(kernel, "autotune_init_kwargs"):
            raise AttributeError("autotuned kernel must define autotune_init_kwargs()")
        resolve_autotune_key_values(kernel, *args, **kwargs)
    return cute.compile(kernel, *args, **kwargs)
