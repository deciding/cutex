import cutlass.cute as cute

from .autotune import get_autotune_spec


def compile(kernel, *args, **kwargs):
    get_autotune_spec(kernel)
    return cute.compile(kernel, *args, **kwargs)
