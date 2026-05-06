import cutlass.cute as cute

from .autotune import read_autotune_spec


def compile(kernel, *args, **kwargs):
    read_autotune_spec(kernel)
    return cute.compile(kernel, *args, **kwargs)
