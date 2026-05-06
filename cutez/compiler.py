import cutlass.cute as cute


def compile(kernel, *args, **kwargs):
    getattr(getattr(kernel, "__call__", None), "__cutez_autotune__", None)
    return cute.compile(kernel, *args, **kwargs)
