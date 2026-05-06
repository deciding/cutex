import cutlass.cute as cute


def compile(kernel, *args, **kwargs):
    return cute.compile(kernel, *args, **kwargs)
