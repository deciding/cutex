def benchmark(fn, *args, warmup=0, rep=0, **kwargs):
    if not callable(fn):
        return fn

    for _ in range(warmup):
        fn(*args, **kwargs)

    best = None
    for _ in range(rep or 1):
        result = fn(*args, **kwargs)
        if best is None or result < best:
            best = result

    return best
